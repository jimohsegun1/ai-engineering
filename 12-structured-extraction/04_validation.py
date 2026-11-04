"""Structured extraction: multi-field validation with Pydantic.

Parsers only catch format errors (invalid JSON, wrong types). They do not
catch semantic errors like a negative price or an empty name that
technically parses fine. Pydantic validators close that gap: they run
automatically after parsing and raise ValidationError if any constraint
is violated, keeping bad data out of your application before it causes
problems downstream.

This file shows three common validator patterns:
  1. field_validator  — constraint on a single field value
  2. model_validator  — cross-field constraint (one field depends on another)
  3. Literal type     — restricts a string field to a fixed set of values

Runs on google/flan-t5-base (local, free, no API key).
"""

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# ---------------------------------------------------------------------------
# Schema with validators
# ---------------------------------------------------------------------------

class JobPosting(BaseModel):
    """A validated job posting extracted from raw text."""

    title: str = Field(description="Job title")
    company: str = Field(description="Hiring company name")
    salary_min: int = Field(description="Minimum annual salary in USD as an integer")
    salary_max: int = Field(description="Maximum annual salary in USD as an integer")
    employment_type: Literal["full-time", "part-time", "contract", "internship"] = Field(
        description="Employment type: one of full-time, part-time, contract, internship"
    )

    # --- Field-level validator ---
    @field_validator("title", "company")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty or whitespace")
        return v.strip()

    @field_validator("salary_min", "salary_max")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("salary must be a positive integer")
        return v

    # --- Cross-field validator ---
    @model_validator(mode="after")
    def min_must_not_exceed_max(self) -> "JobPosting":
        if self.salary_min > self.salary_max:
            raise ValueError(
                f"salary_min ({self.salary_min}) must not exceed "
                f"salary_max ({self.salary_max})"
            )
        return self


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAT_MODEL = "google/flan-t5-base"

# A valid posting — expected to parse and validate cleanly (model permitting).
VALID_TEXT = (
    "We're hiring a Senior Data Engineer at DataFlow Inc. "
    "This is a full-time role with a salary range of $130,000 to $160,000 per year."
)

# A deliberately broken posting — salary_min > salary_max — to show the
# model-validator catching a semantic error that JSON parsing alone would miss.
INVALID_TEXT = (
    "Join NovaTech as a contract Backend Developer. "
    "Salary range: $200,000 minimum, $150,000 maximum."  # inverted — intentionally wrong
)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

print_step(1, "Show the format instructions (notice the Literal constraint)")
parser = PydanticOutputParser(pydantic_object=JobPosting)
print(parser.get_format_instructions())

print_step(2, "Load the model and build the extraction chain")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)
prompt = PromptTemplate(
    template=(
        "Extract the job posting details from the text below.\n"
        "{format_instructions}\n\n"
        "Text: {text}\n\n"
        "JSON:"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
raw_chain = prompt | llm | StrOutputParser()


def extract_and_validate(label: str, text: str) -> None:
    print(f"\n--- {label} ---")
    print(f"Input: {text!r}")
    raw = raw_chain.invoke({"text": text})
    print(f"Raw output: {raw!r}")
    try:
        posting = parser.parse(raw)
        print(f"Parsed + validated successfully!")
        print(f"  title:           {posting.title!r}")
        print(f"  company:         {posting.company!r}")
        print(f"  employment_type: {posting.employment_type!r}")
        print(f"  salary range:    ${posting.salary_min:,} – ${posting.salary_max:,}")
    except Exception as e:
        print(f"Failed ({type(e).__name__}): {e}")


print_step(3, "Extract from valid text — expect success")
extract_and_validate("VALID posting", VALID_TEXT)

print_step(4, "Extract from invalid text — expect validator rejection")
extract_and_validate("INVALID posting (inverted salary range)", INVALID_TEXT)

print_step(5, "Why this matters")
print(
    "Pydantic validators fire after the JSON parses successfully, catching semantic\n"
    "errors your own code would otherwise have to check for manually everywhere.\n"
    "The extraction pipeline produces a typed, validated object or raises — never\n"
    "a dict of unchecked strings that silently propagates bad data."
)
