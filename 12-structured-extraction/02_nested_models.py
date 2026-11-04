"""Structured extraction: nested Pydantic models.

Real-world entities rarely fit a flat key-value schema. An invoice has a
vendor *and* a list of line items; a meeting note has an attendee *list*,
each attendee with their own fields. This file shows how Pydantic handles
nesting (a model that contains another model) and lists, and how
PydanticOutputParser generates — and parses — that more complex shape.

Runs on google/flan-t5-base (local, free, no API key). Because the model
is small it may not fully populate every nested field; the raw output is
printed before the parse so you can see exactly what it returned.
"""

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# ---------------------------------------------------------------------------
# Nested schema: Address embedded inside a Company
# ---------------------------------------------------------------------------

class Address(BaseModel):
    """A physical mailing address."""
    street: str = Field(description="Street name and number")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class Company(BaseModel):
    """A company extracted from a passage."""
    name: str = Field(description="Official company name")
    industry: str = Field(description="Industry sector the company operates in")
    founded: int = Field(description="Year the company was founded as an integer")
    headquarters: Address = Field(description="Physical headquarters address")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAT_MODEL = "google/flan-t5-base"

TEXT = (
    "Meridian Technologies was founded in 2011 and operates in the renewable "
    "energy sector. The company is headquartered at 44 Solar Lane, Austin, "
    "United States, and has grown to over 2 000 employees worldwide."
)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

print_step(1, "Show the auto-generated format instructions for the nested schema")
parser = PydanticOutputParser(pydantic_object=Company)
print(parser.get_format_instructions())

print_step(2, "Load the model and build the extraction chain")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)
prompt = PromptTemplate(
    template=(
        "Extract the company information from the text below.\n"
        "{format_instructions}\n\n"
        "Text: {text}\n\n"
        "JSON:"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
raw_chain = prompt | llm | StrOutputParser()

print_step(3, "Run the chain and inspect the raw output")
print(f"Input text:\n  {TEXT}\n")
raw_output = raw_chain.invoke({"text": TEXT})
print(f"Raw model output:\n  {raw_output!r}")

print_step(4, "Parse into a typed Company (with nested Address)")
try:
    company = parser.parse(raw_output)
    print("Parsed successfully!")
    print(f"  name:        {company.name!r}")
    print(f"  industry:    {company.industry!r}")
    print(f"  founded:     {company.founded}")
    print(f"  headquarters:")
    print(f"    street:  {company.headquarters.street!r}")
    print(f"    city:    {company.headquarters.city!r}")
    print(f"    country: {company.headquarters.country!r}")
except Exception as e:
    print(
        f"Parsing failed ({type(e).__name__}): nested schemas add extra JSON depth "
        "that small models struggle to reproduce reliably.\n"
        "A larger instruction-tuned model handles nesting far more consistently."
    )
    print(f"\nRaw output was: {raw_output!r}")
