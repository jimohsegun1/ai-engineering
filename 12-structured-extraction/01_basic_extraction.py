"""Structured extraction: a single Pydantic model from free-form text.

The simplest form of structured extraction: define a Pydantic schema and
ask the LLM to fill it from a block of text, rather than returning a
free-text answer. PydanticOutputParser generates the format instructions
automatically and parses the raw output back into a typed object.

Because flan-t5-base is a small model it frequently mis-formats JSON, so
failed parses are caught and shown explicitly — that failure itself is part
of the lesson. A larger model (Ollama, Qorebit) would rarely need the
fallback path.

Runs on the same free local google/flan-t5-base model used across this
project — no API key needed.
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
# Schema
# ---------------------------------------------------------------------------

class Person(BaseModel):
    """Structured representation of a person mentioned in a passage."""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age of the person as an integer")
    occupation: str = Field(description="Job title or role of the person")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAT_MODEL = "google/flan-t5-base"

TEXT = (
    "Dr. Sarah Chen, a 38-year-old neuroscientist at the city hospital, "
    "has spent the last decade researching how sleep deprivation affects "
    "memory consolidation in adults."
)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

print_step(1, "Build the parser and show its auto-generated format instructions")
parser = PydanticOutputParser(pydantic_object=Person)
print(parser.get_format_instructions())

print_step(2, "Load the model and build the extraction chain")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 150},
)
prompt = PromptTemplate(
    template=(
        "Extract the person's details from the text below.\n"
        "{format_instructions}\n\n"
        "Text: {text}\n\n"
        "JSON:"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
# Two chains: one returns raw text (for inspection), one tries to parse it.
raw_chain = prompt | llm | StrOutputParser()

print_step(3, "Run the chain and inspect the raw output")
print(f"Input text:\n  {TEXT}\n")
raw_output = raw_chain.invoke({"text": TEXT})
print(f"Raw model output:\n  {raw_output!r}")

print_step(4, "Parse the raw output into a typed Person object")
try:
    person = parser.parse(raw_output)
    print(f"Parsed successfully!")
    print(f"  name:       {person.name!r}")
    print(f"  age:        {person.age}")
    print(f"  occupation: {person.occupation!r}")
except Exception as e:
    print(
        f"Parsing failed ({type(e).__name__}): the model's output didn't match the schema.\n"
        "This is expected for small models like flan-t5-base. A larger instruction-tuned\n"
        "model (Ollama llama3.2, Qorebit) produces valid JSON almost every time."
    )
    print(f"\nRaw output was: {raw_output!r}")
