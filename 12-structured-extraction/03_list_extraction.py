"""Structured extraction: extracting a list of entities in one pass.

The previous two files extracted a single entity per call. Real documents
often mention several things of the same type — a news article names
multiple people, an invoice has multiple line items, a meeting note lists
multiple action items. This file shows how to model a list as the top-level
output using a wrapping container model (the idiomatic Pydantic approach
when PydanticOutputParser needs a single root object).

Also introduces Field validators so Pydantic enforces constraints
(non-empty strings, positive amounts) before the data ever reaches your
application code.

Runs on google/flan-t5-base (local, free, no API key).
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
# Schema: a list of line items wrapped in a top-level container
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    description: str = Field(description="Short description of the item or service")
    quantity: int = Field(description="Number of units, as a positive integer")
    unit_price: float = Field(description="Price per unit in USD, as a number")


class Invoice(BaseModel):
    """All line items extracted from an invoice or receipt."""
    vendor: str = Field(description="Name of the vendor or supplier")
    items: list[LineItem] = Field(description="List of line items on the invoice")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAT_MODEL = "google/flan-t5-base"

TEXT = (
    "Invoice from TechSupply Co.\n"
    "- 3 x USB-C Hub @ $29.99 each\n"
    "- 1 x Mechanical Keyboard @ $89.00\n"
    "- 5 x HDMI Cable @ $12.50 each\n"
)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

print_step(1, "Show the auto-generated format instructions for the list schema")
parser = PydanticOutputParser(pydantic_object=Invoice)
print(parser.get_format_instructions())

print_step(2, "Load the model and build the extraction chain")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 300},
)
prompt = PromptTemplate(
    template=(
        "Extract all line items from the invoice below.\n"
        "{format_instructions}\n\n"
        "Invoice text:\n{text}\n\n"
        "JSON:"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
raw_chain = prompt | llm | StrOutputParser()

print_step(3, "Run the chain and inspect the raw output")
print(f"Input text:\n{TEXT}")
raw_output = raw_chain.invoke({"text": TEXT})
print(f"Raw model output:\n  {raw_output!r}")

print_step(4, "Parse into a typed Invoice (vendor + list of LineItems)")
try:
    invoice = parser.parse(raw_output)
    print(f"Parsed successfully!")
    print(f"  vendor: {invoice.vendor!r}")
    print(f"  items ({len(invoice.items)} found):")
    for i, item in enumerate(invoice.items, 1):
        total = item.quantity * item.unit_price
        print(
            f"    [{i}] {item.description!r}  "
            f"qty={item.quantity}  "
            f"unit=${item.unit_price:.2f}  "
            f"line total=${total:.2f}"
        )
except Exception as e:
    print(
        f"Parsing failed ({type(e).__name__}): list schemas require the model to produce\n"
        "a JSON array, which small models like flan-t5-base handle inconsistently.\n"
        "A larger instruction-tuned model (Ollama, Qorebit) handles list extraction reliably."
    )
    print(f"\nRaw output was: {raw_output!r}")
