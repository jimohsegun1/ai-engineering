"""Prompt engineering: structured output prompting.

Instead of free-text, the prompt asks for a response in a specific,
machine-parseable format (JSON matching a schema), so it can be validated
and consumed directly by code. PydanticOutputParser generates the format
instructions from the schema and parses the raw response back into an
object — but a small model like flan-t5-base often can't follow the format
reliably, so parsing failures are handled explicitly rather than assumed away.
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


class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="One of: Positive, Negative, Neutral")
    one_word_summary: str = Field(description="A single word summarizing the review")


CHAT_MODEL = "google/flan-t5-base"
REVIEW = "The battery life on this laptop is incredible, it lasts all day."

print_step(1, "Build the parser and inject its format instructions into the prompt")
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
prompt = PromptTemplate(
    template="Analyze this review.\n{format_instructions}\n\nReview: {review}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
print(parser.get_format_instructions())

print_step(2, "Get the raw response, then try to parse it into the schema")
raw_chain = prompt | llm | StrOutputParser()
raw_output = raw_chain.invoke({"review": REVIEW})
print(f"Raw model output:\n  {raw_output}")

try:
    parsed = parser.parse(raw_output)
    print(f"\nParsed successfully: {parsed}")
except Exception as e:
    print(f"\nParsing failed ({type(e).__name__}): the model's output didn't match the schema.")
    print("This is a known limitation of small models like flan-t5-base — a larger, "
          "instruction-tuned model follows structured-output formats far more reliably.")
