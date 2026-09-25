"""Guardrails: validate structured output, retry on failure instead of giving up.

`02-prompt-engineering/05_structured_output_prompting.py` shows the problem:
a small model often doesn't follow a requested JSON schema, and that demo
just catches the parsing error and stops there. A guardrail does one more
thing: when validation fails, it feeds the parser's error back to the LLM
and asks it to fix its own output, instead of surfacing a broken result to
whatever code called it.

`OutputFixingParser` wraps a normal parser with exactly this retry loop.
This file runs the naive parser and the fixing parser side by side on the
same set of inputs, so you can see how many parsing failures the retry
actually recovers.

Uses Ollama (`llama3.2:3b`) rather than `flan-t5-base` — same reason
`06-agents/` does: flan-t5-base doesn't just format its output wrong here,
it barely attempts the schema at all (empty or one-word raw output), which
leaves the retry nothing recoverable to work with. An instruction-tuned
model at least attempts the format, so occasional formatting slips are
exactly the near-miss case a fixing parser is built to repair.
"""

from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="One of: Positive, Negative, Neutral")
    one_word_summary: str = Field(description="A single word summarizing the review")


OLLAMA_MODEL = "llama3.2:3b"
REVIEWS = [
    "The battery life on this laptop is incredible, it lasts all day.",
    "Shipping took three weeks and the box arrived crushed.",
    "It's fine. Does what it says, nothing more.",
    "Customer support resolved my issue in minutes, very impressed.",
    "The screen cracked the first time I dropped it, cheaply made.",
]


print_step(1, "Build the naive parser, the LLM, and the fixing (retry) parser")
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
prompt = PromptTemplate(
    template="Analyze this review.\n{format_instructions}\n\nReview: {review}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
raw_chain = prompt | llm | StrOutputParser()
# max_retries=1: on a parsing failure, ask the LLM once to fix the output,
# showing it the schema and the error message from the failed parse.
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm, max_retries=1)


print_step(2, f"Run {len(REVIEWS)} reviews through both parsers")
naive_successes = 0
retry_successes = 0
for review in REVIEWS:
    raw_output = raw_chain.invoke({"review": review})
    print(f"\nReview: {review}")
    print(f"Raw model output: {raw_output}")

    try:
        parsed = parser.parse(raw_output)
        print(f"[NAIVE PASS] {parsed}")
        naive_successes += 1
        retry_successes += 1  # already valid, nothing to retry
        continue
    except Exception as e:
        print(f"[NAIVE FAIL] {type(e).__name__}")

    try:
        fixed = fixing_parser.parse(raw_output)
        print(f"[RETRY PASS] {fixed}")
        retry_successes += 1
    except Exception as e:
        print(f"[RETRY FAIL] {type(e).__name__}: still invalid after one fix attempt")


print_step(3, "Summary")
total = len(REVIEWS)
print(f"Naive parser:  {naive_successes}/{total} valid")
print(f"With retry:    {retry_successes}/{total} valid")
print(
    "A guardrail like this trades latency and an extra LLM call for a higher "
    "chance of getting usable structured output - it doesn't guarantee success, "
    "but it recovers some failures a naive parse would've just surfaced as an "
    "error."
)
