"""Guardrails: validate structured output, retry on failure instead of giving up.

`02-prompt-engineering/05_structured_output_prompting.py` shows the problem:
a small model often doesn't follow a requested JSON schema, and that demo
just catches the parsing error and stops there. A guardrail does one more
thing: when validation fails, it feeds the parser's error back to the LLM
and asks it to fix its own output, instead of surfacing a broken result to
whatever code called it. `OutputFixingParser` wraps a normal parser with
exactly this retry loop.

Uses Ollama (`llama3.2:3b`) rather than `flan-t5-base` — flan-t5-base barely
attempted the schema at all here (empty or one-word raw output), leaving
the retry nothing recoverable to work with.

This file runs the SAME retry-guarded parser against two different prompts,
because testing turned up something worth keeping rather than prompting
away: with only `parser.get_format_instructions()` (the bare JSON schema,
no filled example) in the prompt, llama3.2:3b consistently echoed the
*schema itself* back — the literal field descriptions and types — instead
of a filled instance, on every single review. `OutputFixingParser`'s retry
did not recover a single one of those failures, because its repair prompt
shows the model the same confusing schema again. Adding one concrete
filled-in example to the prompt fixed the problem outright. The two runs
below are shown side by side so you see both the failure and the fix,
instead of only the fixed version.
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


print_step(1, "Build the parser, the LLM, and the fixing (retry) parser")
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
# max_retries=1: on a parsing failure, ask the LLM once to fix the output,
# showing it the schema and the error message from the failed parse.
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm, max_retries=1)

BARE_PROMPT = PromptTemplate(
    template="Analyze this review.\n{format_instructions}\n\nReview: {review}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
GUIDED_PROMPT = PromptTemplate(
    template=(
        "Analyze this review. Respond with ONLY a JSON object filled in with "
        "real values, no other text, following this example:\n"
        '{{"sentiment": "Positive", "one_word_summary": "reliable"}}\n\n'
        "{format_instructions}\n\n"
        "Review: {review}"
    ),
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def run_variant(name: str, prompt: PromptTemplate) -> tuple[int, int]:
    chain = prompt | llm | StrOutputParser()
    naive_successes = 0
    retry_successes = 0
    for review in REVIEWS:
        raw_output = chain.invoke({"review": review})
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

    print(f"\n{name}: naive {naive_successes}/{len(REVIEWS)}, with retry {retry_successes}/{len(REVIEWS)}")
    return naive_successes, retry_successes


print_step(2, f"Variant A: bare schema, no filled example ({len(REVIEWS)} reviews)")
bare_naive, bare_retry = run_variant("Bare schema", BARE_PROMPT)

print_step(3, f"Variant B: schema + one filled example ({len(REVIEWS)} reviews)")
guided_naive, guided_retry = run_variant("Schema + example", GUIDED_PROMPT)


print_step(4, "Summary")
total = len(REVIEWS)
print(f"Bare schema:        naive {bare_naive}/{total}, with retry {bare_retry}/{total}")
print(f"Schema + example:   naive {guided_naive}/{total}, with retry {guided_retry}/{total}")
print(
    "\nThe retry guardrail didn't rescue the bare-schema prompt at all here: "
    "llama3.2:3b wasn't making small formatting mistakes, it was echoing the "
    "abstract schema back instead of filling it in, on every review - and "
    "OutputFixingParser's repair prompt shows the model that same confusing "
    "schema again, so it fails the same way. One filled example in the "
    "prompt fixed it completely, with no retry needed at all.\n"
    "\n"
    "The lesson: a retry guardrail is a safety net for near-miss formatting "
    "errors, not a fix for a model that has misunderstood the task. Good "
    "prompting prevents more failures than automated repair recovers."
)
