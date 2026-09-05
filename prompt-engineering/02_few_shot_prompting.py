"""Prompt engineering: few-shot prompting.

A handful of input/output examples are shown before the real question, so
the model can copy the pattern (task and output format) instead of guessing
it from an instruction alone. Compare with 01_zero_shot_prompting.py, which
asks the same kind of question with no examples at all.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
REVIEW = "The battery life on this laptop is incredible, it lasts all day."

EXAMPLES = [
    {"review": "I love how fast this phone charges.", "sentiment": "Positive"},
    {"review": "The screen cracked after one week of light use.", "sentiment": "Negative"},
    {"review": "It arrived on time, nothing special about it.", "sentiment": "Neutral"},
]

print_step(1, "Build a few-shot prompt")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 20},
)
example_prompt = PromptTemplate.from_template("Review: {review}\nSentiment: {sentiment}")
few_shot_prompt = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=example_prompt,
    prefix="Classify the sentiment of each review as Positive, Negative, or Neutral.",
    suffix="Review: {review}\nSentiment:",
    input_variables=["review"],
)
print(few_shot_prompt.format(review=REVIEW))

chain = few_shot_prompt | llm | StrOutputParser()

print_step(2, "Invoke with the examples shown above")
result = chain.invoke({"review": REVIEW})
print(f"Review: {REVIEW!r}")
print(f"Sentiment: {result!r}")
