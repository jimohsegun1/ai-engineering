"""Prompt engineering: zero-shot prompting.

Just an instruction, no examples of the desired output — the model has to
infer the task and format entirely from the wording of the prompt itself.
Runs on a local, free Hugging Face model, same as rag_pipeline_huggingface.py.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
REVIEW = "The battery life on this laptop is incredible, it lasts all day."

print_step(1, "Build a zero-shot prompt")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 20},
)
prompt = PromptTemplate.from_template(
    "Classify the sentiment of this review as Positive, Negative, or Neutral: {review}"
)
chain = prompt | llm | StrOutputParser()

print_step(2, "Invoke with no examples given")
result = chain.invoke({"review": REVIEW})
print(f"Review: {REVIEW!r}")
print(f"Sentiment: {result!r}")
