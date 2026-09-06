"""LCEL chain composition: a plain-function transform step.

RunnableLambda wraps an ordinary Python function so it can sit inside a
chain like any other step — a chain step doesn't have to call an LLM at all,
it can just reshape or clean up data on its way to the next step.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
MAX_CHARS = 60
RAW_TEXT = "   WHAT IS      the boiling point of water   in Celsius?   \n"


def clean_text(text: str) -> str:
    return " ".join(text.split()).lower()[:MAX_CHARS]


print_step(1, "Build the transform step (RunnableLambda)")
transform = RunnableLambda(lambda inputs: {"question": clean_text(inputs["question"])})
print(f"Raw input:     {RAW_TEXT!r}")
print(f"After cleanup: {clean_text(RAW_TEXT)!r}")

print_step(2, "Chain it before the prompt")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
prompt = PromptTemplate.from_template("Answer this question concisely:\n\n{question}")
chain = transform | prompt | llm | StrOutputParser()

answer = chain.invoke({"question": RAW_TEXT})
print(f"Answer: {answer}")
