"""Prompt engineering: chain-of-thought prompting.

Asking the model to reason step by step before giving a final answer, versus
asking it to jump straight to the answer — compares both on the same
question so you can see whether spelling out the reasoning steps helps.
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
QUESTION = "A train travels 60 miles in 2 hours, then 90 miles in 3 hours. What is its average speed for the whole trip?"

print_step(1, "Build a direct-answer prompt and a chain-of-thought prompt")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 150},
)
direct_prompt = PromptTemplate.from_template("Answer with just the final number: {question}")
cot_prompt = PromptTemplate.from_template(
    "{question}\nLet's think step by step, then give the final answer."
)

direct_chain = direct_prompt | llm | StrOutputParser()
cot_chain = cot_prompt | llm | StrOutputParser()

print_step(2, "Direct answer (no reasoning steps)")
direct_answer = direct_chain.invoke({"question": QUESTION})
print(f"Question: {QUESTION}")
print(f"Answer: {direct_answer}")

print_step(3, "Chain-of-thought answer (reasoning steps requested)")
cot_answer = cot_chain.invoke({"question": QUESTION})
print(f"Question: {QUESTION}")
print(f"Answer: {cot_answer}")
