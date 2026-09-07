"""Chain composition: LLMMathChain.

A specialized chain: the LLM translates a word problem into a single-line
Python expression, then the chain evaluates it with numexpr instead of
trusting the model to do arithmetic itself. Requires the LLM to follow an
exact ```text ...``` output format, which a small model like flan-t5-base
often can't — handled explicitly rather than assumed to succeed.
"""

from langchain.chains import LLMMathChain
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
QUESTION = "What is 37593 times 67?"

print_step(1, "Build the math chain")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
chain = LLMMathChain.from_llm(llm=llm)

print_step(2, "Invoke: LLM writes the expression, numexpr evaluates it")
print(f"Question: {QUESTION}")
try:
    result = chain.invoke({"question": QUESTION})
    print(f"Answer: {result['answer']}")
except Exception as e:
    print(f"Failed ({type(e).__name__}): the model didn't follow the expected ```text``` format.")
    print("This is a known limitation of small models like flan-t5-base — a larger, "
          "instruction-tuned model follows this kind of format far more reliably.")
