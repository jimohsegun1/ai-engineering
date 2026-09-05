"""LCEL chain composition: a single-step chain.

The simplest possible chain: prompt | llm | output_parser, piped together
with LangChain Expression Language's `|` operator. Every other file in this
folder builds on this same three-piece shape. Runs on a local, free Hugging
Face model, same as rag_pipeline_huggingface.py.
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
PHRASE = "Where is the nearest train station?"

print_step(1, "Build the chain (prompt | llm | output_parser)")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
prompt = PromptTemplate.from_template("Translate this English phrase to French: {phrase}")
chain = prompt | llm | StrOutputParser()
print(f"Loaded model: {CHAT_MODEL}")

print_step(2, "Invoke the chain")
result = chain.invoke({"phrase": PHRASE})
print(f"Input:  {PHRASE!r}")
print(f"Output: {result!r}")
