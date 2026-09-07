"""Chain composition: the "refine" documents chain.

Processes documents one at a time in sequence, refining a running answer
with each new document instead of summarizing them independently like
07_map_reduce_chain.py does. Each step only ever sees the previous running
answer plus one new document, never the whole set at once.
"""

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
DOCS = [
    Document(page_content="Machine learning is a field of AI where models learn patterns from data instead of following hardcoded rules."),
    Document(page_content="Deep learning uses neural networks with many layers and has driven most recent AI progress."),
    Document(page_content="Reinforcement learning trains an agent to make decisions by rewarding good actions and penalizing bad ones."),
]

print_step(1, "Build the refine summarization chain")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
chain = load_summarize_chain(llm, chain_type="refine")

print_step(2, "Refine the running summary one document at a time")
for i, doc in enumerate(DOCS, start=1):
    print(f"  Document {i}: {doc.page_content}")

result = chain.invoke({"input_documents": DOCS})
print(f"\nFinal (refined) summary: {result['output_text']}")
