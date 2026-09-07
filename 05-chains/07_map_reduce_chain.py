"""Chain composition: the "map-reduce" documents chain.

Summarizes each document separately (the "map" step, all independent so
they could run in parallel), then combines those partial summaries into one
final summary (the "reduce" step) — a way to handle more documents than
would ever fit stuffed into a single prompt at once.
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

print_step(1, "Build the map-reduce summarization chain")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
chain = load_summarize_chain(llm, chain_type="map_reduce")

print_step(2, "Map: summarize each document independently")
for i, doc in enumerate(DOCS, start=1):
    print(f"  Document {i}: {doc.page_content}")

print_step(3, "Reduce: combine the per-document summaries into one")
result = chain.invoke({"input_documents": DOCS})
print(f"Final summary: {result['output_text']}")
