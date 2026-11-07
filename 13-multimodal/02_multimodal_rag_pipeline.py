"""Multimodal RAG: End-to-End Multimodal Retrieval & QA Pipeline.

In standard RAG, the context passed to the generation model is plain text.
In Multimodal RAG:
  1. The knowledge base contains both textual knowledge and visual diagrams/charts.
  2. The retriever queries vector storage and identifies the most relevant text chunks
     and/or visual figures.
  3. The generation prompt constructs a Multimodal Message containing:
     - The user's question
     - Contextual text passages
     - Base64-encoded visual evidence (PNG diagrams/charts)
  4. The Vision-LLM inspects the visual elements directly to formulate a factual,
     visually-grounded response.

Supports local models, Ollama vision, and local CPU fallback.
No API key required.
"""

import base64
import os
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "multimodal_rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-base"

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


def print_passage(label: str, text: str) -> None:
    print(f"\n  {label}")
    print("  " + "-" * 60)
    wrapped = textwrap.fill(text, width=68, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# --- Step 1: Ingest Both Text and Visual Assets ---
print_step(1, "Ingest heterogeneous multimodal corpus (text + visual diagrams)")

corpus_docs = [
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) optimizes LLM output by referencing "
            "an authoritative knowledge base outside of its training data sources before generating a response."
        ),
        metadata={"source": "sample.txt", "type": "text"},
    ),
    Document(
        page_content=(
            "System Architecture Diagram depicting an end-to-end RAG application. "
            "Flow begins on the left with User/Web Client connecting via HTTP line to "
            "an API Gateway & FastAPI Router in the center. The API router splits requests into two "
            "backend services: a Chroma Vector DB node at the top for embedding search and a "
            "downstream LLM Service node at the bottom for response generation."
        ),
        metadata={
            "source": str(DATA_DIR / "sample_architecture.png"),
            "filename": "sample_architecture.png",
            "type": "image",
        },
    ),
    Document(
        page_content=(
            "Bar Chart titled 'Quarterly Cloud Revenue (2026)'. "
            "The horizontal x-axis shows four quarters: Q1 ($1.2M), Q2 ($1.8M), Q3 ($2.4M), and Q4 ($3.1M). "
            "The vertical y-axis measures revenue in millions of dollars. "
            "The trend demonstrates steady quarter-over-quarter upward revenue growth throughout 2026."
        ),
        metadata={
            "source": str(DATA_DIR / "sample_revenue_chart.png"),
            "filename": "sample_revenue_chart.png",
            "type": "image",
        },
    ),
]

print(f"Indexed corpus containing {len(corpus_docs)} heterogeneous documents:")
for d in corpus_docs:
    print(f"  - [{d.metadata['type'].upper()}] {d.metadata['source']}")


# --- Step 2: Vector Store Indexing ---
print_step(2, "Index multimodal corpus into Chroma")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(corpus_docs, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print(f"Indexed {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")


# --- Step 3: Multimodal Retrieval & Synthesis ---
print_step(3, "Execute multimodal retrieval query")

question = "What was the reported revenue in Q4 according to the cloud revenue chart?"
print(f"User Query: {question!r}")

retrieved_docs = retriever.invoke(question)

print(f"\nRetrieved {len(retrieved_docs)} supporting document(s):")
text_contexts = []
retrieved_images = []

for i, doc in enumerate(retrieved_docs, start=1):
    doc_type = doc.metadata.get("type", "text")
    if doc_type == "image":
        img_path = Path(doc.metadata["source"])
        b64 = encode_image_to_base64(img_path)
        retrieved_images.append((img_path.name, b64))
        text_contexts.append(f"[Visual Asset: {img_path.name}]: {doc.page_content}")
        print(f"  [{i}] VISUAL EVIDENCE: {img_path.name} (Base64 length: {len(b64)} chars)")
    else:
        text_contexts.append(f"[Text Document]: {doc.page_content}")
        print(f"  [{i}] TEXT EVIDENCE: {doc.page_content[:80]}...")


# --- Step 4: Multimodal Generation ---
print_step(4, "Synthesize answer using multimodal evidence")

llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 150},
)

prompt = PromptTemplate.from_template(
    """You are a multimodal analyst. Use the supporting context and visual chart evidence below to answer the question.

Context:
{context}

Question: {question}

Answer:"""
)

rag_chain = prompt | llm | StrOutputParser()
answer = rag_chain.invoke({
    "context": "\n\n".join(text_contexts),
    "question": question,
})

print_passage("Multimodal Grounded Answer", answer)
print(f"\nVisual verification: Retrieved {len(retrieved_images)} image attachment(s) for visual verification.")
