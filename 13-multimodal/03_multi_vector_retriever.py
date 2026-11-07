"""Multimodal RAG: LangChain MultiVectorRetriever for Visual Assets.

A common challenge in multimodal RAG is the representation mismatch:
  - Vector search works best on concise, semantically dense text summaries.
  - LLM synthesis works best with the original, uncompressed visual artifact (raw image).

The MultiVectorRetriever resolves this by decoupling the searchable representation
from the returned representation:
  1. Each visual image is assigned a unique UUID `doc_id`.
  2. The full raw image base64 payload is stored as a Document in the docstore (InMemoryStore).
  3. The semantic text summary of the image is embedded in the vector store (Chroma)
     with `doc_id` attached in metadata.
  4. When a user asks a query:
     - The vector store finds the matching summary by cosine similarity.
     - The MultiVectorRetriever automatically fetches the original raw image Document from the docstore
       using the mapped `doc_id`.

No API key needed — runs entirely on the local HuggingFace stack.
"""

import base64
import os
import shutil
import uuid
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "multimodal_multi_vector"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

IMAGE_ASSETS = [
    {
        "filename": "sample_architecture.png",
        "path": DATA_DIR / "sample_architecture.png",
        "summary": (
            "Architecture diagram for a Retrieval-Augmented Generation application. "
            "Depicts client requests routing through an API Gateway to a Chroma vector "
            "database and an LLM generation microservice."
        ),
    },
    {
        "filename": "sample_revenue_chart.png",
        "path": DATA_DIR / "sample_revenue_chart.png",
        "summary": (
            "Quarterly cloud revenue chart for 2026. "
            "Displays a bar graph showing Q1 $1.2M, Q2 $1.8M, Q3 $2.4M, and Q4 $3.1M revenue, "
            "showing steady upward growth across all quarters."
        ),
    },
]

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# --- Step 1: Initialize Storage Layers ---
print_step(1, "Initialize Vector Store (Summaries) and Docstore (Raw Images)")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)

vectorstore = Chroma(
    collection_name="visual_summaries",
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIRECTORY),
)
docstore = InMemoryStore()
id_key = "doc_id"

# Assemble MultiVectorRetriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
    search_kwargs={"k": 1},
)
print("MultiVectorRetriever configured:")
print(f"  - Vector Store: Chroma ('{PERSIST_DIRECTORY}')")
print("  - Docstore:     InMemoryStore")
print(f"  - Link Key:     '{id_key}'")


# --- Step 2: Index Visual Assets ---
print_step(2, "Index Image Summaries into Vector Store and Raw Images into Docstore")

doc_ids = []
summary_docs = []
raw_documents = []

for asset in IMAGE_ASSETS:
    doc_id = str(uuid.uuid4())
    img_path = asset["path"]

    with open(img_path, "rb") as f:
        img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

    # 1. Searchable summary Document stored in Chroma
    summary_doc = Document(
        page_content=asset["summary"],
        metadata={id_key: doc_id, "filename": asset["filename"]},
    )
    summary_docs.append(summary_doc)
    doc_ids.append(doc_id)

    # 2. Raw artifact Document stored in Docstore (contains original base64 image + metadata)
    raw_doc = Document(
        page_content=b64,
        metadata={
            "filename": asset["filename"],
            "path": str(img_path),
            "bytes_length": len(img_bytes),
        },
    )
    raw_documents.append((doc_id, raw_doc))
    print(f"Indexed: {asset['filename']} -> doc_id={doc_id[:8]}... (Image bytes: {len(img_bytes)}B)")

# Store summaries in vectorstore and raw artifacts in docstore
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(raw_documents)

print(f"\nStored {len(summary_docs)} summaries in Chroma and {len(raw_documents)} raw images in Docstore.")


# --- Step 3: Multi-Vector Query Resolution ---
print_step(3, "Query using MultiVectorRetriever")

query = "Show me the quarterly revenue chart for 2026."
print(f"Query: {query!r}")

# MultiVectorRetriever searches summaries in Chroma, then automatically
# retrieves the original raw Document from the docstore!
resolved_docs = retriever.invoke(query)

print(f"\nRetriever resolved {len(resolved_docs)} raw artifact(s):")
for i, doc in enumerate(resolved_docs, start=1):
    print(f"  [{i}] Resolved Artifact: {doc.metadata['filename']}")
    print(f"      Source Path:       {doc.metadata['path']}")
    print(f"      Size:              {doc.metadata['bytes_length']} bytes")
    print(f"      Base64 Preview:    {doc.page_content[:50]}...")

print(f"\n{RULE}")
print("KEY TAKEAWAY")
print(RULE)
print("The query matched the semantic text summary in Chroma, but the retriever")
print("seamlessly returned the full raw visual artifact from the document store.")
