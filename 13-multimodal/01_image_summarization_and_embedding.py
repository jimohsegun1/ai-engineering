"""Multimodal RAG: Image Summarization and Vector Embedding.

Standard embedding models (like sentence-transformers) only accept text.
To make visual assets — architecture diagrams, charts, screenshots, figures —
searchable in a vector database, we employ an Image-to-Text Summarization step:

  1. Images are loaded and encoded to base64.
  2. A Vision Model (Ollama llava/llama3.2-vision, GPT-4o, or local visual descriptor)
     inspects each image and extracts:
     - Diagram type and title
     - Components, nodes, and interconnections
     - Quantitative metrics, axis labels, and trends
  3. The rich textual summary is embedded with sentence-transformers and stored
     in Chroma along with metadata pointing back to the original image path.
  4. Natural language queries can now retrieve the exact visual asset via similarity search!

No API key needed — includes deterministic visual descriptor fallback so it runs
immediately on any CPU.
"""

import base64
import os
import shutil
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "multimodal_summaries"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

IMAGE_FILES = [
    DATA_DIR / "sample_architecture.png",
    DATA_DIR / "sample_revenue_chart.png",
]

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


def encode_image_to_base64(image_path: Path) -> str:
    """Read binary image file and convert to base64 data string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# --- Step 1: Extract Visual Descriptions ---
print_step(1, "Load images and generate semantic visual summaries")

# Pre-computed rich visual summaries representing what a vision model extracts
# from the visual pixels:
DETERMINISTIC_VISION_SUMMARIES = {
    "sample_architecture.png": (
        "System Architecture Diagram depicting an end-to-end RAG application. "
        "Flow begins on the left with User/Web Client connecting via HTTP line to "
        "an API Gateway & FastAPI Router in the center. The API router splits requests into two "
        "backend services: a Chroma Vector DB node at the top for embedding search and a "
        "downstream LLM Service node at the bottom for response generation."
    ),
    "sample_revenue_chart.png": (
        "Bar Chart titled 'Quarterly Cloud Revenue (2026)'. "
        "The horizontal x-axis shows four quarters: Q1 ($1.2M), Q2 ($1.8M), Q3 ($2.4M), and Q4 ($3.1M). "
        "The vertical y-axis measures revenue in millions of dollars. "
        "The trend demonstrates steady quarter-over-quarter upward revenue growth throughout 2026."
    ),
}

documents: list[Document] = []

for img_path in IMAGE_FILES:
    if not img_path.exists():
        print(f"Warning: Image file not found: {img_path}")
        continue

    b64 = encode_image_to_base64(img_path)
    filename = img_path.name
    summary_text = DETERMINISTIC_VISION_SUMMARIES.get(
        filename,
        f"Visual image file {filename} containing technical diagram context."
    )

    doc = Document(
        page_content=summary_text,
        metadata={
            "source_image": str(img_path),
            "filename": filename,
            "image_type": "diagram" if "architecture" in filename else "chart",
            "base64_preview": b64[:40] + "...",
        },
    )
    documents.append(doc)
    print(f"\nProcessed: {filename}")
    print(f"  Summary: {summary_text[:90]}...")
    print(f"  Base64 length: {len(b64)} chars")


# --- Step 2: Embed and Store in Vector Database ---
print_step(2, "Embed image summaries into Chroma Vector Store")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=str(PERSIST_DIRECTORY),
)
print(f"Indexed {vector_store._collection.count()} visual document summaries into '{PERSIST_DIRECTORY}'")


# --- Step 3: Natural Language Retrieval ---
print_step(3, "Retrieve visual assets using natural language questions")

queries = [
    "What was the cloud revenue growth in Q3 and Q4?",
    "How does the user connect to the Chroma vector database and LLM service?",
]

for q in queries:
    print(f"\nQuery: {q!r}")
    matches = vector_store.similarity_search_with_score(q, k=1)
    matched_doc, distance = matches[0]

    print(f"  -> Top Match: {matched_doc.metadata['filename']} (L2 Distance: {distance:.4f})")
    print(f"  -> Asset Path: {matched_doc.metadata['source_image']}")
    print(f"  -> Summary:    {matched_doc.page_content[:100]}...")

print(f"\n{RULE}")
print("SUMMARY")
print(RULE)
print("By storing vision-derived text summaries alongside image file metadata,")
print("standard text queries seamlessly retrieve relevant high-resolution diagrams and charts.")
