"""Multimodal RAG: End-to-End Multimodal QA with Qorebit (GPT-4o Vision).

Hosted twin of 02_multimodal_rag_pipeline.py: uses Qorebit's OpenAI-compatible
endpoint running GPT-4o with native vision support.

Multimodal Prompting with GPT-4o:
  Unlike text-only models that rely purely on OCR transcripts, GPT-4o takes
  the actual base64 image payload inside a structured user message:
    content = [
      {"type": "text", "text": "Question: ... Context: ..."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  The model's visual encoder inspects pixel data, chart trends, and diagram
  layouts directly to answer questions with high fidelity.

Needs QOREBIT_API_KEY in .env.
"""

import base64
import os
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "multimodal_rag_qorebit"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"

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


# --- Step 1: Ingest Corpus ---
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


# --- Step 2: Vector Store Indexing ---
print_step(2, "Index multimodal corpus into Chroma")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(corpus_docs, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print(f"Indexed {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")


# --- Step 3: Multimodal Retrieval & Message Construction ---
print_step(3, "Retrieve visual assets and construct Multimodal Vision Payload")

question = "What was the revenue progression in 2026 according to the cloud revenue chart?"
print(f"User Query: {question!r}")

retrieved_docs = retriever.invoke(question)

# Build multimodal message content with text + base64 image URL:
message_contents: list[dict] = [
    {
        "type": "text",
        "text": f"You are a multimodal analyst. Answer the following question based on the attached visual evidence:\n\nQuestion: {question}",
    }
]

for doc in retrieved_docs:
    if doc.metadata.get("type") == "image":
        img_path = Path(doc.metadata["source"])
        b64_data = encode_image_to_base64(img_path)
        message_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_data}"},
        })
        print(f"Attached visual evidence: {img_path.name} (Base64 length: {len(b64_data)})")
    else:
        message_contents.append({
            "type": "text",
            "text": f"Additional Context: {doc.page_content}",
        })


# --- Step 4: Invoke Vision Model ---
print_step(4, "Invoke Qorebit GPT-4o with Multimodal Message")

if "QOREBIT_API_KEY" not in os.environ or not os.environ["QOREBIT_API_KEY"].startswith("qb_"):
    print("\nNote: QOREBIT_API_KEY not found in .env.")
    print("Multimodal message payload constructed successfully:")
    print(f"  - Content parts: {len(message_contents)} (Text prompt + Base64 image attachment)")
    print("Add your QOREBIT_API_KEY to .env to execute live GPT-4o vision inference.")
else:
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        api_key=os.environ["QOREBIT_API_KEY"],
        base_url=QOREBIT_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "AI Engineering Multimodal RAG Learning Project",
            "User-Agent": "rag-app-learning-project/1.0",
        },
    )
    human_msg = HumanMessage(content=message_contents)
    response = llm.invoke([human_msg])

    print_passage("GPT-4o Vision Answer", response.content)
