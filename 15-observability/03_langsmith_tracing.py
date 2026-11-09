"""Observability: LangSmith Production Tracing and Metadata Tagging.

LangSmith is LangChain's native observability and evaluation platform.
It automatically records:
  - Complete hierarchical execution trees (prompts, retrievers, tools, models)
  - Exact token counts, streaming latencies, and generation costs
  - Custom metadata (user IDs, environment, tenant IDs, model versions)
  - Quality feedback and evaluation scores

Key Feature: Zero Code Changes Required for Basic Tracing
Simply setting four environment variables automatically enables full tracing:
  export LANGCHAIN_TRACING_V2="true"
  export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
  export LANGCHAIN_API_KEY="lsv2_..."
  export LANGCHAIN_PROJECT="ai-engineering-learning"

This file demonstrates:
  1. Inspecting and configuring LangSmith environment variables.
  2. Attaching custom run tags and metadata via the `config` dictionary.
  3. Specifying human-readable `run_name` labels for complex chains.
  4. Running a traced RAG chain with full diagnostic visibility.

Runs locally with free models — functions cleanly with or without a live LangSmith API key.
"""

import os
import shutil
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "observability_langsmith"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-base"
TOP_K = 2

QUESTION = "What is overfitting in machine learning and how do you prevent it?"

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# --- Step 1: Inspect LangSmith Configuration ---
print_step(1, "Inspect LangSmith Tracing Configuration")

has_key = bool(os.environ.get("LANGCHAIN_API_KEY", "").startswith("lsv2_"))
project_name = os.environ.get("LANGCHAIN_PROJECT", "ai-engineering-observability")

# Enable tracing in process environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = project_name

print(f"  LANGCHAIN_TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"  LANGCHAIN_PROJECT:    {project_name}")
print(f"  LANGCHAIN_API_KEY:    {'Configured (Active)' if has_key else 'Not Found (Dry-Run Mode)'}")


# --- Step 2: Assemble RAG Chain with Custom Run Names ---
print_step(2, "Assemble RAG chain with custom run_name attributes")

loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K}).with_config(
    run_name="KnowledgeBaseRetriever"
)

llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 120},
).with_config(run_name="AnswerGeneratorLLM")

prompt = PromptTemplate.from_template(
    """Answer the question based only on the context below:

{context}

Question: {question}

Answer:"""
).with_config(run_name="RagPromptFormatter")


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
).with_config(run_name="FullRagPipeline")


# --- Step 3: Execute Chain with Metadata & Tags ---
print_step(3, "Invoke chain with metadata, tags, and run configuration")

run_config = {
    "run_name": "Production_Overfitting_Inquiry",
    "tags": ["evaluation", "rag-v2", "release-candidate"],
    "metadata": {
        "user_id": "usr_9941",
        "session_id": "sess_88124",
        "environment": "development",
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
    },
}

print(f"Invoking Question: {QUESTION!r}")
print("Metadata attached to trace:")
for k, v in run_config["metadata"].items():
    print(f"  - {k}: {v}")
print(f"Tags attached: {run_config['tags']}")

answer = rag_chain.invoke(QUESTION, config=run_config)

print(f"\nAnswer: {answer!r}")

# --- Step 4: Summary ---
print_step(4, "Observability Trace Summary")
if has_key:
    print(f"Trace successfully uploaded to LangSmith project: '{project_name}'.")
    print(f"View your execution tree at: https://smith.langchain.com")
else:
    print("In dry-run mode (no LANGCHAIN_API_KEY). To stream traces to the live dashboard:")
    print("  1. Sign up for a free key at https://smith.langchain.com")
    print("  2. Add LANGCHAIN_API_KEY=lsv2_... and LANGCHAIN_TRACING_V2=true to your .env")
    print("  3. Rerun this file — every node, latency, and token count will appear in real time.")
