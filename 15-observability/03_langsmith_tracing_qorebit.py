"""Observability: LangSmith Production Tracing with Qorebit (GPT-4o).

Hosted twin of 03_langsmith_tracing.py: combines LangSmith tracing with
Qorebit's OpenAI-compatible endpoint running GPT-4o.

When tracing commercial models (like GPT-4o), LangSmith automatically captures:
  - Exact prompt tokens and completion tokens
  - Cost accounting based on model pricing tiers
  - Latency breakdown: Time-to-First-Token (TTFT) and total generation time
  - System prompts, retrieved context, and formatted user instructions

Needs:
  - QOREBIT_API_KEY in .env
  - Optional: LANGCHAIN_API_KEY in .env (for live cloud dashboard sync)
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "observability_qorebit"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
TOP_K = 2

QUESTION = "What is overfitting in machine learning and how do you prevent it?"

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# --- Step 1: LangSmith Tracing Setup ---
print_step(1, "Configure LangSmith Tracing Environment")

has_ls_key = bool(os.environ.get("LANGCHAIN_API_KEY", "").startswith("lsv2_"))
has_qb_key = bool(os.environ.get("QOREBIT_API_KEY", "").startswith("qb_"))

if has_ls_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "ai-engineering-qorebit-prod")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

print(f"  LANGCHAIN_TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"  LANGCHAIN_PROJECT:    {os.environ.get('LANGCHAIN_PROJECT', 'ai-engineering-qorebit-prod')}")
print(f"  LANGCHAIN_API_KEY:    {'Configured' if has_ls_key else 'Not Found (Dry-Run / Local Mode)'}")
print(f"  QOREBIT_API_KEY:      {'Configured' if has_qb_key else 'Not Found'}")


# --- Step 2: Index Chroma Store ---
print_step(2, "Index Chroma vector store")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K}).with_config(
    run_name="ChromaKnowledgeRetriever"
)


# --- Step 3: Assemble Chain with GPT-4o ---
print_step(3, "Assemble Qorebit GPT-4o RAG pipeline")

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    api_key=os.environ.get("QOREBIT_API_KEY", "qb_placeholder"),
    base_url=QOREBIT_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Engineering Observability Learning App",
        "User-Agent": "rag-app-learning-project/1.0",
    },
).with_config(run_name="GPT4oGenerator")

prompt = PromptTemplate.from_template(
    """Answer the question based only on the context below:

{context}

Question: {question}

Answer:"""
).with_config(run_name="FormatContextPrompt")


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
).with_config(run_name="ProductionQorebitPipeline")


# --- Step 4: Execute Traced Run ---
print_step(4, "Execute traced run with custom metadata and tags")

run_config = {
    "run_name": "Qorebit_GPT4o_Overfitting_Analysis",
    "tags": ["production", "gpt-4o", "langsmith-traced"],
    "metadata": {
        "user_tier": "enterprise",
        "pipeline_version": "2.4.0",
        "provider": "qorebit",
    },
}

if not has_qb_key:
    print("Note: QOREBIT_API_KEY not configured in .env.")
    print("Pipeline structure compiled successfully with run names:")
    print("  - Full Pipeline:        ProductionQorebitPipeline")
    print("  - Knowledge Retriever:  ChromaKnowledgeRetriever")
    print("  - Formatter Prompt:     FormatContextPrompt")
    print("  - LLM Generator:        GPT4oGenerator")
    print("\nAdd your QOREBIT_API_KEY to .env to execute live GPT-4o calls.")
else:
    answer = rag_chain.invoke(QUESTION, config=run_config)
    print(f"Question: {QUESTION!r}")
    print(f"\nAnswer: {answer}")
    if has_ls_key:
        print(f"\nTrace sent to LangSmith dashboard under project '{os.environ['LANGCHAIN_PROJECT']}'.")
