"""RAG pipeline with a persistent vector store and --rebuild flag.

Every earlier RAG demo deletes its Chroma folder and rebuilds from
scratch on every run. That's fine while learning, but slow and wasteful
when the document hasn't changed. This file keeps the store between runs:

  - First run (or --rebuild): builds the store and saves it to db/.
  - Later runs: loads the existing store and skips steps 1-4 entirely,
    jumping straight to the similarity search and generation.
  - Pass --rebuild to force a fresh rebuild at any time.

Usage:

  # First run — builds and saves the store (steps 1-4 run):
  python rag-app/rag_pipeline_persistent.py

  # Later run — reuses the saved store (steps 1-4 skipped, near-instant):
  python rag-app/rag_pipeline_persistent.py

  # Force a fresh rebuild even if the store exists:
  python rag-app/rag_pipeline_persistent.py --rebuild

  # Different question, same store:
  python rag-app/rag_pipeline_persistent.py --question "What is overfitting?"

  # All flags:
  python rag-app/rag_pipeline_persistent.py --help

No API key needed — runs on the same free local stack.
"""

import argparse
import os
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


def print_passage(label: str, text: str, char_count: int | None = None) -> None:
    suffix = f" ({char_count} chars)" if char_count is not None else ""
    print(f"\n  {label}{suffix}")
    print("  " + "-" * 60)
    wrapped = textwrap.fill(text, width=68, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG pipeline with a persistent vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        default="What is Retrieval-Augmented Generation and why is it useful?",
        help="Question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing vector store and rebuild it from scratch.",
    )
    parser.add_argument(
        "--doc", "-d",
        default=str(PROJECT_ROOT / "data" / "sample.txt"),
        help="Path to the input document (used only when building/rebuilding).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Characters per chunk (used only when building/rebuilding).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Character overlap between chunks (used only when building/rebuilding).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--db-dir",
        default=str(PROJECT_ROOT / "db" / "huggingface_persistent"),
        help="Directory for the persistent Chroma vector store.",
    )
    return parser.parse_args()


args = parse_args()

DOCUMENT_PATH = Path(args.doc)
CHUNK_SIZE = args.chunk_size
CHUNK_OVERLAP = args.chunk_overlap
TOP_K = args.top_k
PERSIST_DIRECTORY = Path(args.db_dir)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-base"
QUESTION = args.question

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

store_exists = PERSIST_DIRECTORY.exists() and any(PERSIST_DIRECTORY.iterdir())

if args.rebuild and store_exists:
    print(f"\n--rebuild passed: deleting existing store at '{PERSIST_DIRECTORY}'")
    shutil.rmtree(PERSIST_DIRECTORY)
    store_exists = False

if not store_exists:
    # --- Build the store (steps 1-4) ---
    print_step(1, "Prepare input document")
    loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {DOCUMENT_PATH}")

    print_step(2, "Chunking")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, start=1):
        print_passage(f"Chunk {i}/{len(chunks)}", chunk.page_content, len(chunk.page_content))

    print_step(3, "Create embeddings")
    print(f"Loaded embedding model: {EMBEDDING_MODEL}")

    print_step(4, "Store embeddings in vector database (persisted)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )
    print(f"Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")
    print(f"Store saved — future runs will skip steps 1-4 (pass --rebuild to redo them).")
else:
    # --- Reuse the existing store ---
    print(f"\n[Steps 1-4 skipped — reusing existing store at '{PERSIST_DIRECTORY}']")
    print(f"(Pass --rebuild to force a fresh rebuild.)\n")
    vector_store = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embeddings,
    )
    print(f"Loaded existing store: {vector_store._collection.count()} chunks")


# --- Step 5: Similarity search ---
print_step(5, "Similarity search")
results = vector_store.similarity_search(QUESTION, k=TOP_K)
print(f"Query: {QUESTION!r}")
print(f"Top {TOP_K} matching chunks:")
for i, doc in enumerate(results, start=1):
    print_passage(f"Result {i}/{TOP_K}", doc.page_content)


# --- Step 6: RAG pipeline ---
print_step(6, "RAG pipeline")

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)
prompt = PromptTemplate.from_template(
    """Answer the question using only the context below.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke(QUESTION)
print(f"Question: {QUESTION}")
print_passage("Answer", answer)
