"""RAG pipeline with CLI flags.

Same pipeline as rag_pipeline_huggingface.py but all previously
hard-coded constants are now proper CLI arguments, so you can run
different questions, documents, and settings without editing the file.

Usage examples:

  # Default settings (same question and document as the original):
  python rag-app/rag_pipeline_cli.py

  # Custom question:
  python rag-app/rag_pipeline_cli.py --question "What is overfitting?"

  # Custom document and chunk size:
  python rag-app/rag_pipeline_cli.py --doc data/sample.txt --chunk-size 300

  # See all available flags:
  python rag-app/rag_pipeline_cli.py --help

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
        description="RAG pipeline with configurable CLI flags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        default="What is Retrieval-Augmented Generation and why is it useful?",
        help="Question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--doc", "-d",
        default=str(PROJECT_ROOT / "data" / "sample.txt"),
        help="Path to the input document (absolute or relative to CWD).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Character overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--db-dir",
        default=str(PROJECT_ROOT / "db" / "huggingface_cli"),
        help="Directory for the Chroma vector store.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name.",
    )
    parser.add_argument(
        "--chat-model",
        default="google/flan-t5-base",
        help="HuggingFace generation model name.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate in the answer.",
    )
    return parser.parse_args()


args = parse_args()

DOCUMENT_PATH = Path(args.doc)
CHUNK_SIZE = args.chunk_size
CHUNK_OVERLAP = args.chunk_overlap
TOP_K = args.top_k
PERSIST_DIRECTORY = Path(args.db_dir)
EMBEDDING_MODEL = args.embedding_model
CHAT_MODEL = args.chat_model
MAX_NEW_TOKENS = args.max_new_tokens
QUESTION = args.question

print(f"\nConfiguration:")
print(f"  document:    {DOCUMENT_PATH}")
print(f"  chunk_size:  {CHUNK_SIZE}  overlap: {CHUNK_OVERLAP}")
print(f"  top_k:       {TOP_K}")
print(f"  db_dir:      {PERSIST_DIRECTORY}")
print(f"  embeddings:  {EMBEDDING_MODEL}")
print(f"  chat_model:  {CHAT_MODEL}")
print(f"  question:    {QUESTION!r}")


# --- Step 1: Prepare input document ---
print_step(1, "Prepare input document")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s) from {DOCUMENT_PATH}")


# --- Step 2: Chunking ---
print_step(2, "Chunking")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks, start=1):
    print_passage(f"Chunk {i}/{len(chunks)}", chunk.page_content, len(chunk.page_content))


# --- Step 3: Create embeddings ---
print_step(3, "Create embeddings")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print(f"Loaded embedding model: {EMBEDDING_MODEL}")


# --- Step 4: Store embeddings in vector database ---
print_step(4, "Store embeddings in vector database")
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(PERSIST_DIRECTORY),
)
print(f"Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")


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
    pipeline_kwargs={"max_new_tokens": MAX_NEW_TOKENS},
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
