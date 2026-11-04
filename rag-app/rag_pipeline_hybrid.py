"""RAG pipeline with hybrid search (vector + BM25).

Dense vector search is good at semantic similarity — it finds chunks that
*mean* the same thing as the query even when they use different words.
BM25 keyword search is good at exact-term matching — it reliably finds
chunks that *contain* the exact words in the query. Neither is best on
its own for every question. EnsembleRetriever merges both ranked lists
using Reciprocal Rank Fusion (RRF), so chunks that rank highly in either
method are promoted — combining the strengths of both.

Pipeline:
  1-4. Same load → chunk → embed → store as the base pipeline.
  5.   EnsembleRetriever: BM25 retriever + Chroma retriever → fused results.
  6.   Standard RAG generation on the fused context.

No API key needed — runs on the same free local stack.
"""

import os
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "huggingface_hybrid"
CHAT_MODEL = "google/flan-t5-base"
TOP_K = 3

# EnsembleRetriever weight for each retriever — must sum to 1.0.
# 0.5 / 0.5 gives equal weight to BM25 and vector search.
# Increase the vector weight (e.g. 0.3 / 0.7) if your queries are
# more semantic; increase BM25 (e.g. 0.7 / 0.3) for exact-term lookups.
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"

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


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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


# --- Step 5: Hybrid retrieval (BM25 + vector) ---
print_step(5, "Hybrid retrieval — BM25 + vector search fused with RRF")

# BM25Retriever works directly on the Document objects — no embedding needed.
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = TOP_K

# Standard dense retriever from Chroma.
vector_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

# EnsembleRetriever merges both ranked lists using Reciprocal Rank Fusion.
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[BM25_WEIGHT, VECTOR_WEIGHT],
)

print(f"Query: {QUESTION!r}")
print(f"BM25 weight: {BM25_WEIGHT}  |  Vector weight: {VECTOR_WEIGHT}")

# Show each retriever's results separately for comparison, then the fused set.
bm25_results = bm25_retriever.invoke(QUESTION)
vector_results = vector_retriever.invoke(QUESTION)
fused_results = ensemble_retriever.invoke(QUESTION)

print(f"\nBM25 alone returned {len(bm25_results)} chunk(s):")
for i, doc in enumerate(bm25_results, 1):
    print(f"  [{i}] {doc.page_content[:80].strip()!r}...")

print(f"\nVector search alone returned {len(vector_results)} chunk(s):")
for i, doc in enumerate(vector_results, 1):
    print(f"  [{i}] {doc.page_content[:80].strip()!r}...")

print(f"\nFused (hybrid) results — {len(fused_results)} chunk(s):")
for i, doc in enumerate(fused_results, 1):
    print_passage(f"Result {i}/{len(fused_results)}", doc.page_content)


# --- Step 6: RAG pipeline on fused context ---
print_step(6, "RAG pipeline — answer from hybrid context")

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
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke(QUESTION)
print(f"Question: {QUESTION}")
print(f"\nAnswer: {answer}")
