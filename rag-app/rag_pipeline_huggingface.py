"""RAG (Retrieval-Augmented Generation) app built with LangChain.

This version runs entirely on local, free Hugging Face models for both
embeddings and generation — no API key, no internet-dependent LLM call,
no billing. Compare with rag_pipeline.py, which uses Qorebit (a hosted
OpenAI-compatible API) for generation instead.

Steps:
1. Prepare input document
2. Chunking
3. Create embeddings
4. Store embeddings in vector database
5. Similarity search
6. RAG pipeline
"""

import os
import shutil
import textwrap
from pathlib import Path

# Opt out of Chroma's anonymous usage telemetry (must be set before chromadb
# is imported). requirements.txt also pins a compatible `posthog` version,
# since a too-new posthog release breaks chromadb's telemetry call outright.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Resolved from this file's own location, not the current working directory,
# so this script runs correctly no matter which directory you run it from.
RAG_APP_DIR = Path(__file__).resolve().parent

DOCUMENT_PATH = RAG_APP_DIR / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = RAG_APP_DIR / "db" / "huggingface_local"
CHAT_MODEL = "google/flan-t5-base"
TOP_K = 3

QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"

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


# --- Step 1: Prepare input document ---
# Load the raw text file into LangChain's Document format (text + metadata).
print_step(1, "Prepare input document")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s) from {DOCUMENT_PATH}")


# --- Step 2: Chunking ---
# Split the document into overlapping chunks so retrieval can return
# focused passages instead of the whole file.
print_step(2, "Chunking")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks, start=1):
    print_passage(f"Chunk {i}/{len(chunks)}", chunk.page_content, len(chunk.page_content))


# --- Step 3: Create embeddings ---
# An embedding model turns text into a vector of numbers that captures
# meaning, so similar text ends up with similar vectors.
print_step(3, "Create embeddings")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print(f"Loaded embedding model: {EMBEDDING_MODEL}")


# --- Step 4: Store embeddings in vector database ---
# Chroma embeds every chunk and stores the vectors (plus original text)
# on disk so they can be searched later without re-embedding. We wipe
# any previous run's data first so re-running this script doesn't keep
# appending duplicate chunks to the same collection.
print_step(4, "Store embeddings in vector database")
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(PERSIST_DIRECTORY),
)
print(f"Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")


# --- Step 5: Similarity search ---
# Embed the query and ask the vector database for the closest chunks.
print_step(5, "Similarity search")
results = vector_store.similarity_search(QUESTION, k=TOP_K)
print(f"Query: {QUESTION!r}")
print(f"Top {TOP_K} matching chunks:")
for i, doc in enumerate(results, start=1):
    print_passage(f"Result {i}/{TOP_K}", doc.page_content)


# --- Step 6: RAG pipeline ---
# Wire retrieval + prompt + LLM together: retrieve relevant chunks,
# inject them as context, and have the LLM generate a grounded answer.
# Runs fully locally via `transformers` (no API key, no network call) —
# free to run as many times as you like.
print_step(6, "RAG pipeline")


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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
