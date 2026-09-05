"""RAG (Retrieval-Augmented Generation) app built with LangChain.

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

# Must be set before chromadb is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Resolved from this file's location so it works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "qorebit"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
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
# Free local model — Qorebit's API doesn't cover embeddings.
print_step(3, "Create embeddings")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print(f"Loaded embedding model: {EMBEDDING_MODEL}")


# --- Step 4: Store embeddings in vector database ---
print_step(4, "Store embeddings in vector database")
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)  # avoid duplicating chunks on rerun
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
# Commented out to avoid burning Qorebit credits; uncomment when ready.

# def format_docs(docs) -> str:
#     return "\n\n".join(doc.page_content for doc in docs)
#
#
# retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
# llm = ChatOpenAI(
#     model=CHAT_MODEL,
#     temperature=0,
#     api_key=os.environ["QOREBIT_API_KEY"],
#     base_url=QOREBIT_BASE_URL,
#     default_headers={
#         "HTTP-Referer": "http://localhost",
#         "X-Title": "AI Engineering RAG Learning App",
#         "User-Agent": "rag-app-learning-project/1.0",  # Qorebit's WAF blocks the default one
#     },
# )
# prompt = ChatPromptTemplate.from_template(
#     """Answer the question using only the context below.
# If the context doesn't contain the answer, say you don't know.
#
# Context:
# {context}
#
# Question: {question}
#
# Answer:"""
# )
#
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
#
# answer = rag_chain.invoke(QUESTION)
# print_step(6, "RAG pipeline")
# print(f"Question: {QUESTION}")
# print_passage("Answer", answer)
