"""Document loader: CSVLoader.

Returns one Document per CSV row, with each column rendered as a
"column: value" line in page_content and the row index in metadata.
Rows are already small and self-contained, so no further chunking is needed.
"""

import os
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "loader_csv"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
TOP_K = 3

QUESTION = "Which language uses static typing and was created after the year 2000?"

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


# --- Step 1: Load document (CSVLoader) ---
print_step(1, "Load document (CSVLoader)")
loader = CSVLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s) (one per row) from {DOCUMENT_PATH}")
for doc in documents:
    print_passage(f"Row {doc.metadata['row']}", doc.page_content)


# --- Step 2: Chunking ---
# Skipped — each row is already a small, self-contained unit of text.
print_step(2, "Chunking")
chunks = documents
print(f"Using all {len(chunks)} rows as-is, no splitting needed")


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
print(f"Stored {vector_store._collection.count()} rows in '{PERSIST_DIRECTORY}'")


# --- Step 5: Similarity search ---
print_step(5, "Similarity search")
results = vector_store.similarity_search(QUESTION, k=TOP_K)
print(f"Query: {QUESTION!r}")
print(f"Top {TOP_K} matching rows:")
for i, doc in enumerate(results, start=1):
    print_passage(f"Result {i}/{TOP_K} (row {doc.metadata['row']})", doc.page_content)


# --- Step 6: RAG pipeline ---
# Commented out on purpose — uncomment when you're ready to call Qorebit.
# print_step(6, "RAG pipeline")
#
#
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
#         "User-Agent": "rag-app-learning-project/1.0",
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
# print(f"Question: {QUESTION}")
# print_passage("Answer", answer)
