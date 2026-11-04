"""RAG pipeline with cross-encoder reranking.

Standard similarity search ranks retrieved chunks by their embedding
distance to the query — fast, but purely based on how close the vector
representations are. A cross-encoder reranker sees the query and each
chunk *together* and assigns a relevance score using a small dedicated
model, which is slower but far more accurate at distinguishing truly
relevant passages from superficially similar ones.

Pipeline:
  1-4. Same load → chunk → embed → store as the base pipeline.
  5a.  Retrieve TOP_K_FETCH candidates by vector similarity (cast wide).
  5b.  Rerank those candidates with a cross-encoder (pick the best TOP_K).
  6.   Feed only the reranked top-K to the LLM.

The extra step is especially valuable when TOP_K_FETCH is large and the
embedding model retrieves some off-topic chunks that share surface-level
vocabulary with the query.

No API key needed — both models run locally.
"""

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
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Lightweight cross-encoder trained on MS MARCO passage ranking — accurate
# and fast enough to run on CPU with no GPU needed.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "huggingface_reranked"
CHAT_MODEL = "google/flan-t5-base"

TOP_K_FETCH = 6   # candidates pulled from Chroma (cast wider than needed)
TOP_K = 3         # top chunks kept after reranking (passed to the LLM)

QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


def print_passage(label: str, text: str, score: float | None = None) -> None:
    score_note = f"  [rerank score: {score:.4f}]" if score is not None else ""
    print(f"\n  {label}{score_note}")
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


# --- Step 5a: Broad similarity search (fetch more than we need) ---
print_step(5, f"Similarity search — fetch {TOP_K_FETCH} candidates, rerank to {TOP_K}")
candidates = vector_store.similarity_search(QUESTION, k=TOP_K_FETCH)
print(f"Query: {QUESTION!r}")
print(f"\nFetched {len(candidates)} candidates from Chroma (before reranking):")
for i, doc in enumerate(candidates, 1):
    print(f"  [{i}] {doc.page_content[:80].strip()!r}...")


# --- Step 5b: Rerank with cross-encoder ---
print(f"\nLoading cross-encoder: {RERANK_MODEL}")
reranker = CrossEncoder(RERANK_MODEL)

# Score each (query, passage) pair — higher = more relevant.
pairs = [(QUESTION, doc.page_content) for doc in candidates]
scores = reranker.predict(pairs)

# Sort by score descending and keep the top TOP_K.
ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
reranked_docs = [doc for _, doc in ranked[:TOP_K]]
reranked_scores = [score for score, _ in ranked[:TOP_K]]

print(f"\nTop {TOP_K} after reranking:")
for i, (doc, score) in enumerate(zip(reranked_docs, reranked_scores), 1):
    print_passage(f"Result {i}/{TOP_K}", doc.page_content, score=score)


# --- Step 6: RAG pipeline with reranked context ---
print_step(6, "RAG pipeline — answer from reranked context")

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

# The reranked docs are injected directly — no retriever needed here because
# retrieval and reranking already happened above.
context_str = format_docs(reranked_docs)
chain = prompt | llm | StrOutputParser()

answer = chain.invoke({"context": context_str, "question": QUESTION})
print(f"Question: {QUESTION}")
print(f"\nAnswer: {answer}")
