"""RAG evaluation: a small local eval harness, no API key needed.

Every earlier RAG demo prints one answer to one question and you eyeball
whether it looks right. An eval harness replaces eyeballing with a fixed set
of questions, each paired with keywords the correct answer must contain, run
automatically every time. This file checks two layers separately:

1. Retrieval quality: did the vector store's top-k chunks actually contain
   the keywords, independent of the LLM?
2. Answer quality: did the final generated answer contain them too?

Splitting the two makes failures diagnosable — a retrieval miss means the
chunking/embedding/search setup needs work, while a retrieval hit but answer
miss points at the prompt or the generation model instead. Runs on the same
free local Hugging Face embeddings + flan-t5-base + Chroma stack as
rag-app/rag_pipeline_huggingface.py.
"""

import os
import shutil
from pathlib import Path

# Must be set before chromadb is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "eval_rag"
CHAT_MODEL = "google/flan-t5-base"
TOP_K = 3

# Each question is paired with keywords the correct answer/context should
# contain. A question "passes" if at least one expected keyword shows up.
EVAL_SET = [
    {
        "question": "What are the three categories of machine learning?",
        "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
    },
    {
        "question": "What happens when a model overfits?",
        "expected_keywords": ["overfitting", "noise", "unseen data"],
    },
    {
        "question": "What does Retrieval-Augmented Generation do?",
        "expected_keywords": ["retrieving", "external knowledge", "context"],
    },
    {
        "question": "What metrics are used to evaluate a supervised model?",
        "expected_keywords": ["accuracy", "precision", "recall", "f1"],
    },
    {
        "question": "What are neural networks inspired by?",
        "expected_keywords": ["human brain", "neurons"],
    },
]


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# --- Step 1: Load, chunk, and embed the document ---
print_step(1, "Load, chunk, and embed the document")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)  # avoid duplicating chunks on rerun
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
print(f"Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")


# --- Step 2: Retrieval eval ---
print_step(2, f"Retrieval eval ({len(EVAL_SET)} questions, top {TOP_K} chunks each)")
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
retrieval_hits = 0
for item in EVAL_SET:
    retrieved = retriever.invoke(item["question"])
    context = format_docs(retrieved)
    passed = contains_any_keyword(context, item["expected_keywords"])
    retrieval_hits += passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {item['question']}")

retrieval_score = retrieval_hits / len(EVAL_SET)
print(f"\nRetrieval score: {retrieval_hits}/{len(EVAL_SET)} ({retrieval_score:.0%})")


# --- Step 3: Build the full RAG chain ---
print_step(3, "Build the full RAG chain (retriever + prompt + generation)")
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


# --- Step 4: Answer eval ---
print_step(4, f"Answer eval ({len(EVAL_SET)} questions, end-to-end)")
answer_hits = 0
for item in EVAL_SET:
    answer = rag_chain.invoke(item["question"])
    passed = contains_any_keyword(answer, item["expected_keywords"])
    answer_hits += passed
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {item['question']}")
    print(f"  Answer: {answer}")

answer_score = answer_hits / len(EVAL_SET)
print(f"\nAnswer score: {answer_hits}/{len(EVAL_SET)} ({answer_score:.0%})")


# --- Step 5: Summary ---
print_step(5, "Summary")
print(f"Retrieval score: {retrieval_hits}/{len(EVAL_SET)} ({retrieval_score:.0%})")
print(f"Answer score:    {answer_hits}/{len(EVAL_SET)} ({answer_score:.0%})")
if answer_score < retrieval_score:
    print("Answers lag retrieval — the chunks have the right info but generation drops it.")
elif answer_score > retrieval_score:
    print("Answers beat retrieval — flan-t5 may be using pretrained knowledge, not just context.")
else:
    print("Retrieval and answer quality match.")
