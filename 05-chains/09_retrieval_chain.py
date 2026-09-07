"""Chain composition: create_retrieval_chain (the modern RetrievalQA).

Wires a retriever together with a stuff-documents chain in one call — the
current LCEL-based replacement for the older RetrievalQA chain class. It's
the same retrieve-then-generate idea as rag-app/rag_pipeline.py, but built
from official helper functions instead of hand-assembled with
RunnablePassthrough. Runs entirely on the free local flan-t5-base + Chroma
stack used throughout this project.
"""

import os
import shutil
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
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
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "chains_retrieval"
CHAT_MODEL = "google/flan-t5-base"
TOP_K = 3

QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"

print_step(1, "Load, chunk, and embed the document")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
print(f"Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")

print_step(2, "Build the retrieval chain (retriever + stuff-documents chain)")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)
prompt = PromptTemplate.from_template("Answer using only this context:\n\n{context}\n\nQuestion: {input}")
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

print_step(3, "Invoke: retrieve, then stuff into the prompt, then generate")
result = rag_chain.invoke({"input": QUESTION})
print(f"Question: {QUESTION}")
print(f"Retrieved {len(result['context'])} chunks")
print(f"Answer: {result['answer']}")
