"""LangGraph: chaining multiple specialized nodes.

Every earlier file in this folder had one job per graph. This one has two
distinct roles wired together: a "researcher" node that retrieves relevant
passages from a Chroma vector store (no LLM — the same retriever setup as
06-agents/04_retriever_tool_agent.py), and a "writer" node that takes those
passages and rewrites them in a specific style. Each node only sees what it
needs from the shared state, the same way separate agents would hand off
work to each other in a larger multi-agent system.
"""

import os
import shutil
from pathlib import Path
from typing import TypedDict

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

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
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "langgraph_multi_agent"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K = 3

QUESTION = "What is Retrieval-Augmented Generation?"


class State(TypedDict):
    question: str
    context: str
    answer: str


print_step(1, "Load, chunk, and embed the document (used by the researcher node)")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
writer_prompt = PromptTemplate.from_template(
    "Explain this to a complete beginner, in two short sentences, using only the context "
    "below.\n\nContext:\n{context}\n\nQuestion: {question}"
)


def researcher(state: State) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join(doc.page_content for doc in docs)}


def writer(state: State) -> dict:
    prompt_value = writer_prompt.invoke({"context": state["context"], "question": state["question"]})
    response = llm.invoke(prompt_value)
    return {"answer": response.content}


print_step(2, "Build the graph: START -> researcher -> writer -> END")
graph_builder = StateGraph(State)
graph_builder.add_node("researcher", researcher)
graph_builder.add_node("writer", writer)
graph_builder.add_edge(START, "researcher")
graph_builder.add_edge("researcher", "writer")
graph_builder.add_edge("writer", END)
graph = graph_builder.compile()

print_step(3, "Run the graph")
result = graph.invoke({"question": QUESTION})
print(f"Question: {QUESTION}")
print(f"\nResearcher's retrieved context:\n{result['context']}")
print(f"\nWriter's beginner-friendly answer:\n{result['answer']}")
