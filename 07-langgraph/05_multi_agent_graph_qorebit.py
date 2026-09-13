"""LangGraph: chaining multiple specialized nodes, on Qorebit instead of Ollama.

Same researcher/writer setup as 05_multi_agent_graph.py, but the writer
node's LLM call goes to Qorebit's hosted model (gpt-4o) instead of local
Ollama. This is a live Qorebit call — see the note in
06-agents/01_react_agent_qorebit.py about why agent demos don't have a free
partial run to fall back to.
"""

import os
import shutil
from pathlib import Path
from typing import TypedDict

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()

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
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "langgraph_multi_agent_qorebit"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
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

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    api_key=os.environ["QOREBIT_API_KEY"],
    base_url=QOREBIT_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Engineering RAG Learning App",
        "User-Agent": "rag-app-learning-project/1.0",
    },
)
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
