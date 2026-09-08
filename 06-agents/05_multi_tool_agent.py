"""Agents: choosing among multiple tools.

Every earlier file in this folder gave the agent exactly one tool, so there
was nothing to actually choose between. This one hands it three at once —
a calculator, a word counter, and a document retriever — and asks a batch of
questions that each only need one of them, to see whether the model reaches
for the right tool (or no tool at all) each time.
"""

import os
import shutil
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
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
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "agents_multi_tool"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K = 3

QUESTIONS = [
    "What is 144 divided by 12?",
    "How many words are in the phrase 'artificial intelligence is transforming the world'?",
    "What is Retrieval-Augmented Generation?",
]


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '144 / 12'."""
    import numexpr

    try:
        return str(numexpr.evaluate(expression.strip("'\" ")).item())
    except Exception as e:
        return f"Error: could not evaluate {expression!r} ({e})"


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


print_step(1, "Build the retriever tool")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever_tool = create_retriever_tool(
    vector_store.as_retriever(search_kwargs={"k": TOP_K}),
    name="search_ml_notes",
    description="Search notes about machine learning and RAG concepts.",
)

print_step(2, "Build one agent with all three tools")
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
tools = [calculator, word_count, retriever_tool]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant with access to several tools. "
                   "Pick whichever tool (if any) actually helps answer the question."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print_step(3, "Ask each question and see which tool (if any) gets picked")
for question in QUESTIONS:
    result = agent_executor.invoke({"input": question})
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['output']}")
