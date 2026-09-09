"""Agents: choosing among multiple tools, on Qorebit instead of Ollama.

Same three-tool setup as 05_multi_tool_agent.py, but with Qorebit's hosted
model (gpt-4o) as the LLM. This is a live Qorebit call for all three
questions — see the note in 01_react_agent_qorebit.py about why agent demos
don't have a free partial run to fall back to.

Needs `disable_streaming=True` on the LLM — see the note in
03_tool_calling_agent_qorebit.py for why.
"""

import os
import shutil
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "agents_multi_tool_qorebit"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
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
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    disable_streaming=True,  # Qorebit mangles tool_call ids when streamed, see module docstring
    api_key=os.environ["QOREBIT_API_KEY"],
    base_url=QOREBIT_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Engineering RAG Learning App",
        "User-Agent": "rag-app-learning-project/1.0",
    },
)
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
