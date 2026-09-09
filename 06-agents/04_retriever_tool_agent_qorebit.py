"""Agents: a retriever as a tool, on Qorebit instead of Ollama.

Same retriever-as-tool setup as 04_retriever_tool_agent.py, but with
Qorebit's hosted model (gpt-4o) as the LLM. Embeddings still use the free
local HuggingFace model — only generation goes through Qorebit. This is a
live Qorebit call for both questions; unlike llama3.2:3b (which sometimes
called the tool anyway or hallucinated an unavailable one), gpt-4o reliably
calls search_ml_notes for the RAG question and answers the geography
question directly with no tool call at all.

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
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "agents_retriever_tool_qorebit"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
TOP_K = 3

QUESTIONS = [
    "What is Retrieval-Augmented Generation?",
    "What is the capital of France?",
]

print_step(1, "Load, chunk, and embed the document")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

print_step(2, "Wrap the retriever as a tool the agent can choose to call")
retriever_tool = create_retriever_tool(
    retriever,
    name="search_ml_notes",
    description="Search a set of notes about machine learning and RAG concepts. "
                 "Use this for questions about ML/AI terminology or techniques.",
)

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
tools = [retriever_tool]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Only call search_ml_notes if the question is "
                   "specifically about machine learning, AI, or RAG concepts. For anything else "
                   "(general knowledge, geography, etc.) answer directly from what you already "
                   "know — do not call any tool."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print_step(3, "Ask a question that needs the retriever, then one that doesn't")
for question in QUESTIONS:
    result = agent_executor.invoke({"input": question})
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['output']}")
