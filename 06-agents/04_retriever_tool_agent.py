"""Agents: a retriever as a tool.

09_retrieval_chain.py (in 05-chains/) always retrieves before answering,
every time, whether the question needs it or not. Wrapping the same Chroma
retriever as a tool instead lets the agent decide for itself whether a given
question needs a document lookup at all — the model sees the retriever
described like any other tool and only calls it when it judges that's
useful.

In practice, llama3.2:3b doesn't always get this right: it reliably calls
the tool for the ML question (good), but for the unrelated second question
it sometimes calls the retriever anyway and still answers correctly despite
irrelevant context, and sometimes hallucinates an entirely different,
unavailable tool instead of just answering directly — even when told not to
call any tool. That's a real limitation of small models with tool-calling,
not a bug in this code.
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
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "agents_retriever_tool"
OLLAMA_MODEL = "llama3.2:3b"
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

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
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
