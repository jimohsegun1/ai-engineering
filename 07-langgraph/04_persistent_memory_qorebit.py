"""LangGraph: persistent memory via a checkpointer, on Qorebit instead of Ollama.

Same setup as 04_persistent_memory.py, but with Qorebit's hosted model
(gpt-4o) as the LLM. This is a live Qorebit call — see the note in
06-agents/01_react_agent_qorebit.py about why agent demos don't have a free
partial run to fall back to. No `disable_streaming` needed here: this agent
has no tools bound, so there's never a tool_call id for Qorebit's gateway
to mangle.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
THREAD_ID = "demo-conversation-1"

TURNS = [
    "My favorite number is 7.",
    "What's my favorite number plus 10?",
]

print_step(1, "Build the agent (Qorebit LLM) with a checkpointer attached")
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
agent = create_react_agent(llm, tools=[], checkpointer=MemorySaver())
config = {"configurable": {"thread_id": THREAD_ID}}

print_step(2, "Run two turns on the same thread_id")
for i, message in enumerate(TURNS, start=1):
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    print(f"\nTurn {i}")
    print(f"User: {message}")
    print(f"AI:   {result['messages'][-1].content}")
