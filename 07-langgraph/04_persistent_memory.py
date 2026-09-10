"""LangGraph: persistent memory via a checkpointer.

03_tool_calling_agent.py forgets everything between calls — every invoke()
starts fresh. Passing a checkpointer (here, MemorySaver, in-process only)
and a thread_id makes the graph save its state after every step and reload
it on the next call with the same thread_id, so a follow-up question can
refer back to an earlier one. This is the modern replacement for manually
wiring up 04-memory/'s ConversationBufferMemory around an agent.
"""

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


OLLAMA_MODEL = "llama3.2:3b"
THREAD_ID = "demo-conversation-1"

TURNS = [
    "My favorite number is 7.",
    "What's my favorite number plus 10?",
]

print_step(1, "Build the agent with a checkpointer attached")
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
agent = create_react_agent(llm, tools=[], checkpointer=MemorySaver())
config = {"configurable": {"thread_id": THREAD_ID}}

print_step(2, "Run two turns on the same thread_id")
for i, message in enumerate(TURNS, start=1):
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    print(f"\nTurn {i}")
    print(f"User: {message}")
    print(f"AI:   {result['messages'][-1].content}")
