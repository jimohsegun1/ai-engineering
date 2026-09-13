"""LangGraph: streaming graph execution, on Qorebit instead of Ollama.

Same two stream_mode demos as 06_streaming.py, but with Qorebit's hosted
model (gpt-4o) as the LLM. This is a live Qorebit call — see the note in
06-agents/01_react_agent_qorebit.py about why agent demos don't have a free
partial run to fall back to. No `disable_streaming` here: neither node
binds tools, so there's no tool_call id for Qorebit's gateway to mangle,
and token streaming (stream_mode="messages") needs streaming left on.
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
TOPIC = "recursion"


class State(TypedDict):
    topic: str
    draft: str
    critique: str


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


def write_draft(state: State) -> dict:
    response = llm.invoke(f"Write a two-sentence explanation of {state['topic']}.")
    return {"draft": response.content}


def critique_draft(state: State) -> dict:
    response = llm.invoke(
        f"In one sentence, point out one thing that could be improved in this "
        f"explanation:\n\n{state['draft']}"
    )
    return {"critique": response.content}


print_step(1, "Build the graph: START -> write_draft -> critique_draft -> END")
graph_builder = StateGraph(State)
graph_builder.add_node("write_draft", write_draft)
graph_builder.add_node("critique_draft", critique_draft)
graph_builder.add_edge(START, "write_draft")
graph_builder.add_edge("write_draft", "critique_draft")
graph_builder.add_edge("critique_draft", END)
graph = graph_builder.compile()

print_step(2, "Stream node-by-node updates (stream_mode='updates')")
for update in graph.stream({"topic": TOPIC}, stream_mode="updates"):
    node_name = next(iter(update))
    print(f"[{node_name}] -> {update[node_name]}")

print_step(3, "Stream LLM tokens as they're generated (stream_mode='messages')")
current_node = None
for message_chunk, metadata in graph.stream({"topic": TOPIC}, stream_mode="messages"):
    node_name = metadata.get("langgraph_node", "?")
    if node_name != current_node:
        print(f"\n\n--- {node_name} ---")
        current_node = node_name
    print(message_chunk.content, end="", flush=True)
print()
