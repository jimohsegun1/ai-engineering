"""LangGraph: streaming graph execution.

Every earlier file in this folder calls graph.invoke() and only sees the
final state once the whole graph has finished. For anything user-facing
(a chat UI, a progress indicator) you usually want to see results as they
happen instead. graph.stream() supports several stream_mode values; this
file demonstrates the two most common ones:

- stream_mode="updates": yields one event per node, right after that node
  finishes, containing only the fields that node changed. Good for showing
  step-by-step progress ("drafting...", "critiquing...").
- stream_mode="messages": yields individual LLM token chunks as they're
  generated, across every node in the graph, paired with metadata about
  which node produced them. Good for the classic word-by-word typing effect.
"""

from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


OLLAMA_MODEL = "llama3.2:3b"
TOPIC = "recursion"


class State(TypedDict):
    topic: str
    draft: str
    critique: str


llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


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
