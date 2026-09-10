"""LangGraph: the basic StateGraph mechanic.

Every LangGraph app is a graph of nodes (plain functions) that read and
write a shared state, connected by edges. This file has no LLM at all —
just two nodes run in sequence — to show the underlying mechanic before any
of the later files add a model into the mix.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


class State(TypedDict):
    text: str
    word_count: int
    category: str


TEXT = "LangGraph models an application as a graph of nodes sharing one state object."


def count_words(state: State) -> dict:
    return {"word_count": len(state["text"].split())}


def classify_length(state: State) -> dict:
    count = state["word_count"]
    if count < 8:
        category = "short"
    elif count < 16:
        category = "medium"
    else:
        category = "long"
    return {"category": category}


print_step(1, "Build the graph: START -> count_words -> classify_length -> END")
graph_builder = StateGraph(State)
graph_builder.add_node("count_words", count_words)
graph_builder.add_node("classify_length", classify_length)
graph_builder.add_edge(START, "count_words")
graph_builder.add_edge("count_words", "classify_length")
graph_builder.add_edge("classify_length", END)
graph = graph_builder.compile()

print_step(2, "Run the graph")
result = graph.invoke({"text": TEXT})
print(f"Input text: {TEXT!r}")
print(f"Word count: {result['word_count']}")
print(f"Category:   {result['category']}")
