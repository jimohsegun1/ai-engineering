"""LangGraph: conditional edges.

add_conditional_edges routes execution to one of several nodes based on a
plain function's return value — the graph-based equivalent of
05-chains/04_router_chain.py's RunnableBranch. No LLM needed: the router
here is a simple keyword rule, same approach as the router chain used.
"""

from typing import TypedDict

import numexpr
from langgraph.graph import END, START, StateGraph

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


class State(TypedDict):
    question: str
    answer: str


QUESTIONS = [
    "What is 12 plus 30?",
    "What is the capital of Japan?",
]


def route_question(state: State) -> str:
    keywords = ("plus", "minus", "times", "divided", "+", "-", "*", "/")
    if any(word in state["question"].lower() for word in keywords):
        return "math"
    return "general"


def handle_math(state: State) -> dict:
    expression = state["question"].lower()
    for word, symbol in [("plus", "+"), ("minus", "-"), ("times", "*"), ("divided by", "/")]:
        expression = expression.replace(word, symbol)
    expression = "".join(ch for ch in expression if ch.isdigit() or ch in "+-*/.").strip()
    return {"answer": str(numexpr.evaluate(expression).item())}


def handle_general(state: State) -> dict:
    return {"answer": "I can only answer math questions in this demo."}


print_step(1, "Build the graph with a conditional router at the start")
graph_builder = StateGraph(State)
graph_builder.add_node("handle_math", handle_math)
graph_builder.add_node("handle_general", handle_general)
graph_builder.add_conditional_edges(
    START, route_question, {"math": "handle_math", "general": "handle_general"}
)
graph_builder.add_edge("handle_math", END)
graph_builder.add_edge("handle_general", END)
graph = graph_builder.compile()

print_step(2, "Run each question through the graph")
for question in QUESTIONS:
    result = graph.invoke({"question": question})
    route = route_question({"question": question, "answer": ""})
    print(f"\nQuestion: {question}")
    print(f"Routed to: handle_{route}")
    print(f"Answer: {result['answer']}")
