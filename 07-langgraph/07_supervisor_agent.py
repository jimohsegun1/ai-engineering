"""LangGraph: supervisor multi-agent pattern.

02_conditional_graph.py routes with a hand-written rule (does the text
contain digits and an operator?). A supervisor is the more general version
of that idea: an LLM call decides which specialist should handle the
request, instead of a fixed rule, so it can scale past two branches without
hand-coding every routing case. This file has a supervisor node and three
specialist worker nodes (math, writing, general knowledge); the supervisor
only picks a route, it never answers the question itself.
"""

import numexpr
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


OLLAMA_MODEL = "llama3.2:3b"
ROUTES = ["math", "writing", "general"]

QUESTIONS = [
    "What is 24 times 7, plus 10?",
    "Rewrite 'the meeting got moved, again, sorry everyone' more professionally.",
    "What is the capital of Japan?",
]


class State(TypedDict):
    question: str
    route: str
    answer: str


llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


def supervisor(state: State) -> dict:
    prompt = (
        "Classify the question into exactly one category: math, writing, or general. "
        "Reply with only that one word.\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt).content.strip().lower()
    route = next((r for r in ROUTES if r in response), "general")
    print(f"Supervisor routed to: {route}")
    return {"route": route}


def math_specialist(state: State) -> dict:
    expression = state["question"].lower()
    for word, symbol in [("plus", "+"), ("minus", "-"), ("times", "*"), ("divided by", "/")]:
        expression = expression.replace(word, symbol)
    expression = "".join(ch for ch in expression if ch.isdigit() or ch in "+-*/.").strip()
    result = numexpr.evaluate(expression).item()
    return {"answer": str(result)}


def writing_specialist(state: State) -> dict:
    prompt = f"Rewrite this message in a clear, professional tone:\n\n{state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}


def general_specialist(state: State) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content}


print_step(1, "Build the graph: supervisor routes to one of three specialists")
graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("math_specialist", math_specialist)
graph_builder.add_node("writing_specialist", writing_specialist)
graph_builder.add_node("general_specialist", general_specialist)
graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["route"],
    {
        "math": "math_specialist",
        "writing": "writing_specialist",
        "general": "general_specialist",
    },
)
graph_builder.add_edge("math_specialist", END)
graph_builder.add_edge("writing_specialist", END)
graph_builder.add_edge("general_specialist", END)
graph = graph_builder.compile()

print_step(2, "Run each question through the graph")
for question in QUESTIONS:
    result = graph.invoke({"question": question})
    print(f"\nQuestion: {question}")
    print(f"Answer:   {result['answer']}")
