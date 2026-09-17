"""Deployment: serve a LangGraph agent behind a FastAPI HTTP endpoint.

Every earlier demo calls `graph.invoke(...)` once, in-process, and exits.
That's fine for learning, but a real application needs the graph to keep
running as a long-lived service other programs can call over HTTP — this
file is the same supervisor graph from
07-langgraph/07_supervisor_agent.py, wrapped in a small FastAPI app instead
of a script.

The graph is built once at server startup (so the Ollama connection and
routing logic are ready before the first request) rather than once per
request. A request is a plain JSON POST instead of a Python function call,
so Pydantic models define the request/response shape FastAPI validates
against.

Run it:
    python 09-deployment/01_fastapi_agent_service.py
Then, from another terminal:
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is 24 times 7, plus 10?\"}"

Needs the same Ollama setup as 07-langgraph/ (see the README) — no API key.
"""

from typing import TypedDict

import numexpr
import uvicorn
from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

OLLAMA_MODEL = "llama3.2:3b"
ROUTES = ["math", "writing", "general"]


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


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    route: str
    answer: str


app = FastAPI(title="LangGraph Supervisor Agent Service")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = graph.invoke({"question": request.question})
    return ChatResponse(route=result["route"], answer=result["answer"])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
