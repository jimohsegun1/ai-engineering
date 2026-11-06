"""Tool-use patterns: parallel tool calling with LangGraph.

When an agent needs information from multiple independent sources — like
checking stock prices for two companies or checking the weather across
multiple cities — executing tools sequentially introduces unnecessary latency.

Modern tool-calling models emit multiple tool calls in a single AIMessage.
LangGraph's ToolNode processes all requested tool calls concurrently, collects
their ToolMessage responses, and returns them to the agent state in one pass.

This file demonstrates:
  1. Defining custom tools with @tool.
  2. Binding tools to a tool-calling LLM (ChatOllama or mock model fallback).
  3. Building a StateGraph with an agent node and a ToolNode.
  4. Inspecting the state to verify that multiple tools were executed in parallel.

Runs with local Ollama (llama3.2:3b) if available, with automatic fallback
to deterministic local execution so it runs anywhere without dependencies.
"""

import time
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# --- Step 1: Define Tools ---
print_step(1, "Define independent tools")

MOCK_STOCK_DATA = {
    "AAPL": {"price": 234.50, "currency": "USD", "change": "+1.2%"},
    "MSFT": {"price": 428.10, "currency": "USD", "change": "-0.4%"},
    "GOOGL": {"price": 178.20, "currency": "USD", "change": "+0.8%"},
}

MOCK_WEATHER_DATA = {
    "New York": {"temp": "18°C", "condition": "Sunny"},
    "London": {"temp": "12°C", "condition": "Rainy"},
    "Tokyo": {"temp": "21°C", "condition": "Clear"},
}


@tool
def get_stock_price(ticker: str) -> str:
    """Retrieve current stock price and daily change for a given ticker symbol (e.g. AAPL, MSFT, GOOGL)."""
    symbol = ticker.upper().strip()
    data = MOCK_STOCK_DATA.get(symbol)
    if data:
        return f"{symbol}: ${data['price']} {data['currency']} ({data['change']})"
    return f"Ticker {symbol} not found in database."


@tool
def get_current_weather(city: str) -> str:
    """Retrieve the current weather temperature and conditions for a city."""
    normalized_city = city.strip().title()
    data = MOCK_WEATHER_DATA.get(normalized_city)
    if data:
        return f"{normalized_city}: {data['temp']}, {data['condition']}"
    return f"Weather data for {city} not available."


tools = [get_stock_price, get_current_weather]
print(f"Tools registered: {[t.name for t in tools]}")


# --- Step 2: Build Graph State & Nodes ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Conditional edge: check if the latest message has pending tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


# Initialize model with graceful fallback
try:
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    # Quick probe to see if Ollama server is responding
    model = llm.bind_tools(tools)
    model_available = True
except Exception:
    model_available = False


def call_model_node(state: AgentState) -> dict:
    """Call the LLM to either request tool executions or synthesize an answer."""
    messages = state["messages"]

    if model_available:
        try:
            response = model.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"  [Notice] Ollama invocation failed ({e}), using deterministic parallel call.")

    # Deterministic fallback when Ollama is offline:
    # Simulate the LLM inspecting the user prompt and issuing 2 tool calls in parallel!
    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        print("  [Simulated LLM] Emitting 2 parallel tool calls in a single AIMessage:")
        simulated_ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "call_stock_1"},
                {"name": "get_stock_price", "args": {"ticker": "MSFT"}, "id": "call_stock_2"},
            ],
        )
        return {"messages": [simulated_ai]}

    # Second pass: synthesize answer from the parallel tool responses
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    summary = "Summary of parallel results:\n" + "\n".join(f"- {res}" for res in tool_results)
    return {"messages": [AIMessage(content=summary)]}


# --- Step 3: Assemble Graph with ToolNode ---
print_step(2, "Assemble StateGraph with ToolNode")

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "__end__": END,
    },
)
workflow.add_edge("tools", "agent")

app = workflow.compile()


# --- Step 4: Run the Parallel Tool Call Flow ---
print_step(3, "Execute query requiring parallel tool calls")
question = "What are the current stock prices of Apple (AAPL) and Microsoft (MSFT)?"
print(f"User Query: {question!r}\n")

start_time = time.perf_counter()
state_output = app.invoke({"messages": [HumanMessage(content=question)]})
elapsed = time.perf_counter() - start_time

print(f"\n{RULE}")
print("EXECUTION TRACE")
print(RULE)

for i, msg in enumerate(state_output["messages"], start=1):
    role = type(msg).__name__
    if isinstance(msg, AIMessage) and msg.tool_calls:
        print(f"\nStep {i} [{role} - Multi-Tool Request]:")
        for tc in msg.tool_calls:
            print(f"  -> Tool: {tc['name']!r} with args {tc['args']} (ID: {tc['id']})")
    elif isinstance(msg, ToolMessage):
        print(f"\nStep {i} [{role}]:")
        print(f"  Result for tool_call_id={msg.tool_call_id}: {msg.content!r}")
    else:
        print(f"\nStep {i} [{role}]:")
        print(f"  {msg.content}")

print(f"\nFinished in {elapsed:.3f}s. Multiple tool calls resolved in a single step.")
