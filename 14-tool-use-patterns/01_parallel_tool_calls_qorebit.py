"""Tool-use patterns: parallel tool calling with Qorebit (GPT-4o).

Hosted twin of 01_parallel_tool_calls.py: uses Qorebit's OpenAI-compatible
endpoint running GPT-4o.

Modern frontier models like GPT-4o natively identify when multiple independent
tools can be called in parallel and return multiple tool_calls within a single
completion response. LangGraph's ToolNode executes them concurrently.

Needs QOREBIT_API_KEY in .env.
"""

import os
import time
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

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


# --- Step 2: Assemble Graph ---
print_step(2, "Initialize Qorebit GPT-4o and bind tools")

llm = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0,
    api_key=os.environ.get("QOREBIT_API_KEY", "qb_placeholder"),
    base_url="https://api.qorebit.ai/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Engineering Tool Use Learning Project",
        "User-Agent": "rag-app-learning-project/1.0",
    },
)
model_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


def call_model_node(state: AgentState) -> dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


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


# --- Step 3: Run Execution ---
print_step(3, "Execute query requiring parallel tool calls")
question = "What is the current stock price of Apple (AAPL) and the current weather in Tokyo?"
print(f"User Query: {question!r}\n")

if "QOREBIT_API_KEY" not in os.environ or not os.environ["QOREBIT_API_KEY"].startswith("qb_"):
    print("Note: QOREBIT_API_KEY not found in .env. Showing compiled graph structure:")
    print("Graph compiled successfully with nodes:", list(workflow.nodes.keys()))
    print("Add your QOREBIT_API_KEY to .env to execute live requests.")
else:
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
                print(f"  -> Tool: {tc['name']!r} with args {tc['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"\nStep {i} [{role}]: {msg.content}")
        else:
            print(f"\nStep {i} [{role}]: {msg.content}")

    print(f"\nFinished in {elapsed:.3f}s.")
