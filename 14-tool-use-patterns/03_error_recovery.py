"""Tool-use patterns: error recovery and self-healing in LangGraph.

When an LLM invokes tools, failures frequently occur in production:
  - Malformed parameters (e.g. date format '06/11/2026' instead of '2026-11-06')
  - Out-of-range numerical values or invalid enum keys
  - Upstream network timeouts or rate limits

In a fragile agent architecture, an unhandled tool exception crashes the process.
In a self-healing LangGraph architecture:
  1. Tool errors are caught safely within the tool or execution wrapper.
  2. The failure is returned to the agent as a diagnostic ToolMessage containing
     the exact reason and the expected schema/format.
  3. The agent reads the error feedback, corrects its arguments, and retries.
  4. Loop protection (MAX_RETRIES) guarantees the graph terminates even if the
     failure is unrecoverable.

Runs locally with Ollama (llama3.2:3b) with automatic fallback simulation
so the full error-correction cycle is demonstrated under all environments.
"""

from datetime import datetime
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# --- Step 1: Define Tool with Strict Validation ---
print_step(1, "Define tool with strict argument validation")


@tool
def calculate_currency_conversion(amount: float, from_curr: str, to_curr: str, date: str) -> str:
    """Convert an amount from one currency to another on a specific date.
    
    Args:
        amount: Numerical amount to convert (must be positive).
        from_curr: 3-letter currency code (e.g. USD, EUR, GBP).
        to_curr: 3-letter currency code (e.g. USD, EUR, GBP).
        date: Valuation date in strict ISO format 'YYYY-MM-DD'.
    """
    if amount <= 0:
        raise ValueError(f"Invalid amount '{amount}': must be greater than zero.")

    # Strict date parsing requirement:
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Invalid date format '{date}'. REQUIRED format: 'YYYY-MM-DD' (e.g. '2026-11-06')."
        )

    # Mock conversion rates
    rates = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("GBP", "USD"): 1.28,
        ("USD", "GBP"): 0.78,
    }
    pair = (from_curr.upper(), to_curr.upper())
    rate = rates.get(pair, 1.0)
    converted = amount * rate
    return f"{amount} {from_curr.upper()} = {converted:.2f} {to_curr.upper()} (rate {rate} on {parsed_date.strftime('%Y-%m-%d')})"


# Robust wrapper: catches tool exceptions and converts them to diagnostic ToolMessages
def safe_tool_executor(tool_call: dict) -> ToolMessage:
    """Execute a tool call safely, returning a diagnostic error message on failure."""
    call_id = tool_call["id"]
    args = tool_call["args"]

    try:
        output = calculate_currency_conversion.invoke(args)
        return ToolMessage(content=str(output), tool_call_id=call_id)
    except Exception as e:
        error_msg = f"TOOL EXECUTION ERROR: {str(e)}"
        print(f"  [Tool Exception Trapped] -> {error_msg}")
        return ToolMessage(content=error_msg, tool_call_id=call_id)


# --- Step 2: Build Self-Healing Graph ---
class AgentRecoveryState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int


MAX_RETRIES = 2

# Probe Ollama
try:
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    model = llm.bind_tools([calculate_currency_conversion])
    model_available = True
except Exception:
    model_available = False


def agent_node(state: AgentRecoveryState) -> dict:
    """Agent node that inspects previous messages, including error feedback."""
    messages = state["messages"]

    if model_available:
        try:
            res = model.invoke(messages)
            return {"messages": [res]}
        except Exception:
            pass

    # Deterministic simulation of the self-healing cycle:
    # 1. Turn 1: Model makes an intentional format mistake (date: '11/06/2026' instead of ISO 'YYYY-MM-DD').
    # 2. Turn 2: Model receives ToolMessage with error instructions, corrects date to '2026-11-06', and succeeds.
    last_msg = messages[-1]

    if isinstance(last_msg, HumanMessage):
        print("\n  [Agent Decision 1] Calling tool with initial attempt (deliberately malformed date)...")
        bad_call = AIMessage(
            content="",
            tool_calls=[{
                "name": "calculate_currency_conversion",
                "args": {"amount": 500.0, "from_curr": "USD", "to_curr": "EUR", "date": "11/06/2026"},
                "id": "conv_call_1",
            }],
        )
        return {"messages": [bad_call]}

    if isinstance(last_msg, ToolMessage) and "ERROR" in last_msg.content:
        print("\n  [Agent Decision 2 - Self-Correction] Detected tool error feedback in state.")
        print(f"  Received: {last_msg.content}")
        print("  Applying fix -> Reformatting date to '2026-11-06' and retrying call...")
        good_call = AIMessage(
            content="",
            tool_calls=[{
                "name": "calculate_currency_conversion",
                "args": {"amount": 500.0, "from_curr": "USD", "to_curr": "EUR", "date": "2026-11-06"},
                "id": "conv_call_2",
            }],
        )
        return {"messages": [good_call]}

    # Turn 3: Tool succeeded, emit final user answer
    tool_content = [m.content for m in messages if isinstance(m, ToolMessage) and "ERROR" not in m.content]
    return {"messages": [AIMessage(content=f"Conversion completed successfully: {tool_content[-1]}")]}


def tool_execution_node(state: AgentRecoveryState) -> dict:
    """Executes requested tool calls with safe exception handling."""
    last_msg = state["messages"][-1]
    retries = state.get("retry_count", 0)

    tool_messages = []
    has_error = False

    for tc in getattr(last_msg, "tool_calls", []):
        tmsg = safe_tool_executor(tc)
        tool_messages.append(tmsg)
        if "ERROR" in tmsg.content:
            has_error = True

    new_retries = retries + 1 if has_error else retries
    return {"messages": tool_messages, "retry_count": new_retries}


def route_after_agent(state: AgentRecoveryState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "__end__"


def route_after_tools(state: AgentRecoveryState) -> Literal["agent", "__end__"]:
    retries = state.get("retry_count", 0)
    if retries > MAX_RETRIES:
        print(f"  [Guard] Max retries ({MAX_RETRIES}) reached. Terminating loop to prevent infinite retry.")
        return "__end__"
    return "agent"


# Assemble Recovery Workflow
workflow = StateGraph(AgentRecoveryState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_execution_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "__end__": END})
workflow.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "__end__": END})

app = workflow.compile()


# --- Step 3: Run Error-Recovery Cycle ---
print_step(2, "Run self-healing tool loop")
query = "Convert 500 USD to EUR on November 6, 2026."
print(f"User Request: {query!r}")

final_state = app.invoke({"messages": [HumanMessage(content=query)], "retry_count": 0})

print(f"\n{RULE}")
print("RECOVERY TRACE")
print(RULE)
for i, m in enumerate(final_state["messages"], start=1):
    m_type = type(m).__name__
    if isinstance(m, AIMessage) and m.tool_calls:
        print(f"\nStep {i} [{m_type}]: Tool call requested -> {m.tool_calls[0]['args']}")
    elif isinstance(m, ToolMessage):
        status = "FAILED" if "ERROR" in m.content else "SUCCESS"
        print(f"\nStep {i} [{m_type} - {status}]: {m.content}")
    else:
        print(f"\nStep {i} [{m_type}]: {m.content}")

print(f"\nFinal Status: Recovery loop resolved the issue in {final_state['retry_count']} retry attempt(s).")
