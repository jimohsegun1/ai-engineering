"""Tool-use patterns: tool chaining with LangGraph.

Tool chaining refers to passing the output of one tool directly as the input
to another tool. In complex agentic systems, doing this entirely through
the LLM ("call tool A -> model reads output -> model calls tool B") can be
inefficient, slow, and prone to hallucination when handling large JSON payloads.

This file demonstrates two approaches to tool chaining:
  1. Agent-orchestrated chaining: The model receives Tool A's output in state
     and dynamically invokes Tool B based on the result.
  2. Graph-orchestrated chaining: A deterministic sub-pipeline where Tool A's
     structured output flows directly into Tool B as typed state without
     wasting LLM tokens, followed by an LLM synthesis step.

Scenario: Customer Financial Risk Investigation
  - Step 1: fetch_user_profile(user_id) -> retrieves account_id & country
  - Step 2: fetch_transactions(account_id) -> retrieves recent transaction records
  - Step 3: compute_risk_score(transactions, country) -> calculates anomaly score
  - Step 4: Executive synthesis by the agent.
"""

from typing import Annotated, Sequence, TypedDict

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


# --- Step 1: Define Chained Tools ---
print_step(1, "Define specialized tools designed for chaining")

MOCK_USERS = {
    101: {"name": "Alice Johnson", "account_id": "ACC-9921", "country": "US"},
    102: {"name": "Bob Smith", "account_id": "ACC-7744", "country": "NG"},
}

MOCK_TRANSACTIONS = {
    "ACC-9921": [
        {"id": "TX-1", "amount": 120.0, "category": "retail"},
        {"id": "TX-2", "amount": 45.5, "category": "groceries"},
    ],
    "ACC-7744": [
        {"id": "TX-3", "amount": 9500.0, "category": "crypto_transfer"},
        {"id": "TX-4", "amount": 4200.0, "category": "wire_international"},
    ],
}


@tool
def fetch_user_profile(user_id: int) -> dict:
    """Fetch user account metadata including account_id and country of registration."""
    profile = MOCK_USERS.get(user_id)
    if profile:
        return {"status": "success", "profile": profile}
    return {"status": "error", "message": f"User {user_id} not found"}


@tool
def fetch_account_transactions(account_id: str) -> dict:
    """Fetch recent transaction ledger for an account ID."""
    txs = MOCK_TRANSACTIONS.get(account_id, [])
    return {"account_id": account_id, "transactions": txs, "count": len(txs)}


@tool
def compute_risk_score(account_id: str, total_volume: float, has_high_risk_categories: bool) -> dict:
    """Compute financial risk rating based on transaction volume and category flags."""
    risk_level = "LOW"
    score = 15

    if total_volume > 5000.0 or has_high_risk_categories:
        risk_level = "HIGH"
        score = 88
    elif total_volume > 1000.0:
        risk_level = "MEDIUM"
        score = 52

    return {
        "account_id": account_id,
        "risk_level": risk_level,
        "risk_score": score,
        "requires_manual_review": risk_level == "HIGH",
    }


tools = [fetch_user_profile, fetch_account_transactions, compute_risk_score]


# --- Step 2: Build Deterministic Chaining StateGraph ---
print_step(2, "Assemble Graph-orchestrated Tool Chain (Pipeline + Synthesis)")

class ChainedInvestigationState(TypedDict):
    user_id: int
    account_id: str
    transactions: list
    total_volume: float
    has_high_risk: bool
    risk_assessment: dict
    final_summary: str


def lookup_profile_node(state: ChainedInvestigationState) -> dict:
    """Tool 1: lookup profile."""
    uid = state["user_id"]
    print(f"\n[Node: Profile Lookup] Calling fetch_user_profile(user_id={uid})...")
    res = fetch_user_profile.invoke({"user_id": uid})
    acc_id = res["profile"]["account_id"]
    print(f"  -> Found account_id: {acc_id}")
    return {"account_id": acc_id}


def lookup_transactions_node(state: ChainedInvestigationState) -> dict:
    """Tool 2: fetch transactions using account_id from Tool 1."""
    acc_id = state["account_id"]
    print(f"\n[Node: Transaction Lookup] Chaining output -> fetch_account_transactions(account_id={acc_id!r})...")
    res = fetch_account_transactions.invoke({"account_id": acc_id})
    txs = res["transactions"]
    vol = sum(t["amount"] for t in txs)
    high_risk = any(t["category"] in ("crypto_transfer", "wire_international") for t in txs)
    print(f"  -> Retrieved {len(txs)} transactions, total volume: ${vol:.2f}")
    return {"transactions": txs, "total_volume": vol, "has_high_risk": high_risk}


def compute_risk_node(state: ChainedInvestigationState) -> dict:
    """Tool 3: evaluate risk using aggregated transaction metrics from Tool 2."""
    acc_id = state["account_id"]
    vol = state["total_volume"]
    high_risk = state["has_high_risk"]
    print(f"\n[Node: Risk Scoring] Chaining metrics -> compute_risk_score(vol=${vol:.2f}, high_risk={high_risk})...")
    risk_res = compute_risk_score.invoke({
        "account_id": acc_id,
        "total_volume": vol,
        "has_high_risk_categories": high_risk,
    })
    print(f"  -> Risk Rating: {risk_res['risk_level']} (Score: {risk_res['risk_score']})")
    return {"risk_assessment": risk_res}


def synthesize_report_node(state: ChainedInvestigationState) -> dict:
    """Final synthesis step: formats executive summary."""
    risk = state["risk_assessment"]
    summary = (
        f"INVESTIGATION REPORT FOR USER {state['user_id']}:\n"
        f"Account ID: {state['account_id']}\n"
        f"Transactions Analyzed: {len(state['transactions'])}\n"
        f"Total Volume: ${state['total_volume']:,.2f}\n"
        f"Risk Level: {risk['risk_level']} (Score: {risk['risk_score']}/100)\n"
        f"Manual Review Needed: {risk['requires_manual_review']}"
    )
    return {"final_summary": summary}


# Compile deterministic pipeline graph
chain_flow = StateGraph(ChainedInvestigationState)
chain_flow.add_node("lookup_profile", lookup_profile_node)
chain_flow.add_node("lookup_transactions", lookup_transactions_node)
chain_flow.add_node("compute_risk", compute_risk_node)
chain_flow.add_node("synthesize_report", synthesize_report_node)

chain_flow.add_edge(START, "lookup_profile")
chain_flow.add_edge("lookup_profile", "lookup_transactions")
chain_flow.add_edge("lookup_transactions", "compute_risk")
chain_flow.add_edge("compute_risk", "synthesize_report")
chain_flow.add_edge("synthesize_report", END)

investigation_pipeline = chain_flow.compile()


# --- Step 3: Run the Chained Pipeline ---
print_step(3, "Execute Chained Pipeline on User 102 (High-Risk Profile)")
output = investigation_pipeline.invoke({
    "user_id": 102,
    "account_id": "",
    "transactions": [],
    "total_volume": 0.0,
    "has_high_risk": False,
    "risk_assessment": {},
    "final_summary": "",
})

print(f"\n{RULE}")
print("PIPELINE RESULT")
print(RULE)
print(output["final_summary"])
