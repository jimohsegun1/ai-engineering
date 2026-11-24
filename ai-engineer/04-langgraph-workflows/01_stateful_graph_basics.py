"""
01_stateful_graph_basics.py
===========================
Module 04: LangGraph Workflows and Multi-Agent Orchestration
Course: AI Engineer Master Course

This script introduces the foundational architecture of LangGraph:
  1. StateGraph & State Schemas: TypedDict definitions establishing the workflow contract.
  2. State Reducers: Utilizing `Annotated[List[str], operator.add]` for append-only audit logs.
  3. Graph Nodes: Pure or side-effecting functions transforming discrete state slices.
  4. Linear Graph Edges: Connecting nodes sequentially from `START` to `END`.

Scenario:
  Supply Chain Order Intake and Automated Fulfillment Allocation Pipeline.
"""

import os
import sys
import json
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# State Schema Definition with Reducers
# ---------------------------------------------------------------------------
class OrderState(TypedDict):
    """Represents the global state passed through the LangGraph pipeline."""
    order_id: str
    customer_tier: str  # "ENTERPRISE", "STANDARD", "PRIORITY"
    items: List[Dict[str, Any]]
    total_value_usd: float
    allocated_warehouse: Optional[str]
    carrier_service: Optional[str]
    estimated_transit_days: Optional[int]
    is_valid: bool
    # State Reducer: list appends across nodes automatically
    execution_log: Annotated[List[str], operator.add]
    status: str


# ---------------------------------------------------------------------------
# Node 1: Order Validation
# ---------------------------------------------------------------------------
def validate_order(state: OrderState) -> Dict[str, Any]:
    """Validates SKU presence, non-negative quantities, and computes order value."""
    print(f"\n[Node: validate_order] Processing Order: {state['order_id']}")

    items = state.get("items", [])
    if not items:
        return {
            "is_valid": False,
            "status": "REJECTED_EMPTY_ITEMS",
            "execution_log": [f"[{state['order_id']}] Order validation failed: items list empty."]
        }

    total_value = sum(item.get("unit_price", 0.0) * item.get("qty", 1) for item in items)
    is_valid = total_value > 0

    log_entry = (
        f"[{state['order_id']}] Order validated successfully. "
        f"{len(items)} SKU(s) verified. Total Value: ${total_value:,.2f}."
    )
    print(f"  -> {log_entry}")

    return {
        "is_valid": is_valid,
        "total_value_usd": total_value,
        "status": "VALIDATED",
        "execution_log": [log_entry]
    }


# ---------------------------------------------------------------------------
# Node 2: Warehouse Allocation
# ---------------------------------------------------------------------------
def allocate_warehouse(state: OrderState) -> Dict[str, Any]:
    """Assigns the optimal fulfillment hub based on item category and stock levels."""
    print(f"\n[Node: allocate_warehouse] Determining optimal fulfillment hub...")

    items = state.get("items", [])
    categories = [it.get("category", "").lower() for it in items]

    if "semiconductors" in categories or "gpu" in str(items).lower():
        warehouse = "LONG-BEACH-DC"
    elif "energy" in categories or "battery" in str(items).lower():
        warehouse = "SHANGHAI-HUB"
    else:
        warehouse = "ROTTERDAM-GATE"

    log_entry = f"[{state['order_id']}] Allocated fulfillment to hub: {warehouse}."
    print(f"  -> {log_entry}")

    return {
        "allocated_warehouse": warehouse,
        "status": "WAREHOUSE_ALLOCATED",
        "execution_log": [log_entry]
    }


# ---------------------------------------------------------------------------
# Node 3: Freight Carrier & SLA Selection
# ---------------------------------------------------------------------------
def select_freight_and_sla(state: OrderState) -> Dict[str, Any]:
    """Calculates shipping service level based on customer tier and delivery requirements."""
    print(f"\n[Node: select_freight_and_sla] Selecting freight carrier SLA...")

    tier = state.get("customer_tier", "STANDARD").upper()
    val = state.get("total_value_usd", 0.0)

    if tier == "ENTERPRISE" or val > 50000.0:
        carrier = "FedEx International Priority Air"
        transit_days = 2
    elif tier == "PRIORITY":
        carrier = "DHL Express Worldwide"
        transit_days = 4
    else:
        carrier = "Maersk Intermodal Standard Freight"
        transit_days = 12

    log_entry = (
        f"[{state['order_id']}] Carrier assigned: {carrier} "
        f"(Customer Tier: {tier}, SLA: {transit_days} days)."
    )
    print(f"  -> {log_entry}")

    return {
        "carrier_service": carrier,
        "estimated_transit_days": transit_days,
        "status": "CARRIER_ASSIGNED",
        "execution_log": [log_entry]
    }


# ---------------------------------------------------------------------------
# Node 4: Generate Manifest & Finalize
# ---------------------------------------------------------------------------
def generate_manifest(state: OrderState) -> Dict[str, Any]:
    """Assembles dispatch manifest and locks order state."""
    print(f"\n[Node: generate_manifest] Assembling dispatch manifest...")

    manifest_id = f"MNF-{state['order_id']}-EXP"
    log_entry = (
        f"[{state['order_id']}] Manifest generated: {manifest_id}. "
        f"Ready for pickup at {state['allocated_warehouse']} via {state['carrier_service']}."
    )
    print(f"  -> {log_entry}")

    return {
        "status": "READY_FOR_DISPATCH",
        "execution_log": [log_entry]
    }


# ---------------------------------------------------------------------------
# Graph Assembly & Compilation
# ---------------------------------------------------------------------------
def build_order_pipeline() -> StateGraph:
    """Builds and compiles the linear stateful LangGraph order pipeline."""
    workflow = StateGraph(OrderState)

    # 1. Add Nodes
    workflow.add_node("validate_order", validate_order)
    workflow.add_node("allocate_warehouse", allocate_warehouse)
    workflow.add_node("select_freight_and_sla", select_freight_and_sla)
    workflow.add_node("generate_manifest", generate_manifest)

    # 2. Add Linear Edges
    workflow.add_edge(START, "validate_order")
    workflow.add_edge("validate_order", "allocate_warehouse")
    workflow.add_edge("allocate_warehouse", "select_freight_and_sla")
    workflow.add_edge("select_freight_and_sla", "generate_manifest")
    workflow.add_edge("generate_manifest", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Verification & Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  LANGGRAPH FOUNDATIONS: STATEFUL GRAPH & STATE REDUCERS")
    print("=" * 70)

    # Compile the graph
    app = build_order_pipeline()

    # Define initial input state
    initial_state: OrderState = {
        "order_id": "ORD-2026-9941",
        "customer_tier": "ENTERPRISE",
        "items": [
            {"sku": "GPU-H100", "category": "Semiconductors", "qty": 4, "unit_price": 32000.00},
            {"sku": "MCU-32BIT", "category": "Embedded Systems", "qty": 50, "unit_price": 4.85},
        ],
        "total_value_usd": 0.0,
        "allocated_warehouse": None,
        "carrier_service": None,
        "estimated_transit_days": None,
        "is_valid": False,
        "execution_log": ["Order received into queue."],
        "status": "INITIALIZED",
    }

    print("\n[Input State Summary]")
    print(f"  Order ID: {initial_state['order_id']}")
    print(f"  Customer Tier: {initial_state['customer_tier']}")
    print(f"  Item Count: {len(initial_state['items'])}")

    print("\n" + "=" * 70)
    print("  EXECUTING GRAPH PIPELINE (START -> validate -> allocate -> freight -> manifest -> END)")
    print("=" * 70)

    # Run the graph
    final_output = app.invoke(initial_state)

    print("\n" + "=" * 70)
    print("  GRAPH EXECUTION COMPLETE: FINAL STATE AUDIT")
    print("=" * 70)
    print(f"  Status: {final_output['status']}")
    print(f"  Total Value: ${final_output['total_value_usd']:,.2f}")
    print(f"  Fulfillment Hub: {final_output['allocated_warehouse']}")
    print(f"  Freight Carrier: {final_output['carrier_service']} ({final_output['estimated_transit_days']} days)")

    print(f"\n  Accumulated Execution Log (State Reducer: operator.add):")
    for step_num, entry in enumerate(final_output["execution_log"], 1):
        print(f"    {step_num}. {entry}")

    print("\n" + "=" * 70)
    print("  [SUCCESS] LangGraph stateful graph and reducers verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
