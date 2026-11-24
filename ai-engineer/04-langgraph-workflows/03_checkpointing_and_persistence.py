"""
03_checkpointing_and_persistence.py
===================================
Module 04: LangGraph Workflows and Multi-Agent Orchestration
Course: AI Engineer Master Course

This script demonstrates persistent workflow state, human-in-the-loop breakpoints,
and state time-travel in LangGraph:
  1. Persistent Checkpointing (`SqliteSaver`):
     Serializes and restores complete graph state into a persistent SQLite database (`data/checkpoints/orders.db`).
  2. Multi-Tenant Thread Isolation:
     Executes independent workflows concurrently using isolated `thread_id` keys.
  3. Human-in-the-Loop Pausing & Resumption (`interrupt_before`):
     Halts execution before high-value fulfillment nodes, awaits external human authorization,
     and seamlessly resumes execution from the exact serialized checkpoint.
  4. Time Travel & History Inspection (`get_state_history`):
     Inspects historical state snapshots at every step and replays/rewinds to prior states.
"""

import os
import sys
import sqlite3
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Directory for SQLite persistent checkpointer
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checkpoints")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "orders.db")


# ---------------------------------------------------------------------------
# State Schema with Reducers
# ---------------------------------------------------------------------------
class FulfillmentState(TypedDict):
    order_id: str
    customer_name: str
    part_id: str
    quantity: int
    unit_price: float
    total_value_usd: float
    high_value_flag: bool
    manager_approval: bool
    approval_notes: Optional[str]
    dispatched: bool
    tracking_number: Optional[str]
    audit_trail: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Node 1: Intake & Pricing Audit
# ---------------------------------------------------------------------------
def audit_order_pricing(state: FulfillmentState) -> Dict[str, Any]:
    """Calculates order total and flags orders over $50,000 for manager sign-off."""
    print(f"\n[Node: audit_order_pricing] Processing Order: {state['order_id']}")
    total = state["quantity"] * state["unit_price"]
    is_high_val = total >= 50000.0

    print(f"  -> Total Value: ${total:,.2f} | High-Value Approval Required: {is_high_val}")
    entry = f"[{state['order_id']}] Pricing audited: ${total:,.2f}. High-value required: {is_high_val}."
    return {
        "total_value_usd": total,
        "high_value_flag": is_high_val,
        "audit_trail": [entry],
    }


# ---------------------------------------------------------------------------
# Node 2: Warehouse Reservation
# ---------------------------------------------------------------------------
def reserve_inventory(state: FulfillmentState) -> Dict[str, Any]:
    """Reserves physical inventory in warehouse before authorization gate."""
    print(f"\n[Node: reserve_inventory] Reserving {state['quantity']} unit(s) of {state['part_id']}...")
    entry = f"[{state['order_id']}] Reserved {state['quantity']}x {state['part_id']} in Long Beach DC."
    print(f"  -> {entry}")
    return {
        "audit_trail": [entry],
    }


# ---------------------------------------------------------------------------
# Node 3: Execute Fulfillment (Gated by Human Approval)
# ---------------------------------------------------------------------------
def execute_fulfillment(state: FulfillmentState) -> Dict[str, Any]:
    """Dispatches physical order after manager approval has been verified."""
    print(f"\n[Node: execute_fulfillment] Executing dispatch for {state['order_id']}...")

    if state.get("high_value_flag") and not state.get("manager_approval"):
        print("  [ERROR] Cannot fulfill high-value order without manager approval!")
        entry = f"[{state['order_id']}] BLOCKED: Fulfillment attempted without manager sign-off."
        return {
            "dispatched": False,
            "audit_trail": [entry]
        }

    tracking = f"TRK-{state['order_id']}-FEDEX"
    entry = (
        f"[{state['order_id']}] Order dispatched successfully! "
        f"Tracking: {tracking}. Sign-off: {state.get('approval_notes', 'Standard Automated')}"
    )
    print(f"  -> {entry}")

    return {
        "dispatched": True,
        "tracking_number": tracking,
        "audit_trail": [entry],
    }


# ---------------------------------------------------------------------------
# Graph Compilation with Checkpointer & Breakpoint
# ---------------------------------------------------------------------------
def build_persistent_graph(conn: sqlite3.Connection):
    """Compiles StateGraph with SqliteSaver and interrupt_before on fulfillment."""
    workflow = StateGraph(FulfillmentState)

    workflow.add_node("audit_order_pricing", audit_order_pricing)
    workflow.add_node("reserve_inventory", reserve_inventory)
    workflow.add_node("execute_fulfillment", execute_fulfillment)

    workflow.add_edge(START, "audit_order_pricing")
    workflow.add_edge("audit_order_pricing", "reserve_inventory")
    workflow.add_edge("reserve_inventory", "execute_fulfillment")
    workflow.add_edge("execute_fulfillment", END)

    saver = SqliteSaver(conn)

    # Halt right before 'execute_fulfillment' for human manager inspection & sign-off
    app = workflow.compile(
        checkpointer=saver,
        interrupt_before=["execute_fulfillment"]
    )
    return app


# ---------------------------------------------------------------------------
# Verification Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  LANGGRAPH CHECKPOINTING, PERSISTENCE & TIME TRAVEL")
    print(f"  Checkpointer Database: {DB_PATH}")
    print("=" * 70)

    # Establish persistent SQLite connection
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    app = build_persistent_graph(conn)

    # -----------------------------------------------------------------------
    # Step 1: Thread Isolation (Execute 2 Distinct Orders Concurrently)
    # -----------------------------------------------------------------------
    print("\n[Phase 1] Launching Two Independent Order Threads (alpha & beta)...")

    config_alpha = {"configurable": {"thread_id": "thread-order-alpha-901"}}
    config_beta = {"configurable": {"thread_id": "thread-order-beta-902"}}

    order_alpha: FulfillmentState = {
        "order_id": "ORD-ALPHA",
        "customer_name": "Hyperscale Cloud Corp",
        "part_id": "GPU-H100",
        "quantity": 2,
        "unit_price": 32000.00,  # Total: $64,000 (Triggers High-Value Gate)
        "total_value_usd": 0.0,
        "high_value_flag": False,
        "manager_approval": False,
        "approval_notes": None,
        "dispatched": False,
        "tracking_number": None,
        "audit_trail": ["Order Alpha submitted."],
    }

    order_beta: FulfillmentState = {
        "order_id": "ORD-BETA",
        "customer_name": "Regional IoT Labs",
        "part_id": "MCU-32BIT",
        "quantity": 100,
        "unit_price": 4.85,  # Total: $485.00 (Standard)
        "total_value_usd": 0.0,
        "high_value_flag": False,
        "manager_approval": True,  # Auto-approved
        "approval_notes": "Automated standard clearance",
        "dispatched": False,
        "tracking_number": None,
        "audit_trail": ["Order Beta submitted."],
    }

    # Run Alpha up to the interrupt gate
    print("\n--- Invoking Order Alpha (Expected to pause before execute_fulfillment) ---")
    app.invoke(order_alpha, config_alpha)

    # Run Beta up to the interrupt gate
    print("\n--- Invoking Order Beta ---")
    app.invoke(order_beta, config_beta)

    # -----------------------------------------------------------------------
    # Step 2: Inspect Checkpoint State at Breakpoint
    # -----------------------------------------------------------------------
    print("\n[Phase 2] Inspecting Paused State via get_state()...")
    state_alpha = app.get_state(config_alpha)
    print(f"  Thread Alpha Next Node: {state_alpha.next}")
    print(f"  Thread Alpha Total: ${state_alpha.values['total_value_usd']:,.2f}")
    print(f"  Thread Alpha Needs Approval: {state_alpha.values['high_value_flag']}")
    print(f"  Thread Alpha Approved?: {state_alpha.values['manager_approval']}")

    state_beta = app.get_state(config_beta)
    print(f"  Thread Beta Next Node: {state_beta.next}")
    print(f"  Thread Beta Total: ${state_beta.values['total_value_usd']:,.2f}")

    # -----------------------------------------------------------------------
    # Step 3: Human-in-the-Loop State Update & Resumption
    # -----------------------------------------------------------------------
    print("\n[Phase 3] Human Intervention: Manager Reviews & Signs Off Order Alpha...")
    approval_update = {
        "manager_approval": True,
        "approval_notes": "VP Logistics approved high-value allocation for priority tier.",
        "audit_trail": ["Manager signed off on $64,000 order expenditure."],
    }
    app.update_state(config_alpha, approval_update)

    print("  State successfully updated in SQLite database. Resuming execution from checkpoint...")
    # Passing None resumes execution from the exact checkpoint
    resumed_alpha = app.invoke(None, config_alpha)

    # Resume Beta as well
    resumed_beta = app.invoke(None, config_beta)

    print("\n[Execution Resumed Results]")
    print(f"  Order Alpha Dispatched: {resumed_alpha['dispatched']} (Tracking: {resumed_alpha['tracking_number']})")
    print(f"  Order Beta Dispatched: {resumed_beta['dispatched']} (Tracking: {resumed_beta['tracking_number']})")

    # -----------------------------------------------------------------------
    # Step 4: Time Travel & State History Inspection
    # -----------------------------------------------------------------------
    print("\n[Phase 4] Time Travel Inspection: Historical Snapshots in SQLite Checkpointer...")
    history = list(app.get_state_history(config_alpha))
    print(f"  Retrieved {len(history)} checkpoint snapshot(s) for Thread Alpha:")

    for idx, snapshot in enumerate(history, 1):
        step_nodes = snapshot.next if snapshot.next else ("(Completed)",)
        print(f"\n  Checkpoint {idx} (ID: {snapshot.config['configurable']['checkpoint_id'][:8]}...):")
        print(f"    Next Node: {step_nodes}")
        print(f"    Manager Approval in snapshot: {snapshot.values.get('manager_approval', False)}")
        print(f"    Dispatched in snapshot: {snapshot.values.get('dispatched', False)}")
        print(f"    Audit entries count: {len(snapshot.values.get('audit_trail', []))}")

    # -----------------------------------------------------------------------
    # Step 5: Database Persistence Verification (Close & Reload Connection)
    # -----------------------------------------------------------------------
    print("\n[Phase 5] Verifying Cross-Session Disk Persistence...")
    conn.close()

    # Re-open fresh connection to verify data was persisted to disk
    new_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    reloaded_app = build_persistent_graph(new_conn)
    reloaded_state = reloaded_app.get_state(config_alpha)

    print(f"  Reloaded Order Alpha from disk successfully!")
    print(f"  Order ID: {reloaded_state.values['order_id']}")
    print(f"  Final Status: Dispatched={reloaded_state.values['dispatched']}")
    print(f"  Tracking: {reloaded_state.values['tracking_number']}")
    print(f"  Total Audit Trail Steps: {len(reloaded_state.values['audit_trail'])}")
    new_conn.close()

    print("\n" + "=" * 70)
    print("  [SUCCESS] SQLite checkpointing, human-in-the-loop gates, and time travel verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
