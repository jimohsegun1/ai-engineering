"""
02_conditional_routing_and_cycles.py
====================================
Module 04: LangGraph Workflows and Multi-Agent Orchestration
Course: AI Engineer Master Course

This script demonstrates advanced dynamic graph topologies in LangGraph:
  1. Conditional Edges (`add_conditional_edges`): Branching routing logic based on runtime state.
  2. Iterative Feedback Cycles (Graph Loops): Self-correcting revision loops between generator and critic.
  3. Quality Gates: Enforcing risk and cost thresholds before approving execution.
  4. Terminal Escalation: Graceful failover to human oversight when cycle limits are reached.

Scenario:
  Autonomous Supply Chain Disruption Mitigation & Iterative Plan Refinement.
"""

import os
import sys
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# State Schema with Iteration Tracking & Critic Feedback
# ---------------------------------------------------------------------------
class DisruptionState(TypedDict):
    incident_id: str
    affected_part: str
    origin_hub: str
    destination_hub: str
    disruption_cause: str
    iteration_count: int
    max_iterations: int
    proposed_plan: Dict[str, Any]
    critic_notes: Optional[str]
    risk_score: float  # 0 to 100 (Threshold for auto-approval <= 20.0)
    cost_penalty_usd: float
    status: str
    audit_trail: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Node 1: Disruption Intake & Analysis
# ---------------------------------------------------------------------------
def analyze_disruption(state: DisruptionState) -> Dict[str, Any]:
    """Parses incoming disruption parameters and establishes baseline constraints."""
    print(f"\n[Node: analyze_disruption] Incident: {state['incident_id']}")
    print(f"  Cause: {state['disruption_cause']}")
    print(f"  Route: {state['origin_hub']} -> {state['destination_hub']} (Part: {state['affected_part']})")

    entry = (
        f"[{state['incident_id']}] Triage initiated for {state['affected_part']} "
        f"due to '{state['disruption_cause']}'."
    )
    return {
        "status": "ANALYZED",
        "iteration_count": 0,
        "audit_trail": [entry]
    }


# ---------------------------------------------------------------------------
# Node 2: Mitigation Planner (Self-Refines Based on Critic Feedback)
# ---------------------------------------------------------------------------
def propose_mitigation(state: DisruptionState) -> Dict[str, Any]:
    """Generates or refines a rerouting/reallocation plan based on feedback."""
    iteration = state.get("iteration_count", 0) + 1
    critic_feedback = state.get("critic_notes")

    print(f"\n[Node: propose_mitigation] Generating Plan (Iteration {iteration}/{state['max_iterations']})...")

    if iteration == 1:
        # First attempt: Naive expedited air routing (very fast, but exorbitant cost & high carbon)
        plan = {
            "version": iteration,
            "strategy": "Direct Emergency Air Charter",
            "carrier": "Charter Airways Global",
            "reroute_path": "SHANGHAI -> TOKYO -> LONG_BEACH",
            "transit_days": 2,
            "air_freight_premium": 72000.0,
            "customs_pre_cleared": False,
        }
    elif iteration == 2:
        # Second attempt: Address critic notes (reduce cost by blending scheduled air cargo)
        plan = {
            "version": iteration,
            "strategy": "Scheduled Commercial Air Cargo + Bonded Transit",
            "carrier": "Lufthansa Cargo Express",
            "reroute_path": "SHANGHAI -> FRANKFURT -> LONG_BEACH",
            "transit_days": 4,
            "air_freight_premium": 28000.0,
            "customs_pre_cleared": True,
        }
    else:
        # Third attempt: Fully optimized multimodal fast-sea express
        plan = {
            "version": iteration,
            "strategy": "Fast-Vessel Ocean Express + Expedited Rail",
            "carrier": "Matson Express Logistics",
            "reroute_path": "SHANGHAI -> OAKLAND -> LONG_BEACH",
            "transit_days": 10,
            "air_freight_premium": 9500.0,
            "customs_pre_cleared": True,
        }

    print(f"  -> Proposed Strategy: {plan['strategy']}")
    print(f"     Estimated Transit: {plan['transit_days']} days | Premium: ${plan['air_freight_premium']:,.2f}")

    entry = f"[Iteration {iteration}] Formulated plan v{iteration}: '{plan['strategy']}'."
    return {
        "iteration_count": iteration,
        "proposed_plan": plan,
        "status": "PLAN_PROPOSED",
        "audit_trail": [entry]
    }


# ---------------------------------------------------------------------------
# Node 3: Risk & Cost Critic Evaluator
# ---------------------------------------------------------------------------
def evaluate_risk_and_cost(state: DisruptionState) -> Dict[str, Any]:
    """Evaluates the proposed plan against financial risk and operational compliance."""
    plan = state["proposed_plan"]
    iteration = state["iteration_count"]
    print(f"\n[Node: evaluate_risk_and_cost] Evaluating Plan v{plan['version']}...")

    # Scoring heuristic based on strategy maturity and risk mitigations
    if plan["air_freight_premium"] > 50000.0 and not plan.get("customs_pre_cleared"):
        risk_score = 65.0
        critic_notes = (
            "REJECT: Unbudgeted air charter cost ($72,000) exceeds threshold. "
            "Missing customs pre-clearance creates severe demurrage risk at Long Beach."
        )
    elif plan["transit_days"] > 8:
        risk_score = 35.0
        critic_notes = "WARNING: Transit duration exceeds customer SLA of 5 days."
    else:
        risk_score = 15.0  # Acceptable!
        critic_notes = (
            "APPROVED: Plan meets SLA (4 days), stays within budget ceiling ($28,000), "
            "and includes customs pre-clearance."
        )

    print(f"  -> Risk Score: {risk_score}/100 (Threshold <= 20.0)")
    print(f"  -> Critic Feedback: {critic_notes}")

    entry = f"[Critic Eval v{iteration}] Risk Score: {risk_score:.1f}. Feedback: {critic_notes}"
    return {
        "risk_score": risk_score,
        "cost_penalty_usd": plan["air_freight_premium"],
        "critic_notes": critic_notes,
        "status": "EVALUATION_COMPLETE",
        "audit_trail": [entry]
    }


# ---------------------------------------------------------------------------
# Conditional Routing Function: Quality Gate
# ---------------------------------------------------------------------------
def route_quality_gate(state: DisruptionState) -> str:
    """Routes to approval, revision loop, or human escalation."""
    risk_score = state.get("risk_score", 100.0)
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if risk_score <= 20.0:
        print("\n>>> [Routing Decision] Risk Score <= 20.0 -> Routing to 'approve_and_dispatch'")
        return "approve_and_dispatch"
    elif iteration < max_iter:
        print(f"\n>>> [Routing Decision] Risk Score {risk_score:.1f} > 20.0 and Iteration {iteration} < {max_iter} -> CYCLING BACK to 'propose_mitigation'")
        return "propose_mitigation"
    else:
        print(f"\n>>> [Routing Decision] Max iterations ({max_iter}) reached without acceptable risk -> Routing to 'escalate_to_human'")
        return "escalate_to_human"


# ---------------------------------------------------------------------------
# Node 4: Approve & Dispatch (Terminal Success)
# ---------------------------------------------------------------------------
def approve_and_dispatch(state: DisruptionState) -> Dict[str, Any]:
    """Approves execution of the vetted mitigation plan."""
    print(f"\n[Node: approve_and_dispatch] Locking in approved mitigation plan...")
    plan = state["proposed_plan"]

    entry = (
        f"[{state['incident_id']}] PLAN APPROVED: '{plan['strategy']}' "
        f"committed to dispatch after {state['iteration_count']} iteration(s)."
    )
    print(f"  -> {entry}")

    return {
        "status": "APPROVED_FOR_DISPATCH",
        "audit_trail": [entry]
    }


# ---------------------------------------------------------------------------
# Node 5: Escalate to Human (Terminal Failover)
# ---------------------------------------------------------------------------
def escalate_to_human(state: DisruptionState) -> Dict[str, Any]:
    """Escalates to operations manager when automated cycles fail to reach consensus."""
    print(f"\n[Node: escalate_to_human] Alerting Operations Commander...")
    entry = (
        f"[{state['incident_id']}] ESCALATION: Auto-mitigation could not resolve risk "
        f"within {state['max_iterations']} cycles. Handing off to Human Ops."
    )
    print(f"  -> {entry}")

    return {
        "status": "ESCALATED_TO_HUMAN",
        "audit_trail": [entry]
    }


# ---------------------------------------------------------------------------
# Graph Construction with Conditional Branching & Cycles
# ---------------------------------------------------------------------------
def build_disruption_resolver_graph() -> StateGraph:
    """Builds a state graph containing conditional edges and cyclic loops."""
    workflow = StateGraph(DisruptionState)

    # Add Nodes
    workflow.add_node("analyze_disruption", analyze_disruption)
    workflow.add_node("propose_mitigation", propose_mitigation)
    workflow.add_node("evaluate_risk_and_cost", evaluate_risk_and_cost)
    workflow.add_node("approve_and_dispatch", approve_and_dispatch)
    workflow.add_node("escalate_to_human", escalate_to_human)

    # Linear Edges
    workflow.add_edge(START, "analyze_disruption")
    workflow.add_edge("analyze_disruption", "propose_mitigation")
    workflow.add_edge("propose_mitigation", "evaluate_risk_and_cost")

    # Conditional Edges (Cycles & Branches)
    workflow.add_conditional_edges(
        source="evaluate_risk_and_cost",
        path=route_quality_gate,
        path_map={
            "approve_and_dispatch": "approve_and_dispatch",
            "propose_mitigation": "propose_mitigation",   # The feedback cycle!
            "escalate_to_human": "escalate_to_human",
        }
    )

    # Terminal Edges
    workflow.add_edge("approve_and_dispatch", END)
    workflow.add_edge("escalate_to_human", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Main Verification Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  LANGGRAPH ADVANCED: CONDITIONAL ROUTING & REFLECTION CYCLES")
    print("=" * 70)

    # Compile the cyclic graph
    graph = build_disruption_resolver_graph()

    # Define initial disruption scenario
    initial_incident: DisruptionState = {
        "incident_id": "INC-ROTTERDAM-STRIKE-88",
        "affected_part": "GPU-H100",
        "origin_hub": "SHANGHAI-HUB",
        "destination_hub": "LONG-BEACH-DC",
        "disruption_cause": "Port terminal crane breakdown and 10-day berth backlog",
        "iteration_count": 0,
        "max_iterations": 3,
        "proposed_plan": {},
        "critic_notes": None,
        "risk_score": 100.0,
        "cost_penalty_usd": 0.0,
        "status": "INITIALIZED",
        "audit_trail": ["Incident created from IoT monitoring alert."],
    }

    print("\n[Executing Graph with Iterative Feedback Loop]")
    final_state = graph.invoke(initial_incident)

    print("\n" + "=" * 70)
    print("  WORKFLOW TERMINATION SUMMARY")
    print("=" * 70)
    print(f"  Final Status: {final_state['status']}")
    print(f"  Total Cycles Completed: {final_state['iteration_count']}")
    print(f"  Final Risk Score: {final_state['risk_score']}/100")
    print(f"  Approved Strategy: {final_state['proposed_plan'].get('strategy', 'N/A')}")
    print(f"  Transit Days: {final_state['proposed_plan'].get('transit_days', 'N/A')}")
    print(f"  Freight Premium: ${final_state['cost_penalty_usd']:,.2f}")

    print("\n  Complete Step-by-Step Audit Trail:")
    for idx, log in enumerate(final_state["audit_trail"], 1):
        print(f"    {idx}. {log}")

    print("\n" + "=" * 70)
    print("  [SUCCESS] Conditional edges, evaluation quality gates, and cycles verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
