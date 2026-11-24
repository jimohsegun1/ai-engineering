# Module 04: LangGraph Workflows and Multi-Agent Orchestration

This module provides comprehensive, production-grade implementations of **LangGraph**, the stateful agent orchestration framework from the LangChain ecosystem.

---

## 🏗️ Architecture & Core Concepts

Unlike static linear chains, LangGraph models agentic workflows as cyclic, stateful graphs where state flows across nodes and edges:

```
                  ┌──────────────────────┐
                  │        START         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Disruption Analysis │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Propose Mitigation  │◄─────────────────┐
                  │    (Plan Version)    │                  │
                  └──────────┬───────────┘                  │
                             │                              │ Iterative
                             ▼                              │ Self-Correction
                  ┌──────────────────────┐                  │ Feedback Loop
                  │ Risk & Cost Critic   │                  │ (Reflection)
                  │ (Scoring / Quality)  │                  │
                  └──────────┬───────────┘                  │
                             │                              │
                 [Conditional Routing Gate]                 │
                /            │             \                │
    Score <= 20/             │Score > 20    \ Max Cycles    │
              /              │& Iter < Max   \ Exceeded     │
             ▼               ▼                ▼             │
     ┌──────────────┐ ┌───────────────┐ ┌──────────────┐   │
     │   Approve &  │ │ Feedback Loop │ │  Escalate to │   │
     │   Dispatch   │ └───────┬───────┘ │  Human Ops   │   │
     └───────┬──────┘         └─────────┴───────┼──────┴───┘
             │                                  │
             ▼                                  ▼
      ┌─────────────┐                    ┌─────────────┐
      │     END     │                    │     END     │
      └─────────────┘                    └─────────────┘
```

---

## 📂 Module Contents

| Script | Title | Description |
|---|---|---|
| `01_stateful_graph_basics.py` | Stateful Graphs & Reducers | Foundational LangGraph workflow demonstrating `StateGraph`, typed schemas (`TypedDict`), state reducers (`Annotated[List[str], operator.add]`), and multi-step order validation and allocation. |
| `02_conditional_routing_and_cycles.py` | Conditional Routing & Reflection Loops | Dynamic graph branching via `add_conditional_edges`, automated quality gates, critic evaluation nodes, and self-correcting plan refinement cycles. |
| `03_checkpointing_and_persistence.py` | Persistence, Human-in-the-Loop & Time Travel | Production checkpointer backed by `SqliteSaver` (`data/checkpoints/orders.db`), multi-tenant thread isolation, human breakpoint gates (`interrupt_before`), state mutation (`update_state`), and historical snapshot time travel (`get_state_history`). |

---

## 🚀 Running the Scripts

Activate your virtual environment and execute from the repository root:

### 1. Run Stateful Graph Foundations
```bash
python ai-engineer/04-langgraph-workflows/01_stateful_graph_basics.py
```
*Validates order payloads, routes through warehouse allocation and freight selection, and accumulates an audit trail via state reducers.*

---

### 2. Run Conditional Routing and Feedback Cycles
```bash
python ai-engineer/04-langgraph-workflows/02_conditional_routing_and_cycles.py
```
*Demonstrates a self-correcting supply chain disruption mitigator where plan revisions are evaluated against risk scores and iterated until acceptable.*

---

### 3. Run Checkpointing, Breakpoints and Time Travel
```bash
python ai-engineer/04-langgraph-workflows/03_checkpointing_and_persistence.py
```
*Runs two independent order threads against a persistent SQLite checkpointer, pauses at a high-value authorization gate, accepts manager approval, resumes execution, and displays chronological time-travel history.*

---

## 🧠 LangGraph Primitives Reference

1. **StateGraph**: The root graph compiler defining the shared schema contract across all nodes.
2. **State Reducers (`operator.add`)**: Functions dictating how state keys update when nodes return values (e.g. appending log messages or aggregating tool outputs rather than overwriting).
3. **Nodes**: Discrete execution units (functions or runnables) that receive the current state and return incremental state patches.
4. **Edges**:
   - Fixed Edges: `workflow.add_edge("nodeA", "nodeB")`
   - Conditional Edges: `workflow.add_conditional_edges("critic", router_fn, path_mapping)`
5. **Checkpointers (`SqliteSaver`)**:
   - Persists state checkpoints on every transition to durable storage.
   - Enables thread isolation, long-running workflows, and failure recovery.
6. **Breakpoints (`interrupt_before` / `interrupt_after`)**:
   - Halts graph execution cleanly for human review, verification, or external webhook notifications.
7. **Time Travel (`get_state_history`)**:
   - Allows inspection, replay, or branching from any past checkpoint in the execution history.
