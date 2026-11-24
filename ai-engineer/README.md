# AI Engineer Master Course — Complete Resource Pack

Companion codebase for the [AI Engineer Master Course](https://youtu.be/BXY7mAFN21k).

This repository pack provides practical, production-grade implementations of advanced AI engineering architectures: GraphRAG, Neo4j Knowledge Graphs, Model Context Protocol (MCP), LangGraph, ReAct / Reflection / Human-in-the-Loop, Deep Multi-Agent Systems, Langfuse LLMOps, and Databricks Lakehouse Integrations.

> 📖 **Step-by-step execution guide:** See [**RUNNING_GUIDE.md**](RUNNING_GUIDE.md) for copy-paste terminal commands and expected terminal outputs for every script.

---

## Curriculum Overview

| Module | Name | Key Technologies | Status |
| --- | --- | --- | --- |
| [`01-rag-and-graphrag/`](01-rag-and-graphrag/README.md) | Graph-Augmented Retrieval | NetworkX, HuggingFace MiniLM, Chroma, Hybrid RAG | ✅ Implemented (Phase 1) |
| [`02-neo4j-knowledge-graphs/`](02-neo4j-knowledge-graphs/README.md) | Neo4j Knowledge Graphs & Cypher | Neo4j, Cypher, Text-to-Cypher Agent, Schema Audits | ✅ Implemented (Phase 1) |
| [`03-mcp-servers-and-clients/`](03-mcp-servers-and-clients/README.md) | Model Context Protocol (MCP) | Python MCP SDK, stdio/SSE, LangChain Agent Binding | ✅ Implemented (Phase 2) |
| [`04-langgraph-workflows/`](04-langgraph-workflows/README.md) | Stateful Agent Workflows | LangGraph, StateGraph, Cyclic Routing, SqliteSaver | ✅ Implemented (Phase 2) |
| `05-agent-patterns/` | Cognitive Decision Patterns | ReAct, Semantic Routing, Reflection, Human-in-the-Loop | 📋 Planned (Phase 3) |
| `06-deep-agents-and-guardrails/` | Deep Agents & Production Safety | Supervisor Multi-Agent, Researcher, Pydantic Guards | 📋 Planned (Phase 3) |
| `07-observability-langfuse/` | LLMOps & Distributed Tracing | Langfuse, Nested Traces, Token Accounting, Evals | 📋 Planned (Phase 4) |
| `08-databricks-integrations/` | Lakehouse & Enterprise Analytics | Databricks SQL, Unity Catalog, Genie Integration | 📋 Planned (Phase 4) |

---

## One-Time Environment Setup

From the repository root, activate the shared virtual environment:

```powershell
venv\Scripts\Activate.ps1
```

Copy the environment template:
```powershell
Copy-Item ai-engineer/.env.example ai-engineer/.env
```

*(Optional)* Start the local Neo4j Docker container:
```powershell
docker compose -f ai-engineer/docker-compose.yml up -d
```
Access the Neo4j Browser at: [http://localhost:7474](http://localhost:7474) (User: `neo4j`, Password: `aiengineering_password`).
*(Every script includes automatic offline simulation fallback if Docker is not running).*
