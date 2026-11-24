# Module 03: Model Context Protocol (MCP) Servers and Clients

This module provides production-grade implementations of the **Model Context Protocol (MCP)** specification, standardizing how AI agents access enterprise tools, live database resources, and system prompts across decoupled services.

---

## 🏗️ Architecture Overview

The Model Context Protocol establishes an open, standard JSON-RPC 2.0 communication protocol between AI Host applications (IDE, chat clients, agent runtimes) and external tool/resource providers.

```
┌────────────────────────────────────────────────────────┐
│                   AI Host Application                  │
│       (Claude Desktop, Cursor, Custom Agent CLI)       │
└───────────────────────────┬────────────────────────────┘
                            │
              JSON-RPC 2.0  │ stdio / SSE / HTTP
                            ▼
┌────────────────────────────────────────────────────────┐
│                       MCP Client                       │
│    (02_mcp_client_integration.py / LangChain Agent)    │
│  - Handshake (initialize)                              │
│  - Tool/Resource Discovery                             │
│  - Execution Dispatch                                  │
└───────────────────────────┬────────────────────────────┘
                            │
              JSON-RPC 2.0  │ pipes (stdin / stdout)
                            ▼
┌────────────────────────────────────────────────────────┐
│                       MCP Server                       │
│             (01_mcp_server_custom_tools.py)            │
│  - Tools: get_inventory_status, check_shipping_route   │
│  - Resources: catalog://warehouses                     │
│  - Prompts: supply_chain_triage                        │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Module Contents

| Script | Title | Description |
|---|---|---|
| `01_mcp_server_custom_tools.py` | Custom MCP Server | Production MCP server implementing tools (`get_inventory_status`, `check_shipping_route`, `calculate_tariff_cost`), resource endpoints (`catalog://warehouses`), and prompt templates (`supply_chain_triage`). Supports both local verification and stdio server daemon modes. |
| `02_mcp_client_integration.py` | MCP Client & Discovery | Protocol client launching the server subprocess via `stdio`, executing the JSON-RPC initialization handshake, enumerating tool schemas, reading resource contents, and dispatching tool calls. |
| `03_mcp_langchain_agent.py` | LangChain / LangGraph Agent Integration | Dynamic MCP-to-LangChain adapter wrapping MCP tools into `StructuredTool` objects for autonomous ReAct reasoning. Includes multi-provider support (OpenAI and deterministic offline runner). |

---

## 🚀 Running the Scripts

Activate your virtual environment and run from the repository root:

### 1. Run MCP Server in Self-Verification Mode
```bash
python ai-engineer/03-mcp-servers-and-clients/01_mcp_server_custom_tools.py
```
*Validates tool signatures, registers resources and prompt templates, and runs local execution checks.*

To run in standard Stdio Server Daemon mode for external clients:
```bash
python ai-engineer/03-mcp-servers-and-clients/01_mcp_server_custom_tools.py --serve
```

---

### 2. Run MCP Client Integration Test
```bash
python ai-engineer/03-mcp-servers-and-clients/02_mcp_client_integration.py
```
*Launches the server subprocess over `stdio`, performs protocol capability negotiation, reads warehouse catalogs, and verifies remote tool execution.*

---

### 3. Run MCP Agent with LangChain / LangGraph
```bash
python ai-engineer/03-mcp-servers-and-clients/03_mcp_langchain_agent.py
```
*Adapts registered MCP tools into LangChain tools dynamically and runs a ReAct agent diagnosing supply chain shortages and calculating landed duties.*

---

## 🛠️ MCP Primitives Covered

1. **Tools (`tools/list`, `tools/call`)**:
   Functions that perform computations or mutations and return structured or text content.
2. **Resources (`resources/list`, `resources/read`)**:
   Read-only data URIs (e.g., `catalog://warehouses`) that provide deterministic domain context without consuming LLM reasoning tokens upfront.
3. **Prompts (`prompts/list`, `prompts/get`)**:
   Pre-engineered prompt templates allowing servers to dictate standard operational procedures and triage workflows to client agents.
4. **Transport**:
   Standard `stdio` (standard input/output) inter-process communication supporting process isolation and local sandbox execution.
