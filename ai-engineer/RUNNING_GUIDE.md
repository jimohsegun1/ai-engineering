# AI Engineer Master Course — Complete Execution & Running Guide

This guide provides exact step-by-step instructions, copy-paste terminal commands, expected terminal outputs, and troubleshooting tips for running every script in the `ai-engineer/` codebase.

---

## 1. Prerequisites & Virtual Environment

Every script is executed from the **repository root** with the Python virtual environment activated:

### PowerShell (Windows):
```powershell
# Standing in the project root:
cd C:\Users\jimoh\OneDrive\Desktop\OFFICE\ai-engineering

# Activate virtual environment:
venv\Scripts\Activate.ps1
```

### Git Bash / Linux / macOS:
```bash
cd /c/Users/jimoh/OneDrive/Desktop/OFFICE/ai-engineering
source venv/Scripts/activate
```

When active, your terminal prompt will display `(venv)`.

---

## 2. Module 01: RAG and Advanced GraphRAG

Located in: [`ai-engineer/01-rag-and-graphrag/`](01-rag-and-graphrag/README.md)

### Script 1: Naive Vector RAG vs. GraphRAG (`01_naive_vs_advanced_rag.py`)
Compares flat vector similarity search against multi-hop knowledge graph traversal on a complex supply chain disruption query.

```powershell
python ai-engineer/01-rag-and-graphrag/01_naive_vs_advanced_rag.py
```

**Expected Terminal Output:**
```text
======================================================================
STEP 1: Load, Chunk, and Index Text in Chroma (Vector-Only RAG)
======================================================================
  Indexed 15 chunks from supply_chain.txt into Chroma.

======================================================================
STEP 3: Execute Multi-Hop Traversal from 'SiliconCore Fab'
======================================================================
  Disruption at 'SiliconCore Fab' propagates through 8 distinct supply chains:
    * SiliconCore Fab -> SC-9901 -> Bosch Mobility -> MotorControl-X -> Apex CyberTruck -> SwiftLogistics Corp
    ...

======================================================================
STEP 5: Generate Final Answers with Local LLM
======================================================================
[A] Vector RAG Answer:
    Bosch Mobility

[B] GraphRAG Answer:
    SwiftLogistics Corp
======================================================================
CONCLUSION: GraphRAG overcomes the vector similarity horizon by converting
structural relationships into deterministic, verifiable multi-hop context.
======================================================================
```

---

### Script 2: Entity & Relation Extraction (`02_entity_relation_extraction.py`)
Converts unstructured prose into structured Pydantic entity nodes and typed `(Subject, Predicate, Object)` relationship triples, saving `extracted_graph.json`.

```powershell
python ai-engineer/01-rag-and-graphrag/02_entity_relation_extraction.py
```

**Expected Terminal Output:**
```text
======================================================================
STEP 2: Extract Structured Entities & Typed Relationships
======================================================================
  Extracted 17 Entities:
  Entity Name                  | Category         | Key Attribute
  --------------------------------------------------------------------
  SiliconCore Fab              | Foundry          | Dresden, Germany
  SC-9901                      | Microchip        | Silicon Component
  Bosch Mobility               | Tier1_Supplier   | Stuttgart, Germany
  Apex CyberTruck              | Vehicle          | OEM: Apex Motors
  SwiftLogistics Corporation   | Customer         | PO: ORD-881 (500 units)
  ...

  Extracted 17 Directed Relationships:
  SiliconCore Fab          -> MANUFACTURES     -> SC-9901                 
  SC-9901                  -> SUPPLIED_TO      -> Bosch Mobility          
  Bosch Mobility           -> ASSEMBLES        -> MotorControl-X          
  MotorControl-X           -> INSTALLED_IN     -> Apex CyberTruck         
  Apex CyberTruck          -> DELIVERED_TO     -> SwiftLogistics Corporation

======================================================================
STEP 3: Serialize Knowledge Graph to JSON
======================================================================
  Graph serialized and saved to: extracted_graph.json
```

---

### Script 3: In-Memory GraphRAG Engine (`03_graphrag_in_memory.py`)
Runs an end-to-end GraphRAG pipeline in pure Python using `NetworkX` and `MiniLM` embeddings. Discovers seed nodes, expands $k$-hop subgraphs, and generates answers.

```powershell
python ai-engineer/01-rag-and-graphrag/03_graphrag_in_memory.py
```

**Expected Terminal Output:**
```text
======================================================================
STEP 2: Load Knowledge Graph Data
======================================================================
  Indexed Graph: 17 nodes, 17 edges.

======================================================================
STEP 4: Execute Multi-Hop GraphRAG Queries
======================================================================
Query: 'Which vehicles use chips produced by SiliconCore Fab?'
Seed Nodes Discovered: ['SiliconCore Fab', 'SC-9901']
Traversed Graph Facts (14 edges):
  * Entity 'SiliconCore Fab' (Foundry) -[MANUFACTURES]-> Entity 'SC-9901' (Microchip)
  * Entity 'SC-9901' (Microchip) -[SUPPLIED_TO]-> Entity 'Bosch Mobility' (Tier1_Supplier)
Answer: EuroCross SUV, Apex CyberTruck

Query: 'If TSMC Fab 18 has a fire, which Tier-1 suppliers are affected?'
Seed Nodes Discovered: ['TSMC Fab 18', 'Continental Automotive']
Answer: Continental Automotive
```

---

### Script 4: Hybrid Graph + Vector RAG (`04_hybrid_graph_vector_rag.py`)
Fuses dense vector retrieval (unstructured text paragraphs) with graph traversal (multi-hop lineage) for enterprise-grade grounded answers.

```powershell
python ai-engineer/01-rag-and-graphrag/04_hybrid_graph_vector_rag.py
```

**Expected Terminal Output:**
```text
======================================================================
STEP 3: Execute Hybrid Retrieval & Context Fusion
======================================================================
--- [1] Retrieved Structured Graph Lineage ---
Supply Chain Lineage #1: SiliconCore Fab -> SC-9901 -> Bosch Mobility -> MotorControl-X -> Apex CyberTruck -> SwiftLogistics Corp
  * (SiliconCore Fab) -[MANUFACTURES]-> (SC-9901)
  * (SC-9901) -[SUPPLIED_TO]-> (Bosch Mobility)
  * (Bosch Mobility) -[ASSEMBLES]-> (MotorControl-X)
  * (MotorControl-X) -[INSTALLED_IN]-> (Apex CyberTruck)
  * (Apex CyberTruck) -[DELIVERED_TO]-> (SwiftLogistics Corp)

--- [2] Retrieved Unstructured Text Chunks ---
1. Semiconductor Foundries (Raw Silicon Production)...

======================================================================
STEP 4: Generate Answer with Fused Hybrid Context
======================================================================
Hybrid RAG Answer:
The disruption at SiliconCore Fab halts SC-9901 microcontrollers, causing Bosch Mobility to pause MotorControl-X assembly, which delays fulfillment of SwiftLogistics Corp's purchase order for Apex CyberTruck.
======================================================================
SUCCESS: Hybrid Graph + Vector RAG verified.
======================================================================
```

---

## 3. Module 02: Neo4j Knowledge Graphs and Cypher

Located in: [`ai-engineer/02-neo4j-knowledge-graphs/`](02-neo4j-knowledge-graphs/README.md)

### Script 1: Neo4j Connection & Schema Constraints (`01_neo4j_connection_and_setup.py`)
Initializes the driver, applies schema uniqueness constraints, and performs health checks.
*(Automatically falls back to local simulation mode if Docker/AuraDB is offline).*

```powershell
python ai-engineer/02-neo4j-knowledge-graphs/01_neo4j_connection_and_setup.py
```

**Expected Terminal Output:**
```text
======================================================================
STEP 1: Configure Neo4j Database Connection
======================================================================
  Target URI:      bolt://localhost:7687
  Target Username: neo4j
  Connection Status: CONNECTED (OFFLINE SIMULATED MODE)

======================================================================
STEP 2: Create Enterprise Schema Uniqueness Constraints & Indexes
======================================================================
  [OK] Applied Schema Constraint: 'constraint_foundry_name'
  [OK] Applied Schema Constraint: 'constraint_chip_name'
  [OK] Applied Schema Constraint: 'constraint_supplier_name'
  [OK] Applied Schema Constraint: 'constraint_vehicle_name'
  [OK] Applied Schema Constraint: 'constraint_customer_name'

======================================================================
STEP 3: Verify Active Database Constraints & Health
======================================================================
  Active Schema Constraints (5 verified):
    - constraint_foundry_name
    - constraint_chip_name
    - constraint_supplier_name
    - constraint_vehicle_name
    - constraint_customer_name
======================================================================
SUCCESS: Neo4j schema & connection management verified.
======================================================================
```

---

### Script 2: Idiomatic Cypher Query Patterns (`02_cypher_query_patterns.py`)
Demonstrates core Cypher patterns: `MERGE`, multi-hop paths (`[*1..5]`), degree centrality aggregations, and shortest-path failure analysis.

```powershell
python ai-engineer/02-neo4j-knowledge-graphs/02_cypher_query_patterns.py
```

**Expected Terminal Output:**
```text
======================================================================
STEP 1: Idempotent Node & Edge Creation (MERGE)
======================================================================
[Cypher Pattern]: MERGE Nodes with On-Create Attributes
MERGE (f:Foundry {name: 'SiliconCore Fab'})
...

======================================================================
STEP 2: Multi-Hop Variable-Length Path Traversal
======================================================================
[Cypher Pattern]: Variable-Length Lineage Search (*1..5)
MATCH path = (start:Foundry {name: 'SiliconCore Fab'})-[*1..5]->(target:Customer)
Results:
  {'path': 'SiliconCore Fab -> SC-9901 -> Bosch Mobility -> MotorControl-X -> Apex CyberTruck -> SwiftLogistics Corp', 'hop_count': 5}

======================================================================
STEP 3: Centrality & Network Bottleneck Analysis
======================================================================
[Cypher Pattern]: Degree Centrality & Supply Bottlenecks
Results:
  {'entity': 'Bosch Mobility', 'label': 'Supplier', 'degree': 4}
  {'entity': 'SiliconCore Fab', 'label': 'Foundry', 'degree': 3}
======================================================================
SUCCESS: Key Cypher patterns for GraphRAG demonstrated.
======================================================================
```

---

### Script 3: Bulk Graph Schema Construction (`03_graph_schema_construction.py`)
Demonstrates bulk ingestion of structured entities and relationships into Neo4j using high-throughput parameterized `UNWIND $batch` Cypher statements.

```powershell
python ai-engineer/02-neo4j-knowledge-graphs/03_graph_schema_construction.py
```

**Expected Terminal Output:**
```text
======================================================================
STEP 1: Load Structured Graph Dataset
======================================================================
  Loaded 17 entities and 17 relations ready for Neo4j ingestion.

======================================================================
STEP 2: Generate Parameterized UNWIND Batch Cypher Statements
======================================================================
  [Batch Node Ingestion Query]:
  UNWIND $batch AS row ...

======================================================================
STEP 3: Audit Schema & Verify Node/Edge Counts
======================================================================
  Total Indexed Nodes:         17
  Total Indexed Relationships: 17

  Node Distribution by Label:
    * (:Foundry         ) -> 2 nodes
    * (:Microchip       ) -> 2 nodes
    * (:Tier1_Supplier  ) -> 3 nodes
    * (:Vehicle         ) -> 4 nodes
    * (:Subsystem       ) -> 3 nodes
    * (:Customer        ) -> 3 nodes
======================================================================
SUCCESS: Automated Neo4j graph construction pipeline validated.
======================================================================
```

---

### Script 4: Text-to-Cypher AI Agent (`04_text_to_cypher_agent.py`)
Translates natural language questions into safe, executable Cypher queries with schema inspection, read-only guards, and answer synthesis.

```powershell
python ai-engineer/02-neo4j-knowledge-graphs/04_text_to_cypher_agent.py
```

**Expected Terminal Output:**
```text
======================================================================
STEP 1: Inspect Graph Schema Context
======================================================================
Node Labels: (:Foundry), (:Microchip), (:Supplier), (:Vehicle), (:Customer)
Relationships: [:MANUFACTURES], [:SUPPLIED_TO], [:ASSEMBLES], [:INSTALLED_IN], [:DELIVERED_TO]

======================================================================
STEP 2: Translate Natural Language to Cypher Query
======================================================================
  User Question: 'Which vehicles depend on microchips manufactured in Dresden?'
  [Generated Cypher Query]:
  MATCH (f:Foundry {location: 'Dresden'})-[:MANUFACTURES]->(c:Microchip)-[:SUPPLIED_TO]->(s:Supplier)-[:ASSEMBLES]->(sub:Subsystem)-[:INSTALLED_IN]->(v:Vehicle) RETURN DISTINCT v.name AS vehicle, f.name AS foundry, c.name AS chip;

======================================================================
STEP 3: Security Validation & Read-Only Safety Check
======================================================================
  [PASSED] Query contains no destructive operations (DROP, DELETE, DETACH).

======================================================================
STEP 4: Execute Cypher against Knowledge Graph
======================================================================
  Retrieved 3 matching records from graph:
    * Vehicle: Apex CyberTruck    | Foundry: SiliconCore Fab  | Chip: SC-9901
    * Vehicle: Apex CyberSedan    | Foundry: SiliconCore Fab  | Chip: PC-404
    * Vehicle: EuroCross SUV      | Foundry: SiliconCore Fab  | Chip: SC-9901

======================================================================
STEP 5: Synthesize Natural Language Executive Summary
======================================================================
  Final Agent Response:
  The vehicles depending on microchips manufactured in Dresden (SiliconCore Fab) are Apex CyberTruck, Apex CyberSedan, and EuroCross SUV.
======================================================================
SUCCESS: Text-to-Cypher agent with safety guards and synthesis verified.
======================================================================
```

---

## 4. Module 03: Model Context Protocol (MCP) Servers and Clients

Located in: [`ai-engineer/03-mcp-servers-and-clients/`](03-mcp-servers-and-clients/README.md)

### Script 1: Custom MCP Server (`01_mcp_server_custom_tools.py`)
Implements an MCP server publishing specialized domain tools (`get_inventory_status`, `check_shipping_route`, `calculate_tariff_cost`), warehouse resource endpoints (`catalog://warehouses`), and triage prompts.

```powershell
python ai-engineer/03-mcp-servers-and-clients/01_mcp_server_custom_tools.py
```

**Expected Terminal Output:**
```text
======================================================================
  MODEL CONTEXT PROTOCOL (MCP) - CUSTOM SERVER DEMONSTRATION
  Framework: supply-chain-mcp-server via MCPServer (v2.x)
======================================================================

[Step 1] Inspecting Registered MCP Tools...
  1. get_inventory_status: Query current inventory stock level, warehouse location, reorder thresholds, and unit costs for a given part ID.
  2. check_shipping_route: Evaluate transit duration, carrier availability, carbon footprint, and customs risk score between origin and destination ports.
  3. calculate_tariff_cost: Calculate import tariffs, customs processing duties, and total landed cost based on declared value, product category, and destination country.

[Step 2] Inspecting Registered MCP Resources...
  1. catalog://warehouses: Global Warehouses Status and Capacity (application/json)

[Step 3] Inspecting Registered MCP Prompts...
  1. supply_chain_triage: Standard incident response prompt for diagnosing supply chain bottlenecks and recommending rerouting.

[Step 4] Executing Direct Tool Invocation Tests:
  -> Invoking get_inventory_status(part_id='GPU-H100')...
     Output: { "status": "success", "stock_level": 420, "status": "critical_low", ... }
  -> Invoking check_shipping_route(origin_port='SHANGHAI', destination_port='LONG_BEACH')...
     Output: { "mode": "Ocean Freight (Express)", "transit_days": 14, ... }
  -> Invoking calculate_tariff_cost(part_id='GPU-H100', declared_value=50000.0, destination_country='USA')...
     Output: { "total_tariffs_and_fees_usd": 13923.2, "total_landed_cost_usd": 63923.2, ... }

======================================================================
  [SUCCESS] All MCP tools, resources, and prompts validated successfully!
======================================================================
```

To run as an active stdio server daemon for external clients:
```powershell
python ai-engineer/03-mcp-servers-and-clients/01_mcp_server_custom_tools.py --serve
```

---

### Script 2: MCP Client Integration & Discovery (`02_mcp_client_integration.py`)
Launches the MCP server subprocess via `stdio`, executes the JSON-RPC initialization handshake, queries available tools and resources, and dispatches remote tool calls.

```powershell
python ai-engineer/03-mcp-servers-and-clients/02_mcp_client_integration.py
```

**Expected Terminal Output:**
```text
======================================================================
  MODEL CONTEXT PROTOCOL (MCP) - CLIENT INTEGRATION & DISCOVERY
======================================================================

[Phase 1] Launching Server Subprocess via Stdio Transport...
[Phase 2] Performing JSON-RPC Handshake (initialize)...
  Connected to: supply-chain-mcp-server (v1.0.0)
[Phase 3] Tool Discovery (tools/list)...
  Discovered 3 available tool(s):
  • Tool: get_inventory_status
  • Tool: check_shipping_route
  • Tool: calculate_tariff_cost
[Phase 4] Resource Discovery (resources/list & resources/read)...
  Retrieved 4 global warehouse hubs from catalog://warehouses
[Phase 5] Prompt Template Discovery (prompts/list)...
  Retrieved 'supply_chain_triage'
[Phase 6] Dispatching Tool Calls (tools/call)...
  [Call 1] get_inventory_status(part_id='BAT-LFP-85') -> Stock: 1,850 units | OPTIMAL
  [Call 2] check_shipping_route(SHANGHAI -> LONG_BEACH) -> Ocean Freight | 14 days
  [Call 3] calculate_tariff_cost(BAT-LFP-85, $74,000) -> Landed Cost: $83,506.34

======================================================================
  [SUCCESS] MCP Client stdio integration session completed cleanly!
======================================================================
```

---

### Script 3: LangChain / LangGraph Agent Integration (`03_mcp_langchain_agent.py`)
Dynamically adapts MCP tools into LangChain `StructuredTool` objects and runs an autonomous ReAct reasoning agent that triages a supply chain shortage.

```powershell
python ai-engineer/03-mcp-servers-and-clients/03_mcp_langchain_agent.py
```

**Expected Terminal Output:**
```text
======================================================================
  LANGCHAIN / LANGGRAPH AGENT INTEGRATION WITH MCP TOOLS
======================================================================

[Step 1] Querying MCP Server Tool Registry...
  Found 3 tool(s) registered on MCP server.
[Step 2] Adapting MCP Tools into LangChain StructuredTool Objects...
  • Adapted: get_inventory_status
  • Adapted: check_shipping_route
  • Adapted: calculate_tariff_cost

[Thought 1] Query requests inventory health for part 'GPU-H100'.
[Action 1] Invoking MCP tool: get_inventory_status(part_id='GPU-H100')
[Observation 1] Current Stock: 420 units. Status: critical_low.

[Thought 2] Assessing alternative freight corridors between FRANKFURT and LONG_BEACH.
[Action 2] Invoking MCP tool: check_shipping_route(origin_port='FRANKFURT', destination_port='LONG_BEACH')
[Observation 2] Route transit: 3 days via Air Cargo Freight.

[Thought 3] Calculating customs tariffs for declared value $96,000.00 to USA.
[Action 3] Invoking MCP tool: calculate_tariff_cost(part_id='GPU-H100', declared_value=96000.0, destination_country='USA')
[Observation 3] Landed Cost: $122,732.54 (Tariffs: $26,732.54).

======================================================================
  EXECUTIVE SUPPLY CHAIN TRIAGE & DISRUPTION MITIGATION REPORT
======================================================================
1. Inventory Health: GPU-H100 | 420 units | CRITICAL_LOW
2. Route & Transit: FRANKFURT -> LONG_BEACH | 3 days via Air Cargo
3. Landed Duty: Declared $96,000 | Tariffs $26,732.54 | Total Landed: $122,732.54
4. Recommendation: Immediate dispatch via Lufthansa Cargo recommended.
======================================================================
```

---

## 5. Module 04: LangGraph Workflows and Multi-Agent Orchestration

Located in: [`ai-engineer/04-langgraph-workflows/`](04-langgraph-workflows/README.md)

### Script 1: Stateful Graph Foundations (`01_stateful_graph_basics.py`)
Demonstrates core LangGraph concepts: `StateGraph`, `TypedDict` schemas, append reducers (`Annotated[List[str], operator.add]`), and deterministic multi-node execution.

```powershell
python ai-engineer/04-langgraph-workflows/01_stateful_graph_basics.py
```

**Expected Terminal Output:**
```text
======================================================================
  LANGGRAPH FOUNDATIONS: STATEFUL GRAPH & STATE REDUCERS
======================================================================
[Node: validate_order] Processing Order: ORD-2026-9941
  -> Order validated successfully. 2 SKU(s) verified. Total Value: $128,242.50.
[Node: allocate_warehouse] Determining optimal fulfillment hub...
  -> Allocated fulfillment to hub: LONG-BEACH-DC.
[Node: select_freight_and_sla] Selecting freight carrier SLA...
  -> Carrier assigned: FedEx International Priority Air (2 days).
[Node: generate_manifest] Assembling dispatch manifest...
  -> Manifest generated: MNF-ORD-2026-9941-EXP.

Accumulated Execution Log (State Reducer: operator.add):
  1. Order received into queue.
  2. [ORD-2026-9941] Order validated successfully. Total Value: $128,242.50.
  3. [ORD-2026-9941] Allocated fulfillment to hub: LONG-BEACH-DC.
  4. [ORD-2026-9941] Carrier assigned: FedEx International Priority Air (SLA: 2 days).
  5. [ORD-2026-9941] Manifest generated: MNF-ORD-2026-9941-EXP.
======================================================================
```

---

### Script 2: Conditional Routing and Reflection Cycles (`02_conditional_routing_and_cycles.py`)
Demonstrates dynamic branching (`add_conditional_edges`), quality gates, critic evaluation, and self-correcting plan refinement cycles.

```powershell
python ai-engineer/04-langgraph-workflows/02_conditional_routing_and_cycles.py
```

**Expected Terminal Output:**
```text
======================================================================
  LANGGRAPH ADVANCED: CONDITIONAL ROUTING & REFLECTION CYCLES
======================================================================
[Node: analyze_disruption] Incident: INC-ROTTERDAM-STRIKE-88 (Part: GPU-H100)
[Node: propose_mitigation] Generating Plan (Iteration 1/3)...
  -> Strategy: Direct Emergency Air Charter | Transit: 2 days | Premium: $72,000
[Node: evaluate_risk_and_cost] Evaluating Plan v1...
  -> Risk Score: 65.0/100 (Threshold <= 20.0) -> REJECT: Unbudgeted air charter cost.

>>> [Routing Decision] Risk Score 65.0 > 20.0 -> CYCLING BACK to 'propose_mitigation'

[Node: propose_mitigation] Generating Plan (Iteration 2/3)...
  -> Strategy: Scheduled Commercial Air Cargo + Bonded Transit | Transit: 4 days | Premium: $28,000
[Node: evaluate_risk_and_cost] Evaluating Plan v2...
  -> Risk Score: 15.0/100 (Threshold <= 20.0) -> APPROVED: Within budget ceiling.

>>> [Routing Decision] Risk Score <= 20.0 -> Routing to 'approve_and_dispatch'

[Node: approve_and_dispatch] Locking in approved mitigation plan...
======================================================================
  WORKFLOW TERMINATION SUMMARY: APPROVED_FOR_DISPATCH in 2 cycle(s)
======================================================================
```

---

### Script 3: Checkpointing, Persistence & Time Travel (`03_checkpointing_and_persistence.py`)
Demonstrates durable state persistence with `SqliteSaver` (`data/checkpoints/orders.db`), multi-thread isolation, human-in-the-loop pauses (`interrupt_before`), manager approval updates, and historical state time travel.

```powershell
python ai-engineer/04-langgraph-workflows/03_checkpointing_and_persistence.py
```

**Expected Terminal Output:**
```text
======================================================================
  LANGGRAPH CHECKPOINTING, PERSISTENCE & TIME TRAVEL
  Checkpointer Database: data\checkpoints\orders.db
======================================================================
[Phase 1] Launching Two Independent Order Threads (alpha & beta)...
  Order Alpha ($64,000) pauses before execute_fulfillment (High-Value Gate)
  Order Beta ($485) pauses before execute_fulfillment

[Phase 2] Inspecting Paused State via get_state()...
  Thread Alpha Next Node: ('execute_fulfillment',) | Needs Approval: True

[Phase 3] Human Intervention: Manager Reviews & Signs Off Order Alpha...
  update_state() called with manager sign-off.
  Resuming execution from checkpoint via invoke(None, config)...
  Order Alpha Dispatched: True (Tracking: TRK-ORD-ALPHA-FEDEX)

[Phase 4] Time Travel Inspection: Historical Snapshots in SQLite Checkpointer...
  Retrieved 6 checkpoint snapshot(s) for Thread Alpha:
  Checkpoint 1: (Completed) | Manager Approval: True | Dispatched: True
  Checkpoint 2: ('execute_fulfillment',) | Manager Approval: True
  Checkpoint 3: ('execute_fulfillment',) | Manager Approval: False (Paused state)
  ...

[Phase 5] Verifying Cross-Session Disk Persistence...
  Reloaded Order Alpha from fresh SQLite connection successfully!
======================================================================
  [SUCCESS] SQLite checkpointing, human-in-the-loop gates, and time travel verified!
======================================================================
```

---

## 6. Optional: Spin Up Local Neo4j Container

If you have Docker Desktop installed and want to view the knowledge graph visually in the Neo4j Browser:

```powershell
# Start Neo4j in the background:
docker compose -f ai-engineer/docker-compose.yml up -d

# Open the Neo4j Browser UI in your web browser:
# URL:      http://localhost:7474
# Username: neo4j
# Password: aiengineering_password

# Stop the container when finished:
docker compose -f ai-engineer/docker-compose.yml down
```
*(Every script automatically falls back to local simulation if Docker is not started, so Docker is 100% optional).*
