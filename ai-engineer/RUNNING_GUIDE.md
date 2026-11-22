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

## 4. Optional: Spin Up Local Neo4j Container

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
