"""Module 01: Naive Vector RAG vs. GraphRAG.

Demonstrates why standard vector similarity fails on multi-hop relational questions
and how GraphRAG bridges the gap by traversing explicit entity-relationship paths.

Multi-Hop Question:
  "If SiliconCore Fab in Dresden experiences a complete factory shutdown,
   which commercial fleet customers and vehicle purchase orders will be delayed?"

Comparison:
  - Vector-Only RAG: Embeds text chunks into Chroma. Retrieves the top-k chunks
    closest to the query in embedding space. In a multi-hop scenario, the vector search
    retrieves the Dresden facility description, but fails to traverse the 4 downstream
    links required to identify the affected commercial purchase orders (ORD-881, ORD-442, ORD-105).
  - GraphRAG: Traverses explicit entity relationships from the root disruption node:
    (SiliconCore Fab) -> [MANUFACTURES] -> (SC-9901, PC-404)
                      -> [SUPPLIED_TO]   -> (Bosch Mobility, Apex Power)
                      -> [INSTALLED_IN]  -> (CyberTruck, CyberSedan, EuroCross)
                      -> [DELIVERED_TO]  -> (SwiftLogistics, Nordic Fleet, MetroCity)
"""

import os
import shutil
from pathlib import Path

# Disable Chroma telemetry before import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import networkx as nx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
DOCUMENT_PATH = PROJECT_ROOT / "data" / "supply_chain.txt"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "vector_rag_supply_chain"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-base"

QUERY = "If SiliconCore Fab in Dresden experiences a complete factory shutdown, which commercial fleet customers and purchase orders are delayed?"

RULE = "=" * 70
SUB_RULE = "-" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# ---------------------------------------------------------------------------
# Step 1: Standard Vector RAG Setup
# ---------------------------------------------------------------------------
print_step(1, "Load, Chunk, and Index Text in Chroma (Vector-Only RAG)")

loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = splitter.split_documents(raw_docs)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))

print(f"  Indexed {len(chunks)} chunks from {DOCUMENT_PATH.name} into Chroma.")

# Execute Vector Retrieval (Top-k=2)
vector_docs = vector_store.similarity_search(QUERY, k=2)
print("\n  Top-2 Chunks Retrieved by Vector Search:")
for i, d in enumerate(vector_docs, start=1):
    preview = d.page_content.replace("\n", " ")[:140]
    print(f"    [{i}] {preview}...")


# ---------------------------------------------------------------------------
# Step 2: Build Supply Chain Knowledge Graph (GraphRAG)
# ---------------------------------------------------------------------------
print_step(2, "Construct Knowledge Graph for Multi-Hop Traversal (GraphRAG)")

G = nx.DiGraph()

# Add Nodes with Type Attributes
nodes = [
    ("SiliconCore Fab", {"type": "Foundry", "location": "Dresden"}),
    ("TSMC Fab 18", {"type": "Foundry", "location": "Tainan"}),
    ("SC-9901", {"type": "Microchip", "category": "Powertrain MCU"}),
    ("PC-404", {"type": "Microchip", "category": "Thermal Power Controller"}),
    ("SP-12", {"type": "Microchip", "category": "Radar Processor"}),
    ("Bosch Mobility", {"type": "Tier1_Supplier", "location": "Stuttgart"}),
    ("Apex Power Systems", {"type": "Tier1_Supplier", "location": "Austin"}),
    ("Continental Automotive", {"type": "Tier1_Supplier", "location": "Hanover"}),
    ("MotorControl-X", {"type": "Subsystem", "category": "Inverter & Drive"}),
    ("BMU-300", {"type": "Subsystem", "category": "Battery Management"}),
    ("VisionDrive", {"type": "Subsystem", "category": "Sensor Module"}),
    ("Apex CyberTruck", {"type": "Vehicle", "oem": "Apex Motors"}),
    ("Apex CyberSedan", {"type": "Vehicle", "oem": "Apex Motors"}),
    ("EuroCross SUV", {"type": "Vehicle", "oem": "EuroDrive"}),
    ("SwiftLogistics Corp", {"type": "Customer", "order": "ORD-881", "qty": 500}),
    ("Nordic Fleet Rental", {"type": "Customer", "order": "ORD-442", "qty": 300}),
    ("MetroCity Transit", {"type": "Customer", "order": "ORD-105", "qty": 150}),
]
G.add_nodes_from(nodes)

# Add Multi-Hop Edges
edges = [
    ("SiliconCore Fab", "SC-9901", {"rel": "MANUFACTURES"}),
    ("SiliconCore Fab", "PC-404", {"rel": "MANUFACTURES"}),
    ("TSMC Fab 18", "SP-12", {"rel": "MANUFACTURES"}),
    ("SC-9901", "Bosch Mobility", {"rel": "SUPPLIED_TO"}),
    ("PC-404", "Bosch Mobility", {"rel": "SUPPLIED_TO"}),
    ("PC-404", "Apex Power Systems", {"rel": "SUPPLIED_TO"}),
    ("SP-12", "Continental Automotive", {"rel": "SUPPLIED_TO"}),
    ("Bosch Mobility", "MotorControl-X", {"rel": "ASSEMBLES"}),
    ("Apex Power Systems", "BMU-300", {"rel": "ASSEMBLES"}),
    ("Continental Automotive", "VisionDrive", {"rel": "ASSEMBLES"}),
    ("MotorControl-X", "Apex CyberTruck", {"rel": "INSTALLED_IN"}),
    ("BMU-300", "Apex CyberTruck", {"rel": "INSTALLED_IN"}),
    ("MotorControl-X", "Apex CyberSedan", {"rel": "INSTALLED_IN"}),
    ("BMU-300", "Apex CyberSedan", {"rel": "INSTALLED_IN"}),
    ("MotorControl-X", "EuroCross SUV", {"rel": "INSTALLED_IN"}),
    ("VisionDrive", "EuroCross SUV", {"rel": "INSTALLED_IN"}),
    ("Apex CyberTruck", "SwiftLogistics Corp", {"rel": "DELIVERED_TO"}),
    ("EuroCross SUV", "Nordic Fleet Rental", {"rel": "DELIVERED_TO"}),
    ("Apex CyberSedan", "MetroCity Transit", {"rel": "DELIVERED_TO"}),
]
G.add_edges_from([(u, v, attr) for u, v, attr in edges])

print(f"  Graph initialized with {G.number_of_nodes()} nodes and {G.number_of_edges()} directed relationships.")


# ---------------------------------------------------------------------------
# Step 3: Graph Traversal for Downstream Disruption Paths
# ---------------------------------------------------------------------------
print_step(3, "Execute Multi-Hop Traversal from 'SiliconCore Fab'")

target_root = "SiliconCore Fab"
paths_to_customers = []

for node, data in G.nodes(data=True):
    if data.get("type") == "Customer":
        if nx.has_path(G, target_root, node):
            for path in nx.all_simple_paths(G, source=target_root, target=node):
                paths_to_customers.append(path)

print(f"  Disruption at '{target_root}' propagates through {len(paths_to_customers)} distinct supply chains:")
graph_context_lines = []
for p in paths_to_customers:
    formatted_path = " -> ".join(p)
    print(f"    * {formatted_path}")
    cust_node = p[-1]
    cust_meta = G.nodes[cust_node]
    line = (
        f"Path: {' -> '.join(p)}. Result: Customer '{cust_node}' order "
        f"'{cust_meta.get('order')}' ({cust_meta.get('qty')} units of {p[-2]}) delayed."
    )
    graph_context_lines.append(line)

graph_context = "\n".join(graph_context_lines)


# ---------------------------------------------------------------------------
# Step 4: Compare Vector RAG vs. GraphRAG Context
# ---------------------------------------------------------------------------
print_step(4, "Side-by-Side Context Comparison")

print("\n--- [A] VECTOR RAG CONTEXT FED TO MODEL ---")
vector_context = "\n\n".join(d.page_content for d in vector_docs)
print(vector_context)
print("\n>>> Analysis: Vector search matched keywords 'SiliconCore Fab' and 'Dresden',")
print("    but missed the downstream purchase orders located in distant paragraphs.")

print("\n--- [B] GRAPHRAG CONTEXT FED TO MODEL ---")
print(graph_context)
print("\n>>> Analysis: Graph traversal walked the 4-hop chain from Foundry -> Chips -> Subsystems -> Vehicles -> Customers.")


# ---------------------------------------------------------------------------
# Step 5: Answer Generation Comparison
# ---------------------------------------------------------------------------
print_step(5, "Generate Final Answers with Local LLM")

llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 120},
)

prompt_template = PromptTemplate.from_template(
    """Answer the question based only on the provided context:

Context:
{context}

Question: {question}

Answer:"""
)

chain = prompt_template | llm | StrOutputParser()

print(f"\nQuery: {QUERY!r}")

print("\n[A] Vector RAG Answer:")
ans_vector = chain.invoke({"context": vector_context, "question": QUERY})
print(f"    {ans_vector.strip()}")

print("\n[B] GraphRAG Answer:")
ans_graph = chain.invoke({"context": graph_context, "question": QUERY})
print(f"    {ans_graph.strip()}")

print("\n" + RULE)
print("CONCLUSION: GraphRAG overcomes the vector similarity horizon by converting")
print("structural relationships into deterministic, verifiable multi-hop context.")
print(RULE)
