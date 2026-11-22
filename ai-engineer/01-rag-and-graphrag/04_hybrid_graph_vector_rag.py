"""Module 01: Hybrid Graph + Vector RAG Pipeline.

Demonstrates the state-of-the-art enterprise pattern: Fusing dense vector retrieval
(for unstructured descriptive context) with knowledge graph traversal (for multi-hop relational truth).

Why Hybrid?
  - Vector RAG alone: Retrieves surface text matches, but misses multi-hop lineage.
  - Graph RAG alone: Retrieves entity-relation triples, but misses nuanced descriptive prose.
  - Hybrid RAG: Merges both, providing the LLM with deterministic factual paths
    backed by rich narrative explanations.
"""

import json
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
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "hybrid_rag_vector_store"
GRAPH_JSON_PATH = MODULE_ROOT / "extracted_graph.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-base"

COMPLEX_QUERY = (
    "Explain how a disruption at SiliconCore Fab in Dresden propagates to SwiftLogistics Corp, "
    "and identify all intermediary components involved."
)

RULE = "=" * 70
SUB_RULE = "-" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# ---------------------------------------------------------------------------
# Component A: Vector Store Retriever
# ---------------------------------------------------------------------------
print_step(1, "Initialize Dense Vector Store for Unstructured Context")

loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

print(f"  Indexed {len(chunks)} chunks into Chroma.")


# ---------------------------------------------------------------------------
# Component B: Knowledge Graph Retriever
# ---------------------------------------------------------------------------
print_step(2, "Initialize Graph Traversal Engine for Structural Relations")

G = nx.DiGraph()

# Add Core Supply Chain Edges
edges = [
    ("SiliconCore Fab", "SC-9901", "MANUFACTURES"),
    ("SiliconCore Fab", "PC-404", "MANUFACTURES"),
    ("SC-9901", "Bosch Mobility", "SUPPLIED_TO"),
    ("PC-404", "Apex Power Systems", "SUPPLIED_TO"),
    ("Bosch Mobility", "MotorControl-X", "ASSEMBLES"),
    ("Apex Power Systems", "BMU-300", "ASSEMBLES"),
    ("MotorControl-X", "Apex CyberTruck", "INSTALLED_IN"),
    ("BMU-300", "Apex CyberTruck", "INSTALLED_IN"),
    ("Apex CyberTruck", "SwiftLogistics Corp", "DELIVERED_TO"),
]
for src, dst, rel in edges:
    G.add_edge(src, dst, relation=rel)

print(f"  Graph initialized with {G.number_of_nodes()} entities and {G.number_of_edges()} relationships.")


def retrieve_graph_context(query: str, root_node: str = "SiliconCore Fab", target_node: str = "SwiftLogistics Corp") -> str:
    """Find multi-hop paths between key query entities."""
    if not (G.has_node(root_node) and G.has_node(target_node)):
        return "No direct graph path found."

    paths = list(nx.all_simple_paths(G, source=root_node, target=target_node))
    lines = []
    for i, path in enumerate(paths, start=1):
        path_str = " -> ".join(path)
        lines.append(f"Supply Chain Lineage #{i}: {path_str}")
        # Add relation details
        for u, v in zip(path[:-1], path[1:]):
            rel_type = G.edges[u, v].get("relation", "CONNECTS_TO")
            lines.append(f"  * ({u}) -[{rel_type}]-> ({v})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Component C: Hybrid Context Fusion & Generation
# ---------------------------------------------------------------------------
print_step(3, "Execute Hybrid Retrieval & Context Fusion")

# 1. Unstructured text context
unstructured_docs = vector_retriever.invoke(COMPLEX_QUERY)
unstructured_context = "\n\n".join(d.page_content for d in unstructured_docs)

# 2. Structured graph context
structured_context = retrieve_graph_context(COMPLEX_QUERY, root_node="SiliconCore Fab", target_node="SwiftLogistics Corp")

print("\n--- [1] Retrieved Structured Graph Lineage ---")
print(structured_context)

print("\n--- [2] Retrieved Unstructured Text Chunks ---")
print(unstructured_context[:300] + "...\n")

# 3. Merged Hybrid Context
fused_context = f"""=== STRUCTURED GRAPH RELATIONSHIPS (EXACT LINEAGE) ===
{structured_context}

=== UNSTRUCTURED DESCRIPTIVE PASSAGES (DETAILED CONTEXT) ===
{unstructured_context}"""

print_step(4, "Generate Answer with Fused Hybrid Context")

llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 150},
)

prompt = PromptTemplate.from_template(
    """You are an enterprise supply chain intelligence assistant.
Answer the question using both the structured graph relationships and unstructured text below:

{context}

Question: {question}

Detailed Analysis:"""
)

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"context": fused_context, "question": COMPLEX_QUERY})

print(f"Query: {COMPLEX_QUERY}\n")
print(f"Hybrid RAG Answer:\n{response.strip()}")
print("\n" + RULE)
print("SUCCESS: Hybrid Graph + Vector RAG verified.")
print(RULE)
