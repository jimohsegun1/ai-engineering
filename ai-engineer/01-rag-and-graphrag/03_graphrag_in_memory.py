"""Module 01: In-Memory GraphRAG Engine with NetworkX.

Demonstrates an end-to-end GraphRAG pipeline implemented in pure Python without external databases:
  1. Entity Embedding: Embeds node names and attributes using HuggingFace MiniLM.
  2. Seed Node Retrieval: Finds initial entry-point nodes using semantic cosine similarity.
  3. Subgraph Traversal: Traverses k-hop neighbors outward from the seed nodes.
  4. Knowledge Subgraph Synthesis: Formats the traversed graph paths into grounded context.
  5. Answer Generation: Generates an accurate response to complex relational questions.
"""

import json
import os
from pathlib import Path
from typing import Any

# Disable Chroma telemetry before import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

load_dotenv()

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
GRAPH_DATA_PATH = MODULE_ROOT / "extracted_graph.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-base"

SAMPLE_QUERIES = [
    "Which vehicles use chips produced by SiliconCore Fab?",
    "If TSMC Fab 18 has a fire, which Tier-1 suppliers are affected?",
]

RULE = "=" * 70
SUB_RULE = "-" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


class InMemoryGraphRAG:
    """GraphRAG engine combining vector search for seed discovery with NetworkX graph traversal."""

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.graph = nx.DiGraph()
        self.embeddings = embeddings
        self.node_names: list[str] = []
        self.node_embeddings: np.ndarray | None = None

    def load_graph_data(self, data: dict[str, list[dict]]) -> None:
        """Populate graph from structured entity and relationship lists."""
        for ent in data.get("entities", []):
            self.graph.add_node(
                ent["name"],
                category=ent.get("category", ""),
                attribute=ent.get("location_or_attribute", ""),
            )

        for rel in data.get("relationships", []):
            self.graph.add_edge(
                rel["source"],
                rel["target"],
                rel_type=rel.get("relation_type", "RELATED_TO"),
                desc=rel.get("description", ""),
            )

        # Index node semantic representations
        self.node_names = list(self.graph.nodes())
        descriptions = [
            f"{n} ({self.graph.nodes[n].get('category', '')}): {self.graph.nodes[n].get('attribute', '')}"
            for n in self.node_names
        ]
        vectors = self.embeddings.embed_documents(descriptions)
        self.node_embeddings = np.array(vectors)

    def find_seed_nodes(self, query: str, top_k: int = 2) -> list[str]:
        """Find the most semantically relevant seed nodes for a given query."""
        if self.node_embeddings is None or len(self.node_names) == 0:
            return []

        query_vec = np.array(self.embeddings.embed_query(query))
        # Compute cosine similarity
        norm_node = np.linalg.norm(self.node_embeddings, axis=1)
        norm_query = np.linalg.norm(query_vec)
        scores = np.dot(self.node_embeddings, query_vec) / (norm_node * norm_query + 1e-9)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.node_names[i] for i in top_indices]

    def traverse_subgraph(self, seed_nodes: list[str], max_hops: int = 2) -> list[str]:
        """Extract multi-hop relationship paths starting from seed nodes."""
        facts: list[str] = []
        visited_edges: set[tuple[str, str]] = set()

        for seed in seed_nodes:
            # Gather outgoing paths up to max_hops
            subgraph_nodes = nx.single_source_shortest_path_length(self.graph, seed, cutoff=max_hops)
            for u in subgraph_nodes:
                for v in self.graph.successors(u):
                    if (u, v) not in visited_edges:
                        visited_edges.add((u, v))
                        rel_data = self.graph.edges[u, v]
                        rel_name = rel_data.get("rel_type", "RELATED_TO")
                        u_meta = self.graph.nodes[u].get("category", "")
                        v_meta = self.graph.nodes[v].get("category", "")
                        fact = f"Entity '{u}' ({u_meta}) -[{rel_name}]-> Entity '{v}' ({v_meta})"
                        facts.append(fact)

        return facts

    def query(self, query_text: str, chain: Any, max_hops: int = 2) -> dict[str, Any]:
        """Execute full GraphRAG query: seed discovery -> traversal -> prompt augmentation -> LLM."""
        seeds = self.find_seed_nodes(query_text, top_k=2)
        facts = self.traverse_subgraph(seeds, max_hops=max_hops)

        context_str = "\n".join(facts) if facts else "No relevant graph connections found."
        response = chain.invoke({"context": context_str, "question": query_text})

        return {
            "query": query_text,
            "seed_nodes": seeds,
            "traversed_facts": facts,
            "answer": response.strip(),
        }


def main() -> None:
    print_step(1, "Initialize GraphRAG Engine with Shared MiniLM Embeddings")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    engine = InMemoryGraphRAG(embeddings)

    print_step(2, "Load Knowledge Graph Data")
    if not GRAPH_DATA_PATH.exists():
        # Fallback inline graph if file not yet generated
        raw_data = {
            "entities": [
                {"name": "SiliconCore Fab", "category": "Foundry", "location_or_attribute": "Dresden, Germany"},
                {"name": "TSMC Fab 18", "category": "Foundry", "location_or_attribute": "Tainan, Taiwan"},
                {"name": "SC-9901", "category": "Microchip", "location_or_attribute": "Powertrain MCU"},
                {"name": "PC-404", "category": "Microchip", "location_or_attribute": "Thermal Controller"},
                {"name": "SP-12", "category": "Microchip", "location_or_attribute": "Radar Processor"},
                {"name": "Bosch Mobility", "category": "Tier1_Supplier", "location_or_attribute": "Stuttgart, Germany"},
                {"name": "Continental Automotive", "category": "Tier1_Supplier", "location_or_attribute": "Hanover, Germany"},
                {"name": "Apex CyberTruck", "category": "Vehicle", "location_or_attribute": "OEM: Apex Motors"},
                {"name": "EuroCross SUV", "category": "Vehicle", "location_or_attribute": "OEM: EuroDrive"},
            ],
            "relationships": [
                {"source": "SiliconCore Fab", "target": "SC-9901", "relation_type": "MANUFACTURES"},
                {"source": "SiliconCore Fab", "target": "PC-404", "relation_type": "MANUFACTURES"},
                {"source": "TSMC Fab 18", "target": "SP-12", "relation_type": "MANUFACTURES"},
                {"source": "SC-9901", "target": "Bosch Mobility", "relation_type": "SUPPLIED_TO"},
                {"source": "SP-12", "target": "Continental Automotive", "relation_type": "SUPPLIED_TO"},
                {"source": "Bosch Mobility", "target": "Apex CyberTruck", "relation_type": "INSTALLED_IN"},
                {"source": "Continental Automotive", "target": "EuroCross SUV", "relation_type": "INSTALLED_IN"},
            ],
        }
    else:
        raw_data = json.loads(GRAPH_DATA_PATH.read_text(encoding="utf-8"))

    engine.load_graph_data(raw_data)
    print(f"  Indexed Graph: {engine.graph.number_of_nodes()} nodes, {engine.graph.number_of_edges()} edges.")

    print_step(3, "Initialize Generation Pipeline")
    llm = HuggingFacePipeline.from_model_id(
        model_id=CHAT_MODEL,
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 120},
    )

    prompt = PromptTemplate.from_template(
        """Answer the question based only on the graph facts below:

Graph Facts:
{context}

Question: {question}

Answer:"""
    )
    chain = prompt | llm | StrOutputParser()

    print_step(4, "Execute Multi-Hop GraphRAG Queries")
    for q in SAMPLE_QUERIES:
        res = engine.query(q, chain, max_hops=3)
        print(f"\nQuery: {res['query']!r}")
        print(f"Seed Nodes Discovered: {res['seed_nodes']}")
        print(f"Traversed Graph Facts ({len(res['traversed_facts'])} edges):")
        for f in res["traversed_facts"][:4]:
            print(f"  * {f}")
        print(f"Answer: {res['answer']}")


if __name__ == "__main__":
    main()
