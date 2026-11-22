"""Module 02: Idiomatic Cypher Query Patterns.

Demonstrates essential enterprise Cypher query patterns used in Knowledge Graphs:
  1. Idempotent Node & Edge Creation (MERGE vs CREATE)
  2. Property Filtering & Label Projections (MATCH ... WHERE)
  3. Multi-Hop Path Traversals (MATCH path = (a)-[*1..3]->(b))
  4. Graph Aggregations & Degree Centrality (count(r), collect())
  5. Shortest Path & Impact Analysis
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

load_dotenv()

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "aiengineering_password")

RULE = "=" * 70
SUB_RULE = "-" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


class CypherPatternSimulator:
    """Simulates realistic Cypher query execution for local offline testing."""

    def __init__(self):
        self.entities = {
            "SiliconCore Fab": {"label": "Foundry", "location": "Dresden"},
            "SC-9901": {"label": "Microchip", "type": "MCU"},
            "Bosch Mobility": {"label": "Supplier", "location": "Stuttgart"},
            "MotorControl-X": {"label": "Subsystem", "category": "Drive"},
            "Apex CyberTruck": {"label": "Vehicle", "oem": "Apex Motors"},
            "SwiftLogistics Corp": {"label": "Customer", "order": "ORD-881"},
        }
        self.relationships = [
            ("SiliconCore Fab", "SC-9901", "MANUFACTURES", {"annual_vol": 2500000}),
            ("SC-9901", "Bosch Mobility", "SUPPLIED_TO", {"lead_days": 14}),
            ("Bosch Mobility", "MotorControl-X", "ASSEMBLES", {"unit_cost": 1200}),
            ("MotorControl-X", "Apex CyberTruck", "INSTALLED_IN", {"redundancy": "dual"}),
            ("Apex CyberTruck", "SwiftLogistics Corp", "DELIVERED_TO", {"units": 500}),
        ]

    def execute(self, title: str, cypher: str) -> list[dict[str, Any]]:
        print(f"\n[Cypher Pattern]: {title}")
        print("```cypher")
        print(cypher.strip())
        print("```")

        # Return mock results corresponding to the pattern
        if "MERGE" in cypher:
            rows = [{"nodes_merged": 2, "relationships_created": 1}]
        elif "allShortestPaths" in cypher or "[*1..5]" in cypher:
            rows = [{
                "path": "SiliconCore Fab -> SC-9901 -> Bosch Mobility -> MotorControl-X -> Apex CyberTruck -> SwiftLogistics Corp",
                "hop_count": 5,
            }]
        elif "count(r) AS degree" in cypher or "degree" in cypher.lower():
            rows = [
                {"entity": "Bosch Mobility", "label": "Supplier", "degree": 4},
                {"entity": "SiliconCore Fab", "label": "Foundry", "degree": 3},
                {"entity": "Apex CyberTruck", "label": "Vehicle", "degree": 3},
                {"entity": "SwiftLogistics Corp", "label": "Customer", "degree": 1},
            ]
        else:
            rows = [
                {"foundry": "SiliconCore Fab", "chip": "SC-9901", "supplier": "Bosch Mobility", "vehicle": "Apex CyberTruck"}
            ]

        print("Results:")
        for r in rows:
            print(f"  {r}")
        return rows


def main() -> None:
    simulator = CypherPatternSimulator()

    print_step(1, "Idempotent Node & Edge Creation (MERGE)")
    pattern_1 = """
    MERGE (f:Foundry {name: 'SiliconCore Fab'})
      ON CREATE SET f.location = 'Dresden', f.created_at = timestamp()
    MERGE (c:Microchip {name: 'SC-9901'})
      ON CREATE SET c.category = 'Powertrain MCU'
    MERGE (f)-[r:MANUFACTURES]->(c)
      ON CREATE SET r.yield_rate = 0.985
    RETURN f.name, type(r), c.name;
    """
    simulator.execute("MERGE Nodes with On-Create Attributes", pattern_1)

    print_step(2, "Multi-Hop Variable-Length Path Traversal")
    pattern_2 = """
    MATCH path = (start:Foundry {name: 'SiliconCore Fab'})-[*1..5]->(target:Customer)
    RETURN [n IN nodes(path) | n.name] AS entity_chain,
           length(path) AS hop_count;
    """
    simulator.execute("Variable-Length Lineage Search (*1..5)", pattern_2)

    print_step(3, "Centrality & Network Bottleneck Analysis")
    pattern_3 = """
    MATCH (n)-[r]-()
    RETURN labels(n)[0] AS entity_type,
           n.name AS entity_name,
           count(r) AS degree
    ORDER BY degree DESC
    LIMIT 5;
    """
    simulator.execute("Degree Centrality & Supply Bottlenecks", pattern_3)

    print_step(4, "Shortest Path & Upstream Failure Propagation")
    pattern_4 = """
    MATCH (source:Foundry {name: 'SiliconCore Fab'}),
          (dest:Customer {name: 'SwiftLogistics Corp'}),
          p = allShortestPaths((source)-[*..6]->(dest))
    RETURN [n IN nodes(p) | coalesce(n.name, 'Unknown')] AS critical_path,
           length(p) AS total_hops;
    """
    simulator.execute("All Shortest Paths Traversal", pattern_4)

    print("\n" + RULE)
    print("SUCCESS: Key Cypher patterns for GraphRAG demonstrated.")
    print(RULE)


if __name__ == "__main__":
    main()
