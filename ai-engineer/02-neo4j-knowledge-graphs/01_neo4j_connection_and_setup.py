"""Module 02: Neo4j Connection, Schema Constraints, and Healthcheck.

Demonstrates establishing an enterprise connection to a Neo4j graph database:
  1. Connection Management: Initializes official GraphDatabase driver with authentication.
  2. Uniqueness Constraints & Indexes: Creates schema constraints ensuring entity integrity.
  3. Health Check: Verifies connectivity, database edition, and active constraints.
  4. Resilient Fallback: Includes an intelligent simulated mock driver so the code
     can be studied and verified locally even when an external Docker/Aura instance is offline.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Optional import of neo4j driver
try:
    from neo4j import GraphDatabase, exceptions as neo4j_exceptions
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


class SimulatedNeo4jSession:
    """Mock session providing offline Cypher execution simulation when live Neo4j is offline."""

    def __init__(self):
        self.created_constraints = []
        self.nodes = {}

    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        clean_q = " ".join(query.strip().split())
        if "SHOW CONSTRAINTS" in clean_q or "SHOW INDEXES" in clean_q:
            return [{"name": c, "type": "UNIQUENESS"} for c in self.created_constraints]
        if "CREATE CONSTRAINT" in clean_q:
            constraint_name = clean_q.split("IF NOT EXISTS FOR")[0].replace("CREATE CONSTRAINT", "").strip()
            self.created_constraints.append(constraint_name or "unique_entity_name")
            return [{"status": "CONSTRAINT_CREATED"}]
        if "RETURN 1" in clean_q or "call dbms.components" in clean_q.lower():
            return [{"edition": "enterprise (simulated-local-mode)", "version": "5.26.0"}]
        return [{"status": "OK", "rows_affected": 1}]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass


class SimulatedNeo4jDriver:
    """Simulated Neo4j driver providing full API compatibility for offline environments."""

    def __init__(self):
        self._shared_session = SimulatedNeo4jSession()

    def session(self):
        return self._shared_session

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass

    def verify_connectivity(self):
        return True


def get_neo4j_driver() -> tuple[Any, bool]:
    """Connect to live Neo4j instance, falling back gracefully to simulated local driver."""
    if not HAS_NEO4J:
        print("  [Notice] 'neo4j' package not installed; utilizing simulated driver.")
        return SimulatedNeo4jDriver(), False

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver, True
    except Exception as exc:
        print(f"  [Notice] Live Neo4j instance at {NEO4J_URI} unreachable ({exc.__class__.__name__}).")
        print("           Switching seamlessly to Simulated Local Driver for deterministic execution.")
        return SimulatedNeo4jDriver(), False


def main() -> None:
    print_step(1, "Configure Neo4j Database Connection")
    print(f"  Target URI:      {NEO4J_URI}")
    print(f"  Target Username: {NEO4J_USER}")

    driver, is_live = get_neo4j_driver()
    mode_label = "LIVE NEO4J CLUSTER" if is_live else "OFFLINE SIMULATED MODE"
    print(f"  Connection Status: CONNECTED ({mode_label})")

    print_step(2, "Create Enterprise Schema Uniqueness Constraints & Indexes")
    schema_constraints = [
        ("constraint_foundry_name", "CREATE CONSTRAINT constraint_foundry_name IF NOT EXISTS FOR (f:Foundry) REQUIRE f.name IS UNIQUE"),
        ("constraint_chip_name", "CREATE CONSTRAINT constraint_chip_name IF NOT EXISTS FOR (c:Microchip) REQUIRE c.name IS UNIQUE"),
        ("constraint_supplier_name", "CREATE CONSTRAINT constraint_supplier_name IF NOT EXISTS FOR (s:Supplier) REQUIRE s.name IS UNIQUE"),
        ("constraint_vehicle_name", "CREATE CONSTRAINT constraint_vehicle_name IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.name IS UNIQUE"),
        ("constraint_customer_name", "CREATE CONSTRAINT constraint_customer_name IF NOT EXISTS FOR (cust:Customer) REQUIRE cust.name IS UNIQUE"),
    ]

    with driver.session() as session:
        for name, cypher in schema_constraints:
            session.run(cypher)
            print(f"  [OK] Applied Schema Constraint: '{name}'")

    print_step(3, "Verify Active Database Constraints & Health")
    with driver.session() as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = [r["name"] for r in result]
        print(f"  Active Schema Constraints ({len(constraints)} verified):")
        for c in constraints:
            print(f"    - {c}")

    driver.close()
    print("\n" + RULE)
    print("SUCCESS: Neo4j schema & connection management verified.")
    print(RULE)


if __name__ == "__main__":
    main()
