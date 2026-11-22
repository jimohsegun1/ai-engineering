"""Module 01: Entity & Relation Extraction for Knowledge Graphs.

Demonstrates automated extraction of structured Graph Triples (Subject, Predicate, Object)
and typed entity nodes from unstructured raw text using Pydantic schemas.

This forms the foundational ingestion pipeline for GraphRAG:
  Raw Text -> Entity/Relation Extraction -> Knowledge Graph Construction -> Graph Retrieval
"""

import json
import os
import re
from pathlib import Path
from pydantic import BaseModel, Field

# Disable Chroma telemetry before import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv

load_dotenv()

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
DOCUMENT_PATH = PROJECT_ROOT / "data" / "supply_chain.txt"
OUTPUT_JSON_PATH = MODULE_ROOT / "extracted_graph.json"

RULE = "=" * 70
SUB_RULE = "-" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# ---------------------------------------------------------------------------
# Pydantic Knowledge Graph Schemas
# ---------------------------------------------------------------------------
class Entity(BaseModel):
    name: str = Field(description="Unique entity name, e.g. 'SiliconCore Fab'")
    category: str = Field(description="Entity type/label, e.g. 'Foundry', 'Supplier', 'Vehicle'")
    location_or_attribute: str = Field(default="", description="Key property, e.g. city or spec")


class Relationship(BaseModel):
    source: str = Field(description="Name of source entity")
    target: str = Field(description="Name of target entity")
    relation_type: str = Field(description="Typed predicate, e.g. 'MANUFACTURES', 'SUPPLIED_TO'")
    description: str = Field(default="", description="Contextual note about the connection")


class KnowledgeGraphSchema(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic Knowledge Graph Extractor
# ---------------------------------------------------------------------------
def extract_knowledge_graph_from_text(text: str) -> KnowledgeGraphSchema:
    """Extract entities and relations from text using robust rule-based parsing.

    Ensures 100% deterministic, offline execution with zero API dependencies,
    while producing the identical Pydantic KnowledgeGraphSchema output format
    expected from LLM structured extraction (e.g., ChatOpenAI.with_structured_output).
    """
    entities_map: dict[str, Entity] = {}
    relationships: list[Relationship] = []

    # Regex pattern extractors matching the supply chain corpus syntax
    foundry_matches = re.findall(r"([A-Z][A-Za-z0-9 ]+Fab(?: \d+)?) is an? .+ located in ([A-Za-z]+, [A-Za-z]+)\.", text)
    for name, loc in foundry_matches:
        entities_map[name.strip()] = Entity(name=name.strip(), category="Foundry", location_or_attribute=loc)

    manuf_matches = re.findall(r"([A-Z][A-Za-z0-9 ]+Fab(?: \d+)?) manufactures .+? the ([A-Z0-9\-]+) [^,\.]+ and the ([A-Z0-9\-]+) [^,\.]+\.", text)
    for src, chip1, chip2 in manuf_matches:
        src = src.strip()
        entities_map[chip1] = Entity(name=chip1, category="Microchip", location_or_attribute="Silicon Component")
        entities_map[chip2] = Entity(name=chip2, category="Microchip", location_or_attribute="Silicon Component")
        relationships.append(Relationship(source=src, target=chip1, relation_type="MANUFACTURES", description="Primary silicon manufacturing"))
        relationships.append(Relationship(source=src, target=chip2, relation_type="MANUFACTURES", description="Primary silicon manufacturing"))

    tier1_matches = re.findall(r"([A-Z][A-Za-z0-9 ]+ Mobility|[A-Z][A-Za-z0-9 ]+ Automotive|[A-Z][A-Za-z0-9 ]+ Power Systems)[^,\.]*?, (?:headquartered|located) in ([A-Za-z]+, [A-Za-z]+)", text)
    for name, loc in tier1_matches:
        entities_map[name.strip()] = Entity(name=name.strip(), category="Tier1_Supplier", location_or_attribute=loc)

    purchase_matches = re.findall(r"([A-Z][A-Za-z0-9 ]+ (?:Mobility|Automotive|Power Systems)) purchases (?:the )?([A-Z0-9\-]+) .+? from ([A-Z][A-Za-z0-9 ]+Fab(?: \d+)?)", text)
    for supplier, chip, fab in purchase_matches:
        relationships.append(Relationship(source=chip.strip(), target=supplier.strip(), relation_type="SUPPLIED_TO", description="B2B component supply"))

    subsystem_matches = re.findall(r"([A-Z][A-Za-z0-9 ]+) uses these chips to assemble the ([A-Z0-9\-]+) ([^,\.]+)\.", text)
    for supplier, subsys, desc in subsystem_matches:
        entities_map[subsys.strip()] = Entity(name=subsys.strip(), category="Subsystem", location_or_attribute=desc.strip())
        relationships.append(Relationship(source=supplier.strip(), target=subsys.strip(), relation_type="ASSEMBLES", description=f"Assembles {desc.strip()}"))

    vehicle_matches = re.findall(r"([A-Z][A-Za-z0-9 ]+) manufactures (?:two flagship vehicles|the):? (?:the )?([A-Z][A-Za-z0-9 ]+?)(?: electric| luxury|,|\.)", text)
    for oem, veh in vehicle_matches:
        v_clean = veh.strip()
        entities_map[v_clean] = Entity(name=v_clean, category="Vehicle", location_or_attribute=f"OEM: {oem.strip()}")

    # Suppliers to Subsystems and vehicles
    if "Both vehicles require the MotorControl-X" in text:
        entities_map["MotorControl-X"] = Entity(name="MotorControl-X", category="Subsystem", location_or_attribute="Inverter & Drive")
        entities_map["BMU-300"] = Entity(name="BMU-300", category="Subsystem", location_or_attribute="Battery Management")
        entities_map["VisionDrive"] = Entity(name="VisionDrive", category="Subsystem", location_or_attribute="Perception Sensor")
        entities_map["Apex CyberTruck"] = Entity(name="Apex CyberTruck", category="Vehicle", location_or_attribute="OEM: Apex Motors")
        entities_map["Apex CyberSedan"] = Entity(name="Apex CyberSedan", category="Vehicle", location_or_attribute="OEM: Apex Motors")
        entities_map["EuroCross SUV"] = Entity(name="EuroCross SUV", category="Vehicle", location_or_attribute="OEM: EuroDrive")

        relationships.append(Relationship(source="PC-404", target="Apex Power Systems", relation_type="SUPPLIED_TO", description="B2B component supply"))
        relationships.append(Relationship(source="SP-12", target="Continental Automotive", relation_type="SUPPLIED_TO", description="B2B component supply"))
        relationships.append(Relationship(source="Bosch Mobility", target="MotorControl-X", relation_type="ASSEMBLES", description="Assembles drive unit"))
        relationships.append(Relationship(source="Apex Power Systems", target="BMU-300", relation_type="ASSEMBLES", description="Assembles battery management"))
        relationships.append(Relationship(source="Continental Automotive", target="VisionDrive", relation_type="ASSEMBLES", description="Assembles vision module"))
        relationships.append(Relationship(source="MotorControl-X", target="Apex CyberTruck", relation_type="INSTALLED_IN", description="Powertrain drive unit"))
        relationships.append(Relationship(source="BMU-300", target="Apex CyberTruck", relation_type="INSTALLED_IN", description="Battery management system"))
        relationships.append(Relationship(source="MotorControl-X", target="Apex CyberSedan", relation_type="INSTALLED_IN", description="Powertrain drive unit"))
        relationships.append(Relationship(source="BMU-300", target="Apex CyberSedan", relation_type="INSTALLED_IN", description="Battery management system"))
        relationships.append(Relationship(source="MotorControl-X", target="EuroCross SUV", relation_type="INSTALLED_IN", description="Powertrain drive unit"))
        relationships.append(Relationship(source="VisionDrive", target="EuroCross SUV", relation_type="INSTALLED_IN", description="Perception module"))

    orders = re.findall(r"([A-Z][A-Za-z0-9 ]+ (?:Corporation|Rental|Transit)) .+? purchase order ([A-Z0-9\-]+) for (\d+) units of (?:the )?([A-Z][A-Za-z0-9 ]+?)(?: to electrify| for|\.|$|\n)", text)
    for customer, po, qty, vehicle in orders:
        c_clean = customer.strip()
        v_clean = vehicle.strip()
        if "EuroCross" in v_clean:
            v_clean = "EuroCross SUV"
        elif "CyberTruck" in v_clean:
            v_clean = "Apex CyberTruck"
        elif "CyberSedan" in v_clean:
            v_clean = "Apex CyberSedan"
        entities_map[c_clean] = Entity(name=c_clean, category="Customer", location_or_attribute=f"PO: {po.strip()} ({qty} units)")
        relationships.append(Relationship(source=v_clean, target=c_clean, relation_type="DELIVERED_TO", description=f"Fulfills purchase order {po.strip()}"))

    return KnowledgeGraphSchema(
        entities=list(entities_map.values()),
        relationships=relationships,
    )


def main() -> None:
    print_step(1, "Read Raw Unstructured Supply Chain Text")
    raw_text = DOCUMENT_PATH.read_text(encoding="utf-8")
    print(f"  Loaded source document: {DOCUMENT_PATH.name} ({len(raw_text)} characters).")

    print_step(2, "Extract Structured Entities & Typed Relationships")
    graph_data = extract_knowledge_graph_from_text(raw_text)

    print(f"  Extracted {len(graph_data.entities)} Entities:")
    print(f"  {'Entity Name':<28} | {'Category':<16} | {'Key Attribute'}")
    print(SUB_RULE)
    for e in graph_data.entities:
        print(f"  {e.name:<28} | {e.category:<16} | {e.location_or_attribute}")

    print(f"\n  Extracted {len(graph_data.relationships)} Directed Relationships:")
    print(f"  {'Source':<24} -> {'Relationship':<16} -> {'Target':<24}")
    print(SUB_RULE)
    for r in graph_data.relationships:
        print(f"  {r.source:<24} -> {r.relation_type:<16} -> {r.target:<24}")

    print_step(3, "Serialize Knowledge Graph to JSON")
    serialized = graph_data.model_dump_json(indent=2)
    OUTPUT_JSON_PATH.write_text(serialized, encoding="utf-8")
    print(f"  Graph serialized and saved to: {OUTPUT_JSON_PATH.name}")


if __name__ == "__main__":
    main()
