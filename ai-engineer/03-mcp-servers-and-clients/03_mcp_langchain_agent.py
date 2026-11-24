"""
03_mcp_langchain_agent.py
=========================
Module 03: Model Context Protocol (MCP) Servers and Clients
Course: AI Engineer Master Course

This script demonstrates how to integrate Model Context Protocol (MCP) tools
directly into LangChain and LangGraph agent architectures.

Key Capabilities Demonstrated:
  1. Dynamic MCP-to-LangChain Tool Adapter:
     Translates raw MCP tool schemas and call signatures into standard LangChain BaseTool objects.
  2. Multi-Tool Reasoning Loop (ReAct / LangGraph):
     The agent analyzes complex domain objectives, invokes relevant MCP tools in sequence,
     and aggregates tool results into an operational response.
  3. Resilient Multi-Provider Support:
     - Uses ChatOpenAI if `OPENAI_API_KEY` is configured.
     - Automatically falls back to an offline deterministic reasoning engine when no API keys are present,
       guaranteeing zero-error execution for all learners.
"""

import os
import sys
import json
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field, create_model

# LangChain tool abstractions
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import our custom MCP server definition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
mcp_module = import_module("01_mcp_server_custom_tools")
mcp_server = mcp_module.server

# Persistent thread pool for running async MCP tool calls from sync LangChain tool invocations
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Dynamic MCP-to-LangChain Adapter
# ---------------------------------------------------------------------------
def convert_mcp_tool_to_langchain(mcp_tool_meta, server_instance) -> BaseTool:
    """Wraps an MCP tool into a standard LangChain StructuredTool."""
    tool_name = mcp_tool_meta.name
    tool_desc = mcp_tool_meta.description or f"MCP tool: {tool_name}"

    def tool_runner(**kwargs) -> str:
        """Synchronously invokes the async MCP tool safely across threads."""
        def _invoke_in_thread():
            return asyncio.run(server_instance.call_tool(tool_name, kwargs))

        future = _EXECUTOR.submit(_invoke_in_thread)
        result = future.result()
        if isinstance(result, list):
            return result[0].text if result else ""
        elif hasattr(result, "content") and result.content:
            return result.content[0].text
        return str(result)

    # Build dynamic Pydantic schema from MCP input_schema properties
    properties = mcp_tool_meta.input_schema.get("properties", {})
    required_fields = mcp_tool_meta.input_schema.get("required", [])

    field_definitions = {}
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    for prop_name, prop_spec in properties.items():
        prop_type = type_map.get(prop_spec.get("type", "string"), str)
        is_req = prop_name in required_fields
        default_val = ... if is_req else None
        title = prop_spec.get("title", prop_name)
        field_definitions[prop_name] = (prop_type, Field(default=default_val, description=title))

    args_schema = create_model(f"{tool_name}_Schema", **field_definitions)

    return StructuredTool(
        name=tool_name,
        description=tool_desc,
        args_schema=args_schema,
        func=tool_runner,
    )


# ---------------------------------------------------------------------------
# ReAct Agent Orchestrator (Multi-Provider: OpenAI or Deterministic Fallback)
# ---------------------------------------------------------------------------
class SupplyChainMCPAgent:
    """Autonomous ReAct agent empowered with MCP supply chain tools."""

    def __init__(self, tools: List[BaseTool]):
        self.tools = {t.name: t for t in tools}
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def run(self, user_query: str) -> Dict[str, Any]:
        """Processes the query, selects tools, and synthesizes results."""
        print(f"\n[Agent Intake] User Query: \"{user_query}\"")
        print("-" * 70)

        # 1. Determine execution strategy
        if self.openai_key and not self.openai_key.startswith("sk-placeholder"):
            return self._run_openai_agent(user_query)
        else:
            return self._run_deterministic_react_agent(user_query)

    def _run_openai_agent(self, query: str) -> Dict[str, Any]:
        """Runs with OpenAI GPT model when API key is provided."""
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        print("  [Mode] Using ChatOpenAI (gpt-4o-mini) with bound MCP tools...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        tool_list = list(self.tools.values())
        agent = create_react_agent(llm, tool_list)

        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        final_msg = response["messages"][-1].content
        return {
            "status": "success",
            "provider": "openai",
            "output": final_msg,
            "messages": response["messages"],
        }

    def _run_deterministic_react_agent(self, query: str) -> Dict[str, Any]:
        """Deterministic ReAct reasoning loop ensuring 100% offline verification."""
        print("  [Mode] Running Deterministic ReAct Engine (Zero-API Offline Mode)...")
        thoughts = []
        action_trace = []

        # Step 1: Detect entity and query intent
        query_upper = query.upper()
        detected_part = None
        for candidate in ["GPU-H100", "BAT-LFP-85", "MCU-32BIT", "CHIP-AI-PRO"]:
            if candidate in query_upper:
                detected_part = candidate
                break
        if not detected_part:
            detected_part = "GPU-H100"

        # Thought 1: Inventory status check
        print(f"\n  [Thought 1] Query requests inventory health for part '{detected_part}'.")
        print(f"  [Action 1] Invoking MCP tool: get_inventory_status(part_id='{detected_part}')")
        inv_tool = self.tools["get_inventory_status"]
        inv_raw = inv_tool.invoke({"part_id": detected_part})
        inv_data = json.loads(inv_raw)
        action_trace.append({"tool": "get_inventory_status", "result": inv_data})
        print(f"  [Observation 1] Current Stock: {inv_data['data']['stock_level']} units. Status: {inv_data['data']['status']}.")

        # Thought 2: Route evaluation
        origin = "FRANKFURT" if "FRANKFURT" in query_upper else "SHANGHAI"
        destination = "LONG_BEACH" if ("LONG BEACH" in query_upper or "USA" in query_upper) else "ROTTERDAM"
        print(f"\n  [Thought 2] Assessing alternative freight corridors between {origin} and {destination}.")
        print(f"  [Action 2] Invoking MCP tool: check_shipping_route(origin_port='{origin}', destination_port='{destination}')")
        route_tool = self.tools["check_shipping_route"]
        route_raw = route_tool.invoke({"origin_port": origin, "destination_port": destination})
        route_data = json.loads(route_raw)
        action_trace.append({"tool": "check_shipping_route", "result": route_data})
        print(f"  [Observation 2] Route transit: {route_data['details']['transit_days']} days via {route_data['details']['mode']}.")

        # Thought 3: Tariff calculation
        declared_value = 96000.0 if "96,000" in query or "96000" in query else 50000.0
        dest_country = "USA" if "USA" in query_upper or "UNITED STATES" in query_upper else "EU"
        print(f"\n  [Thought 3] Calculating customs tariffs for declared value ${declared_value:,.2f} to {dest_country}.")
        print(f"  [Action 3] Invoking MCP tool: calculate_tariff_cost(part_id='{detected_part}', declared_value={declared_value}, destination_country='{dest_country}')")
        tariff_tool = self.tools["calculate_tariff_cost"]
        tariff_raw = tariff_tool.invoke({
            "part_id": detected_part,
            "declared_value": declared_value,
            "destination_country": dest_country
        })
        tariff_data = json.loads(tariff_raw)
        action_trace.append({"tool": "calculate_tariff_cost", "result": tariff_data})
        print(f"  [Observation 3] Landed Cost: ${tariff_data['total_landed_cost_usd']:,.2f} (Tariffs: ${tariff_data['total_tariffs_and_fees_usd']:,.2f}).")

        # Synthesize final resolution
        print("\n  [Thought 4] All critical parameters collected from MCP tools. Synthesizing executive report.")
        summary = f"""
======================================================================
  EXECUTIVE SUPPLY CHAIN TRIAGE & DISRUPTION MITIGATION REPORT
======================================================================
1. Inventory Health:
   - Part: {inv_data['data']['name']} ({inv_data['data']['part_id']})
   - Stock Status: {inv_data['data']['stock_level']} units available at {inv_data['data']['primary_warehouse']}
   - Alert Status: {inv_data['data']['status'].upper()} (Reorder needed: {inv_data['reorder_needed']})

2. Route & Transit Feasibility:
   - Corridor: {origin} -> {destination}
   - Freight Mode: {route_data['details']['mode']}
   - Lead Time: {route_data['details']['transit_days']} days
   - Reliability Score: {route_data['details']['reliability_score'] * 100}%
   - Approved Carriers: {', '.join(route_data['details']['carriers'])}

3. Landed Duty & Tariff Analysis:
   - Declared Goods Value: ${tariff_data['declared_value_usd']:,.2f}
   - Effective Tariff Rate: {tariff_data['effective_tariff_rate_pct']}%
   - Total Customs Duties & Fees: ${tariff_data['total_tariffs_and_fees_usd']:,.2f}
   - Projected Total Landed Cost: ${tariff_data['total_landed_cost_usd']:,.2f}

4. Recommendation:
   - Immediate dispatch via {route_data['details']['carriers'][0]} recommended to meet SLA.
   - Expedite customs bonded pre-clearance at {destination} to mitigate demurrage.
======================================================================
"""
        return {
            "status": "success",
            "provider": "deterministic_react_mcp",
            "output": summary,
            "trace": action_trace,
        }


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
async def main():
    print("=" * 70)
    print("  LANGCHAIN / LANGGRAPH AGENT INTEGRATION WITH MCP TOOLS")
    print("=" * 70)

    # 1. Discover tools from MCP server
    print("\n[Step 1] Querying MCP Server Tool Registry...")
    mcp_tools = await mcp_server.list_tools()
    print(f"  Found {len(mcp_tools)} tool(s) registered on MCP server.")

    # 2. Convert MCP tools to LangChain tools
    print("\n[Step 2] Adapting MCP Tools into LangChain StructuredTool Objects...")
    langchain_tools = []
    for t in mcp_tools:
        lc_tool = convert_mcp_tool_to_langchain(t, mcp_server)
        langchain_tools.append(lc_tool)
        print(f"  • Adapted: {lc_tool.name} (args: {list(lc_tool.args.keys())})")

    # 3. Instantiate Agent
    print("\n[Step 3] Initializing Supply Chain MCP Agent...")
    agent = SupplyChainMCPAgent(tools=langchain_tools)

    # 4. Execute complex operational scenario
    scenario_query = (
        "We have an active shortage alert at our Long Beach DC for GPU-H100 accelerators. "
        "Check inventory status, evaluate an express freight corridor from Frankfurt to Long Beach, "
        "and calculate import tariffs for $96,000 worth of parts to the USA."
    )

    result = agent.run(scenario_query)
    print(result["output"])


if __name__ == "__main__":
    asyncio.run(main())
