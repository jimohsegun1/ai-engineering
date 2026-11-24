"""
02_mcp_client_integration.py
============================
Module 03: Model Context Protocol (MCP) Servers and Clients
Course: AI Engineer Master Course

This script implements a complete Model Context Protocol (MCP) client that:
  1. Spawns and manages the MCP server subprocess using the standard `stdio` transport.
  2. Performs the JSON-RPC initialization handshake (`initialize`).
  3. Dynamically queries and inspects server capabilities:
     - Tools (`tools/list`)
     - Resources (`resources/list`, `resources/read`)
     - Prompts (`prompts/list`, `prompts/get`)
  4. Dispatches tool execution requests (`tools/call`) with parameters.
  5. Formats and logs the structured JSON-RPC protocol payloads.
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, Optional

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

# Path to the companion MCP server script
SERVER_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "01_mcp_server_custom_tools.py"
)


async def run_mcp_client():
    """Connects to the Supply Chain MCP Server via stdio and exercises all protocol features."""
    print("=" * 70)
    print("  MODEL CONTEXT PROTOCOL (MCP) - CLIENT INTEGRATION & DISCOVERY")
    print(f"  Target Server: {SERVER_SCRIPT_PATH}")
    print("=" * 70)

    # Configure stdio subprocess transport
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT_PATH, "--serve"],
        env=os.environ.copy()
    )

    print("\n[Phase 1] Launching Server Subprocess via Stdio Transport...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            # -----------------------------------------------------------------
            # Step 1: Handshake & Capability Negotiation
            # -----------------------------------------------------------------
            print("\n[Phase 2] Performing JSON-RPC Handshake (initialize)...")
            init_result = await session.initialize()
            print(f"  Connected to: {init_result.server_info.name} (v{init_result.server_info.version})")
            print(f"  Protocol Version: {init_result.protocol_version}")
            print(f"  Server Capabilities: {init_result.capabilities}")

            # -----------------------------------------------------------------
            # Step 2: Tool Discovery (tools/list)
            # -----------------------------------------------------------------
            print("\n[Phase 3] Tool Discovery (tools/list)...")
            tools_response = await session.list_tools()
            print(f"  Discovered {len(tools_response.tools)} available tool(s):")
            for t in tools_response.tools:
                print(f"\n  • Tool: {t.name}")
                print(f"    Description: {t.description}")
                schema_str = json.dumps(t.input_schema, indent=6)
                print(f"    Input Schema:\n      {schema_str.strip()}")

            # -----------------------------------------------------------------
            # Step 3: Resource Discovery & Reading (resources/list, read)
            # -----------------------------------------------------------------
            print("\n[Phase 4] Resource Discovery (resources/list & resources/read)...")
            try:
                resources_resp = await session.list_resources()
                print(f"  Discovered {len(resources_resp.resources)} resource(s):")
                for r in resources_resp.resources:
                    print(f"  • URI: {r.uri}")
                    print(f"    Name: {r.name}")
                    print(f"    MIME: {r.mime_type}")

                # Read the warehouse catalog resource
                print(f"\n  Reading resource 'catalog://warehouses'...")
                content = await session.read_resource("catalog://warehouses")
                if content.contents:
                    raw_text = content.contents[0].text
                    parsed = json.loads(raw_text)
                    print(f"  Retrieved {len(parsed.get('warehouses', []))} global warehouse hubs:")
                    for wh in parsed.get("warehouses", []):
                        print(f"    - [{wh['id']}] {wh['name']} ({wh['country']}) | Util: {wh['utilization_pct']}% | Status: {wh['status']}")
            except Exception as e:
                print(f"  Note on resource discovery: {e}")

            # -----------------------------------------------------------------
            # Step 4: Prompt Discovery & Fetching (prompts/list, prompts/get)
            # -----------------------------------------------------------------
            print("\n[Phase 5] Prompt Template Discovery (prompts/list)...")
            try:
                prompts_resp = await session.list_prompts()
                print(f"  Discovered {len(prompts_resp.prompts)} prompt template(s):")
                for p in prompts_resp.prompts:
                    print(f"  • Prompt: {p.name} - {p.description}")

                # Fetch prompt with parameters
                print(f"\n  Fetching prompt 'supply_chain_triage' with incident params...")
                prompt_content = await session.get_prompt("supply_chain_triage", {
                    "incident_id": "INC-2026-SH-PORT-01",
                    "urgency": "HIGH"
                })
                if prompt_content.messages:
                    sample_msg = prompt_content.messages[0].content
                    msg_text = sample_msg.text if hasattr(sample_msg, "text") else str(sample_msg)
                    print(f"  Prompt Preview:\n    {msg_text[:200]}...")
            except Exception as e:
                print(f"  Note on prompt discovery: {e}")

            # -----------------------------------------------------------------
            # Step 5: Tool Invocation (tools/call)
            # -----------------------------------------------------------------
            print("\n[Phase 6] Dispatching Tool Calls (tools/call)...")

            # Call 1: Check inventory for BAT-LFP-85
            part_query = "BAT-LFP-85"
            print(f"\n  [Call 1] get_inventory_status(part_id='{part_query}')...")
            inv_res = await session.call_tool("get_inventory_status", {"part_id": part_query})
            inv_text = inv_res.content[0].text if inv_res.content else ""
            inv_data = json.loads(inv_text)
            print("  Result:")
            print(f"    Item: {inv_data['data']['name']}")
            print(f"    Stock: {inv_data['data']['stock_level']} units (Threshold: {inv_data['data']['reorder_threshold']})")
            print(f"    Warehouse: {inv_data['data']['primary_warehouse']}")
            print(f"    Status: {inv_data['data']['status'].upper()}")

            # Call 2: Check shipping route
            print("\n  [Call 2] check_shipping_route(origin_port='SHANGHAI', destination_port='LONG_BEACH')...")
            route_res = await session.call_tool("check_shipping_route", {
                "origin_port": "SHANGHAI",
                "destination_port": "LONG_BEACH"
            })
            route_text = route_res.content[0].text if route_res.content else ""
            route_data = json.loads(route_text)
            print("  Result:")
            print(f"    Mode: {route_data['details']['mode']}")
            print(f"    Transit: {route_data['details']['transit_days']} days")
            print(f"    Reliability: {route_data['details']['reliability_score'] * 100}%")
            print(f"    Carriers: {', '.join(route_data['details']['carriers'])}")

            # Call 3: Calculate tariff cost
            declared_val = 74000.00
            print(f"\n  [Call 3] calculate_tariff_cost(part_id='{part_query}', declared_value={declared_val}, destination_country='USA')...")
            tariff_res = await session.call_tool("calculate_tariff_cost", {
                "part_id": part_query,
                "declared_value": declared_val,
                "destination_country": "USA"
            })
            tariff_text = tariff_res.content[0].text if tariff_res.content else ""
            tariff_data = json.loads(tariff_text)
            print("  Result:")
            print(f"    Declared Value: ${tariff_data['declared_value_usd']:,.2f}")
            print(f"    Tariffs & Fees: ${tariff_data['total_tariffs_and_fees_usd']:,.2f} ({tariff_data['effective_tariff_rate_pct']}%)")
            print(f"    Total Landed Cost: ${tariff_data['total_landed_cost_usd']:,.2f}")

    print("\n" + "=" * 70)
    print("  [SUCCESS] MCP Client stdio integration session completed cleanly!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_mcp_client())
