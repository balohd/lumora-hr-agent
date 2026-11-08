"""CI smoke test: start the MCP server over stdio, discover tools, call two of them.

    python scripts/mcp_smoke.py            # stdio (spawns the server)
    MCP_SERVER_URL=http://host:8765/mcp python scripts/mcp_smoke.py
Exits non-zero on failure.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.mcp_client import MCPClientManager  # noqa: E402


async def main() -> int:
    m = MCPClientManager()
    await m.connect()
    if not m.connected:
        print("MCP connection failed:", m.last_error)
        return 1
    try:
        tools = m.tool_schemas()
        print(f"transport={m.transport} server={m.server_info} tools={len(tools)}")
        for t in tools:
            print(f"  - {t['name']}({', '.join(t['input_schema'].get('properties', {}))})")
        assert len(tools) >= 5, "expected at least 5 MCP tools"
        res = await m.call_tool("search_policy_documents", {"query": "how much notice for PTO", "k": 2})
        print("search_policy_documents ->", [h["citation"] for h in res["hits"]])
        res2 = await m.call_tool("check_pto_balance", {"employee_id": "E1002"})
        print("check_pto_balance ->", json.dumps({k: res2[k] for k in ("employee_id", "current_balance")}))
        assert res["hits"] and res2["current_balance"] == 4.5
    finally:
        await m.close()
    print("MCP smoke test OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
