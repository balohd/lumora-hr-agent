"""MCP tool discovery and tool calls over the real stdio transport (as used by the app)."""
import asyncio

from agent.mcp_client import MCPClientManager

EXPECTED = {"search_policy_documents", "get_policy_section", "list_policy_documents", "lookup_employee_profile",
            "check_pto_balance", "lookup_benefits_status", "check_policy_compliance", "create_mock_hr_ticket",
            "draft_hr_email", "generate_onboarding_checklist"}


def run(coro):
    return asyncio.run(coro)


def test_mcp_tool_discovery_and_calls():
    async def go():
        m = MCPClientManager(server_url="")
        await m.connect()
        try:
            assert m.connected, m.last_error
            names = {t["name"] for t in m.tool_schemas()}
            assert EXPECTED <= names and len(names) >= 5
            schema = next(t for t in m.tool_schemas() if t["name"] == "check_policy_compliance")["input_schema"]
            assert {"employee_id", "request_type"} <= set(schema["properties"])

            res = await m.call_tool("search_policy_documents", {"query": "PTO carryover", "k": 3})
            assert res["hits"] and res["hits"][0]["citation"].startswith("POL-")

            bal = await m.call_tool("check_pto_balance", {"employee_id": "E1002"})
            assert bal["current_balance"] == 4.5

            missing = await m.call_tool("lookup_employee_profile", {"employee_id": "E9999"})
            assert missing["error"] == "employee_not_found"

            private = await m.call_tool("lookup_employee_profile", {"employee_id": "E1002", "requester_id": "E1001"})
            assert private["restricted"] is True and "performance_rating" not in private

            preview = await m.call_tool("create_mock_hr_ticket", {"employee_id": "E1002", "category": "test", "summary": "x"})
            assert preview["status"] == "confirmation_required" and preview["created"] is False
            assert await m.ping()
        finally:
            await m.close()

    run(go())


def test_disabled_tool_and_unreachable_server_fail_gracefully():
    async def go():
        m = MCPClientManager(server_url="", disabled_tools={"check_pto_balance"})
        await m.connect()
        try:
            out = await m.call_tool("check_pto_balance", {"employee_id": "E1001"})
            assert out["error"] == "tool_unavailable"
        finally:
            await m.close()
        bad = MCPClientManager(server_url="http://127.0.0.1:9/mcp")
        await bad.connect()
        assert not bad.connected
        out = await bad.call_tool("search_policy_documents", {"query": "pto"})
        assert out["error"] == "mcp_unavailable"

    run(go())
