"""LLM code paths with a stub LLM (no network): grounded synthesis, citation validation and ReAct mode guardrails."""
import asyncio
import json
from types import SimpleNamespace

from agent.mcp_client import MCPClientManager
from agent.orchestrator import Agent


class StubLLM:
    model = "stub"

    def __init__(self, script):
        self.script = list(script)
        self.available = True

    def describe(self):
        return {"provider": "stub", "model": "stub"}

    def chat(self, messages, tools=None, max_tokens=900, json_mode=False):
        return self.script.pop(0) if self.script else SimpleNamespace(content="done", tool_calls=None)


def call(name, args, i=0):
    return SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def with_agent(script, fn):
    async def go():
        m = MCPClientManager(server_url="")
        await m.connect()
        try:
            return await fn(Agent(m, StubLLM(script)))
        finally:
            await m.close()
    return asyncio.run(go())


def test_llm_synthesis_and_invented_citation_is_flagged():
    text = "**Short answer:** You can carry over 5 days [POL-002 §5]. Also see [POL-099 §1]."
    r = with_agent([SimpleNamespace(content=text, tool_calls=None)],
                   lambda a: a.run("How many PTO days can I carry over?", None))
    assert r["synthesis"] == "llm"
    assert r["unsupported_citations"] == ["POL-099 §1"] and r["grounded"] is False


def test_react_mode_forces_confirmation_and_privacy():
    script = [
        SimpleNamespace(content="", tool_calls=[call("check_pto_balance", {"employee_id": "E1002"}, 1)]),
        SimpleNamespace(content="", tool_calls=[call("create_mock_hr_ticket", {"employee_id": "E1001", "category": "x", "summary": "y", "confirm": True}, 2)]),
        SimpleNamespace(content="**Short answer:** You have 12.83 days [POL-002 §3].", tool_calls=None),
    ]
    r = with_agent(script, lambda a: a.run("How much PTO do I have? open a ticket", "E1001", mode="react"))
    assert r["mode"] == "react"
    bal = next(s for s in r["trace"] if s.get("tool") == "check_pto_balance")
    assert bal["arguments"]["employee_id"] == "E1001"  # overridden to the acting employee
    tk = next(s for s in r["trace"] if s.get("tool") == "create_mock_hr_ticket")
    assert tk["arguments"]["confirm"] is False and r["pending_action"]
