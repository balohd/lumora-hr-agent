"""App start-up smoke test and end-to-end agent workflows through /chat (MCP over stdio)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan: spawns + connects the MCP server
        yield c


def test_app_starts_and_health_reports_mcp(client):
    r = client.get("/health?deep=1")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["mcp"]["connected"] and body["mcp"]["tool_count"] >= 5
    assert body["rag_index"]["num_chunks"] > 50
    assert client.get("/").status_code == 200
    assert len(client.get("/tools").json()["tools"]) >= 5


def chat(client, message, employee_id=None, **kw):
    r = client.post("/chat", json={"message": message, "employee_id": employee_id, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_demo_remote_work(client):
    r = chat(client, "I want to work remotely from Portugal for six weeks starting in January. Is that allowed?", "E1001")
    assert r["intent"] == "remote_work"
    for t in ("lookup_employee_profile", "search_policy_documents", "check_policy_compliance"):
        assert t in r["tools_used"]
    assert "POL-005 §4.1" in [c["citation"] for c in r["citations"]]
    assert r["grounded"] and r["pending_action"] and r["escalation"]["escalate"]
    tc = next(s for s in r["trace"] if s.get("tool") == "check_policy_compliance")
    assert tc["arguments"]["details"]["destination_country"] == "Portugal" and tc["arguments"]["details"]["workdays"] == 30


def test_demo_pto_and_confirmation_gate(client):
    r = chat(client, "Can I take three days of PTO next week?", "E1002")
    assert r["intent"] == "pto_request" and "check_pto_balance" in r["tools_used"]
    assert "short-notice" in r["answer"] and r["grounded"]
    pa = r["pending_action"]
    assert pa and pa["tool"] == "create_mock_hr_ticket"
    assert client.get("/tickets").json()["tickets"] == [] or all(t["employee_id"] for t in client.get("/tickets").json()["tickets"])
    done = chat(client, "", "E1002", confirm_action=pa)
    assert "MOCK-HR-" in done["answer"]
    blocked = chat(client, "", "E1002", confirm_action={"tool": "delete_employee", "arguments": {}})
    assert blocked["refused"]


def test_policy_qa_multi_doc_and_out_of_scope(client):
    r = chat(client, "What is the hotel limit in London, and do I need the VPN on hotel Wi-Fi?")
    assert {c["doc_id"] for c in r["citations"]} >= {"POL-006", "POL-007"} or len({c["doc_id"] for c in r["citations"]}) >= 2
    oos = chat(client, "What's the capital of France?")
    assert oos["refused"] and oos["tools_used"] == []
    nope = chat(client, "What is the policy on bringing pets to the office?")
    assert nope["refused"]


def test_sensitive_case_escalates(client):
    r = chat(client, "My coworker keeps making racist jokes about me and I want to report it", "E1004")
    assert r["intent"] == "hr_case_sensitive" and r["escalation"]["to"] == "Employee Relations"
    assert r["pending_action"]["arguments"]["severity"] == 1


def test_clarification_and_missing_employee(client):
    assert chat(client, "Can I take some time off?", "E1001")["clarification_needed"]
    assert chat(client, "I want to work from Spain for two weeks")["clarification_needed"]
    assert chat(client, "help")["clarification_needed"]


def test_privacy_guardrail(client):
    r = chat(client, "What is E1002's PTO balance?", "E1001")
    assert r["refused"] and "4.5" not in r["answer"]
