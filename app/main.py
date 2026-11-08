"""FastAPI web application: chat UI, /chat API, /health, tool discovery and demo tasks.

Run locally:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.llm import LLM
from agent.mcp_client import MCPClientManager
from agent.orchestrator import Agent

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"
STARTED = time.time()

DEMO_TASKS = [
    {"id": "remote-work-portugal", "title": "Demo 1 — Remote work eligibility (Portugal, six weeks)", "employee_id": "E1001",
     "message": "I want to work remotely from Portugal for six weeks starting in January. Am I allowed to do that, and what do I need to do?",
     "expected_tools": ["lookup_employee_profile", "search_policy_documents", "check_policy_compliance", "get_policy_section",
                        "check_pto_balance", "draft_hr_email", "create_mock_hr_ticket"]},
    {"id": "pto-next-week", "title": "Demo 2 — PTO request guidance (3 days next week)", "employee_id": "E1002",
     "message": "Can I take three days of PTO next week?",
     "expected_tools": ["lookup_employee_profile", "check_pto_balance", "search_policy_documents", "check_policy_compliance",
                        "draft_hr_email", "create_mock_hr_ticket"]},
    {"id": "expense-chair", "title": "Extra — Expense compliance (home office chair)", "employee_id": "E1001",
     "message": "Can I get reimbursed for a $450 ergonomic home office chair?",
     "expected_tools": ["lookup_employee_profile", "search_policy_documents", "check_policy_compliance"]},
    {"id": "benefits-part-time", "title": "Extra — Benefits triage (part-time medical)", "employee_id": "E1003",
     "message": "Am I eligible for medical insurance?",
     "expected_tools": ["lookup_employee_profile", "lookup_benefits_status", "search_policy_documents", "check_policy_compliance"]},
    {"id": "hr-case", "title": "Extra — Sensitive HR case triage", "employee_id": "E1004",
     "message": "My manager keeps making sexual comments about me in meetings and I don't know what to do.",
     "expected_tools": ["search_policy_documents", "get_policy_section", "create_mock_hr_ticket"]},
    {"id": "multi-doc", "title": "Extra — Multi-document policy question", "employee_id": "",
     "message": "I'm traveling to London for a client meeting: what's the hotel limit, and do I need the VPN on hotel Wi-Fi?",
     "expected_tools": ["search_policy_documents"]},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    mcp = MCPClientManager()
    await mcp.connect()
    app.state.mcp = mcp
    app.state.agent = Agent(mcp, LLM())
    log.info("startup complete in %.1fs (mcp connected=%s, llm=%s)", time.time() - STARTED, mcp.connected, app.state.agent.llm.describe())
    yield
    await mcp.close()


app = FastAPI(title="Lumora HR Agent", version="1.0.0", lifespan=lifespan,
              description="Agentic RAG + MCP HR assistant for a fictional company (synthetic data).")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatRequest(BaseModel):
    message: str = Field("", max_length=2000, description="User question or request")
    employee_id: str | None = Field(None, description="Acting (synthetic) employee, e.g. E1001")
    confirm_action: dict | None = Field(None, description="A pending_action returned earlier, sent back only after explicit user confirmation")
    mode: str | None = Field(None, description="planner (default) | react (LLM tool-calling; requires an LLM key)")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC / "index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip() and not req.confirm_action:
        raise HTTPException(400, "message is required")
    return await app.state.agent.run(req.message, req.employee_id, confirm_action=req.confirm_action, mode=req.mode)


@app.get("/health")
async def health(deep: bool = False):
    mcp: MCPClientManager = app.state.mcp
    alive = await mcp.ping() if mcp.connected else False
    if not alive:
        await mcp.connect()
        alive = mcp.connected and await mcp.ping()
    body = {"status": "ok" if alive else "degraded", "app": "lumora-hr-agent", "version": app.version,
            "uptime_s": round(time.time() - STARTED, 1), "mcp": {**mcp.status(), "ping": alive},
            "llm": app.state.agent.llm.describe(), "agent_mode": app.state.agent.mode}
    if deep and alive:
        docs = await mcp.call_tool("list_policy_documents", {})
        body["rag_index"] = docs.get("index") if "error" not in docs else docs
    return JSONResponse(body, status_code=200 if alive else 503)


@app.get("/tools")
async def tools():
    """MCP tool discovery result (tools/list), as seen by the agent's MCP client."""
    mcp: MCPClientManager = app.state.mcp
    return {"transport": mcp.transport, "tools": mcp.tool_schemas()}


@app.get("/demo-tasks")
async def demo_tasks():
    return {"tasks": DEMO_TASKS}


@app.get("/employees")
async def employees():
    data = json.loads((ROOT / "mock_data" / "employees.json").read_text(encoding="utf-8"))
    return {"note": "Synthetic employees", "employees": [
        {"employee_id": e["employee_id"], "name": e["name"], "title": e["title"], "office": e["office"],
         "employment_type": e["employment_type"]} for e in data["employees"]]}


@app.get("/tickets")
async def tickets():
    """Mock HR tickets created during this deployment's lifetime (demo only)."""
    path = ROOT / "data" / "runtime_tickets.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tickets": []}
