"""Lumora HR MCP server.

Exposes policy-RAG tools, synthetic HR data tools and *mock* action tools over the
Model Context Protocol. Default transport is stdio (spawned as a child process by the
web app's MCP client). Use `--transport streamable-http` to run it as a separate service.

    python -m mcp_server.server                              # stdio
    python -m mcp_server.server --transport streamable-http  # http://0.0.0.0:8765/mcp
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import compliance, data

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s mcp_server: %(message)s")
log = logging.getLogger("mcp_server")
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

mcp = FastMCP(
    "lumora-hr",
    instructions=(
        "Tools for the Lumora Systems HR assistant (synthetic data). Use search_policy_documents for policy "
        "evidence and cite doc_id + section. Action tools are MOCK and require explicit confirmation."
    ),
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8765")),
)

_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        from rag.retriever import Retriever

        _retriever = Retriever()
    return _retriever


def _employee_or_error(employee_id: str | None) -> tuple[dict | None, dict | None]:
    eid = data.normalize_employee_id(employee_id)
    if not eid:
        return None, {"error": "missing_employee_id", "message": "An employee_id (e.g. E1001) is required."}
    emp = data.employees().get(eid)
    if not emp:
        return None, {"error": "employee_not_found", "message": f"No employee with id {eid} in the synthetic directory."}
    return emp, None


def _can_view(requester_id: str | None, emp: dict) -> bool:
    """Privacy guardrail: full records only for self or the employee's direct manager."""
    rid = data.normalize_employee_id(requester_id)
    return rid is None or rid == emp["employee_id"] or rid == emp.get("manager_id")


# =============================================================================== RAG tools
@mcp.tool()
def search_policy_documents(query: str, k: int = 5, doc_ids: list[str] | None = None) -> dict[str, Any]:
    """Semantic + keyword (hybrid) search over the Lumora HR policy corpus.

    Args:
        query: natural-language question or keywords, e.g. "notice required for 3 days of PTO".
        k: number of passages to return (1-10).
        doc_ids: optional filter, e.g. ["POL-002", "POL-005"].
    Returns passages with doc_id, title, section_id, citation, snippet and text.
    """
    k = max(1, min(int(k), 10))
    res = retriever().search(query, k=k, doc_ids=doc_ids)
    for h in res["hits"]:
        h["text"] = h["text"][:1200]
    return res


@mcp.tool()
def get_policy_section(doc_id: str, section_id: str | None = None) -> dict[str, Any]:
    """Return the full text of a policy section (and its subsections), e.g. doc_id="POL-005", section_id="7".
    Omit section_id to list the document's sections."""
    r = retriever()
    chunks = r.get_section(doc_id, section_id)
    if not chunks:
        docs = [d["doc_id"] for d in r.list_documents()]
        return {"error": "not_found", "message": f"No section {section_id} in {doc_id}.", "available_documents": docs}
    if section_id is None:
        return {"doc_id": chunks[0]["doc_id"], "title": chunks[0]["title"],
                "sections": sorted({f"{c['section_id']} {c['section_title']}" for c in chunks}, key=lambda s: [int(x) if x.isdigit() else 0 for x in s.split()[0].split(".")])}
    return {"doc_id": chunks[0]["doc_id"], "title": chunks[0]["title"], "section_id": section_id,
            "citation": f"{chunks[0]['doc_id']} §{section_id}",
            "parts": [{"section_id": c["section_id"], "section_title": c["section_title"], "text": c["text"]} for c in chunks]}


@mcp.tool()
def list_policy_documents() -> dict[str, Any]:
    """List all indexed policy documents with their IDs, titles, source formats and sections."""
    r = retriever()
    return {"documents": r.list_documents(), "index": r.manifest}


# =============================================================================== data tools
@mcp.tool()
def lookup_employee_profile(employee_id: str, requester_id: str | None = None) -> dict[str, Any]:
    """Look up a (synthetic) employee profile: title, department, manager, location, employment type,
    work arrangement, tenure, performance rating, PIP status and data access level.
    requester_id enforces privacy: only the employee or their manager receive the full record."""
    emp, err = _employee_or_error(employee_id)
    if err:
        return err
    mgr = data.employees().get(emp.get("manager_id") or "", {})
    directory = {"employee_id": emp["employee_id"], "name": emp["name"], "title": emp["title"],
                 "department": emp["department"], "office": emp["office"],
                 "manager": {"employee_id": mgr.get("employee_id"), "name": mgr.get("name"), "email": mgr.get("email")}}
    if not _can_view(requester_id, emp):
        return {**directory, "restricted": True,
                "message": "Only directory information is shown. Personal HR data is limited to the employee and their manager (POL-012 §6.2)."}
    office = data.offices().get(emp["office"], {})
    t = data.tenure_days(emp)
    return {**emp, "manager": directory["manager"], "tenure_days": t, "tenure_years": round(t / 365.25, 2),
            "holiday_calendar": data.holiday_calendar_for(emp), "office_timezone": office.get("timezone"),
            "introductory_period_complete": t >= 90, "synthetic": True}


@mcp.tool()
def check_pto_balance(employee_id: str) -> dict[str, Any]:
    """Return the employee's current PTO balance, annual entitlement, accrual, carryover, sick days,
    floating holidays, upcoming approved PTO and pending requests (synthetic, in days)."""
    emp, err = _employee_or_error(employee_id)
    if err:
        return err
    bal = data.pto().get(emp["employee_id"])
    if not bal:
        return {"error": "no_pto_record", "message": f"No PTO record for {emp['employee_id']}."}
    return {**bal, "name": emp["name"], "employment_type": emp["employment_type"],
            "policy_refs": ["POL-002 §3", "POL-002 §5", "POL-002 §9", "POL-003 §6"], "synthetic": True}


@mcp.tool()
def lookup_benefits_status(employee_id: str) -> dict[str, Any]:
    """Return the employee's current benefits elections and enrollment status (synthetic), plus open enrollment dates."""
    emp, err = _employee_or_error(employee_id)
    if err:
        return err
    b = data.benefits()
    rec = next((e for e in b["elections"] if e["employee_id"] == emp["employee_id"]), None)
    if not rec:
        return {"error": "no_benefits_record", "employee_id": emp["employee_id"],
                "message": "No benefits record found; escalate to the Benefits team."}
    return {**rec, "name": emp["name"], "employment_type": emp["employment_type"], "fte": emp["fte"],
            "open_enrollment": b["open_enrollment"], "policy_refs": ["POL-008 §2", "POL-008 §12"], "synthetic": True}


@mcp.tool()
def check_policy_compliance(employee_id: str, request_type: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check a proposed request against encoded Lumora policy rules for a specific employee.

    request_type: one of
      - "pto_request"            details: {start_date: YYYY-MM-DD, end_date?: YYYY-MM-DD, num_days?: number, request_date?: YYYY-MM-DD}
      - "temporary_remote_work"  details: {destination_country, destination_state?, workdays? | duration_weeks?, start_date?}
      - "expense"                details: {category or description (e.g. "ergonomic chair", "laptop"), amount?, city?}
      - "benefits_eligibility"   details: {benefit: medical|dental|vision|401k|hsa|life|learning_stipend|wellness_stipend|eap|parental_leave|home_office_stipend}
    Returns a verdict (compliant | allowed_with_approvals | not_compliant | needs_clarification), each check with a
    policy citation, required approvals and alternatives."""
    emp, err = _employee_or_error(employee_id)
    if err:
        return err
    return compliance.run(emp, request_type, details or {})


# =============================================================================== mock action tools
@mcp.tool()
def create_mock_hr_ticket(employee_id: str, category: str, summary: str, severity: int = 3,
                          policy_refs: list[str] | None = None, confirm: bool = False) -> dict[str, Any]:
    """Create a MOCK HR case (writes only to a local demo file; no real system is touched).

    Safety: with confirm=false (default) this returns a preview and creates nothing. Only call with
    confirm=true after the user has explicitly confirmed. Severity 1 = harassment/discrimination/
    retaliation/safety (routes to Employee Relations), 2 = time-sensitive, 3 = standard (POL-012 §4)."""
    emp, err = _employee_or_error(employee_id)
    if err:
        return err
    severity = int(severity) if str(severity).isdigit() else 3
    routed = {1: "Employee Relations", 2: "Specialist team", 3: "People Operations"}.get(severity, "People Operations")
    ticket = {"employee_id": emp["employee_id"], "category": category, "severity": severity, "summary": summary[:500],
              "policy_refs": policy_refs or [], "status": "open", "routed_to": routed, "mock": True}
    if not confirm:
        return {"status": "confirmation_required", "created": False, "preview": ticket,
                "message": "No ticket created. Ask the user to confirm, then call again with confirm=true."}
    saved = data.append_ticket(ticket)
    return {"status": "created", "created": True, "ticket": saved,
            "message": f"Mock ticket {saved['ticket_id']} created (demo only; no real HR system was updated)."}


@mcp.tool()
def draft_hr_email(employee_id: str, recipient: str, purpose: str, key_points: list[str] | None = None) -> dict[str, Any]:
    """Draft (but never send) an email from the employee to their manager, HR or another team.
    recipient: "manager" | "hr" | "benefits" | "global_mobility" | "employee_relations" | an email address."""
    emp, err = _employee_or_error(employee_id)
    if err:
        return err
    mgr = data.employees().get(emp.get("manager_id") or "", {})
    addresses = {"manager": (mgr.get("email"), mgr.get("name", "Manager").split()[0]),
                 "hr": ("people@lumora.example", "People Team"), "benefits": ("benefits@lumora.example", "Benefits team"),
                 "global_mobility": ("mobility@lumora.example", "Global Mobility team"),
                 "employee_relations": ("er@lumora.example", "Employee Relations team")}
    to, greet = addresses.get(recipient.lower(), (recipient, "team"))
    points = "\n".join(f"- {p}" for p in (key_points or []))
    body = (f"Hi {greet},\n\nI'd like to {purpose.rstrip('.')}.\n\n" + (f"Details:\n{points}\n\n" if points else "")
            + "Please let me know if you need anything else from me.\n\nThanks,\n" + emp["name"].split()[0])
    return {"status": "draft_not_sent", "from": emp["email"], "to": to,
            "subject": purpose[:1].upper() + purpose[1:80].rstrip("."), "body": body,
            "note": "Draft only. The assistant never sends email; review and send it yourself."}


@mcp.tool()
def generate_onboarding_checklist(employee_id: str) -> dict[str, Any]:
    """Generate a personalized onboarding checklist with due dates for a (new) employee, based on POL-010,
    their start date, employment type, country and work arrangement."""
    emp, err = _employee_or_error(employee_id)
    if err:
        return err
    start = date.fromisoformat(emp["hire_date"])
    cal = data.holiday_calendar_for(emp)

    def bd(n: int) -> str:  # n-th business day after start
        return data.add_workdays(start + timedelta(days=1), n, cal).isoformat()

    items = [
        {"task": "Attend New Hire Orientation; activate SSO + MFA + password manager", "due": start.isoformat(), "ref": "POL-010 §4"},
        {"task": "Acknowledge Code of Conduct and Information Security Policy in PeopleHub", "due": start.isoformat(), "ref": "POL-010 §4"},
        {"task": "Security Awareness training", "due": (start + timedelta(days=7)).isoformat(), "ref": "POL-010 §5"},
        {"task": "Payroll setup (direct deposit, tax forms)", "due": bd(5), "ref": "POL-010 §5"},
        {"task": "Respectful Workplace training", "due": (start + timedelta(days=30)).isoformat(), "ref": "POL-010 §6"},
        {"task": "Conflict of Interest disclosure", "due": (start + timedelta(days=30)).isoformat(), "ref": "POL-010 §6"},
        {"task": "30-day check-in with manager", "due": (start + timedelta(days=30)).isoformat(), "ref": "POL-010 §6"},
        {"task": "90-day introductory period review", "due": (start + timedelta(days=90)).isoformat(), "ref": "POL-010 §7"},
    ]
    if emp["country"] == "US":
        items.insert(1, {"task": "Form I-9 Section 1 (employee) on or before day one; Section 2 by People Ops", "due": bd(3), "ref": "POL-010 §5"})
    else:
        items.insert(1, {"task": "Right-to-work check", "due": bd(3), "ref": "POL-010 §5"})
    if emp["employment_type"] in ("full_time", "part_time"):
        items.append({"task": "Benefits enrollment in BenefitsHub", "due": (start + timedelta(days=30)).isoformat(), "ref": "POL-008 §3"})
        if emp["country"] == "US":
            items.append({"task": "401(k) election or opt-out (auto-enroll 4%)", "due": (start + timedelta(days=30)).isoformat(), "ref": "POL-008 §7"})
    if emp["data_access_level"] == "Restricted":
        items.append({"task": "Data Privacy training (Restricted access)", "due": (start + timedelta(days=30)).isoformat(), "ref": "POL-010 §6"})
    if emp["work_arrangement"] in ("remote", "hybrid") and emp["employment_type"] != "intern":
        cap = 500 if emp["work_arrangement"] == "remote" else 250
        items.append({"task": f"Optional: claim home office stipend (up to ${cap})", "due": (start + timedelta(days=90)).isoformat(), "ref": "POL-011 §5"})
    return {"employee_id": emp["employee_id"], "name": emp["name"], "start_date": start.isoformat(),
            "work_arrangement": emp["work_arrangement"], "checklist": sorted(items, key=lambda i: i["due"]),
            "note": "Guidance only; official tracking happens in PeopleHub (POL-010 §11)."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default=os.environ.get("MCP_TRANSPORT", "stdio"), choices=["stdio", "streamable-http", "sse"])
    args = ap.parse_args()
    try:
        retriever()  # warm the index so the first tool call is fast
        log.info("RAG index loaded: %s", retriever().manifest.get("embedding"))
    except Exception as exc:  # the server still starts; RAG tools will return errors
        log.error("RAG index failed to load: %s", exc)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
