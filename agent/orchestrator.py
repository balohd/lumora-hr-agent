"""Agent orchestrator.

Architecture (see design-and-evaluation.md):
  1. Guardrail + intent layer (agent.nlu): detects sensitive cases, out-of-scope requests, privacy issues,
     classifies intent and extracts slots (dates, destination, duration, amount, benefit).
  2. Router decides whether RAG alone is sufficient (general policy Q&A) or a multi-step workflow is needed.
  3. Workflow planner executes a sequence of MCP tool calls (every call goes through the MCP client,
     never direct function calls) and records an operational trace.
  4. Synthesis: an LLM (if configured) rewrites the verified answer plan grounded in retrieved evidence;
     otherwise a deterministic template renders it. Citations are validated against retrieved evidence.
  5. Safety: action tools are mock-only and gated behind explicit user confirmation.

AGENT_MODE=planner (default) runs the deterministic workflow planner. AGENT_MODE=react lets the LLM choose
MCP tools via function calling (guardrails, confirmation gating and citation validation still apply).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date
from typing import Any

from rag.textutil import REF_RE, best_snippet, parse_refs, snippet_score, tokenize

from . import nlu, prompts
from .llm import LLM
from .mcp_client import MCPClientManager

log = logging.getLogger("agent.orchestrator")
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "5"))
CONFIRMABLE_TOOLS = {"create_mock_hr_ticket"}
CATALOG = {d["doc_id"]: d["title"] for d in json.loads(
    (Path(__file__).resolve().parents[1] / "corpus" / "catalog.json").read_text(encoding="utf-8"))["documents"]}
COVERAGE_MIN = float(os.environ.get("EVIDENCE_COVERAGE_MIN", "0.5"))
INTENT_QUERIES = {"pto_request": ("PTO request advance notice business days manager approval PeopleHub", ["POL-002"]),
                  "remote_work": ("temporary remote work from another state or country allowance approvals", ["POL-004", "POL-005"]),
                  "onboarding": ("onboarding first day first week first 30 days requirements", ["POL-010"])}
STATUS_ICON = {"pass": "✅", "fail": "❌", "needs_approval": "🟡", "warning": "⚠️", "info": "ℹ️"}


def today() -> date:
    v = os.environ.get("HR_TODAY")
    return date.fromisoformat(v) if v else date.today()


# ============================================================================ trace
class Trace:
    def __init__(self):
        self.steps: list[dict] = []
        self.raw_outputs: list[dict] = []
        self.t0 = time.perf_counter()

    def add(self, kind: str, **kw) -> dict:
        step = {"step": len(self.steps) + 1, "type": kind, **kw}
        self.steps.append(step)
        return step


def _summarize(tool: str, out: dict) -> str:
    if "error" in out:
        return f"ERROR {out['error']}: {out.get('message', '')}"[:200]
    if tool == "search_policy_documents":
        return f"{len(out.get('hits', []))} passages: " + ", ".join(h["citation"] for h in out.get("hits", []))
    if tool == "get_policy_section":
        return f"{out.get('citation') or out.get('doc_id')}: {len(out.get('parts', out.get('sections', [])))} part(s)"
    if tool == "lookup_employee_profile":
        return f"{out.get('name')} | {out.get('title')} | {out.get('employment_type', '')} | {out.get('work_arrangement', '')} | {out.get('office')}" + (" | RESTRICTED VIEW" if out.get("restricted") else "")
    if tool == "check_pto_balance":
        return f"balance {out.get('current_balance')} days (entitlement {out.get('annual_entitlement_days')}/yr)"
    if tool == "lookup_benefits_status":
        med = (out.get("medical") or {}).get("plan") if isinstance(out.get("medical"), dict) else None
        return f"status {out.get('enrollment_status')}; medical: {med}"
    if tool == "check_policy_compliance":
        fails = [c["check"] for c in out.get("checks", []) if c["status"] == "fail"]
        return f"verdict {out.get('verdict')}" + (f"; failed: {', '.join(fails)}" if fails else "")
    if tool == "create_mock_hr_ticket":
        return f"{out.get('status')}" + (f" {out['ticket']['ticket_id']}" if out.get("ticket") else "")
    if tool == "draft_hr_email":
        return f"draft to {out.get('to')} (not sent)"
    if tool == "generate_onboarding_checklist":
        return f"{len(out.get('checklist', []))} checklist items"
    return json.dumps(out)[:160]


def _truncate(obj: Any, max_str: int = 300, max_items: int = 12, depth: int = 0) -> Any:
    """Readable, size-bounded copy of a tool output for the trace (long strings/lists are shortened)."""
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + "…"
    if isinstance(obj, list):
        out = [_truncate(v, max_str, max_items, depth + 1) for v in obj[:max_items]]
        return out + ([f"… {len(obj) - max_items} more"] if len(obj) > max_items else [])
    if isinstance(obj, dict):
        return {k: _truncate(v, max_str if k != "text" else 160, max_items, depth + 1) for k, v in obj.items()}
    return obj


# ============================================================================ answer plan
@dataclass
class Plan:
    intent: str
    short_answer: str = ""
    policy_points: list[str] = field(default_factory=list)
    situation: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    extra_md: str = ""
    escalation: dict | None = None
    pending_action: dict | None = None
    clarification: str | None = None
    refused: bool = False
    use_llm: bool = True

    def render(self) -> str:
        parts = []
        if self.short_answer:
            parts.append(f"**Short answer:** {self.short_answer}")
        if self.policy_points:
            parts.append("**What the policy says**\n" + "\n".join(f"- {p}" for p in self.policy_points))
        if self.situation:
            parts.append("**Your situation**\n" + "\n".join(f"- {s}" for s in self.situation))
        if self.extra_md:
            parts.append(self.extra_md)
        if self.recommendations:
            parts.append("**Recommended next steps** _(recommendations, not policy)_\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(self.recommendations, 1)))
        if self.approvals:
            parts.append("**Approvals needed:** " + "; ".join(self.approvals))
        if self.clarification:
            parts.append(f"**I need a bit more information:** {self.clarification}")
        return "\n\n".join(parts)


class Agent:
    def __init__(self, mcp: MCPClientManager, llm: LLM | None = None, mode: str | None = None):
        self.mcp = mcp
        self.llm = llm or LLM()
        self.mode = (mode or os.environ.get("AGENT_MODE", "planner")).lower()
        self.k = RETRIEVAL_K

    # ---------------------------------------------------------------- tool wrapper
    async def tool(self, trace: Trace, name: str, args: dict, why: str = "") -> dict:
        t = time.perf_counter()
        out = await self.mcp.call_tool(name, args)
        trace.raw_outputs.append(out)
        trace.add("tool_call", tool=name, arguments=args, reason=why, status="error" if "error" in out else "ok",
                  result_summary=_summarize(name, out), output=_truncate(out), latency_ms=round((time.perf_counter() - t) * 1000))
        return out

    # ---------------------------------------------------------------- entry point
    async def run(self, message: str, employee_id: str | None = None, confirm_action: dict | None = None,
                  mode: str | None = None) -> dict:
        trace = Trace()
        ctx = {"message": message or "", "employee_id": (employee_id or "").strip().upper() or None,
               "hits": {}, "sections": {}, "tool_outputs": {}}
        mode = (mode or self.mode).lower()
        try:
            if confirm_action:
                plan = await self._confirm(ctx, trace, confirm_action)
                parsed = nlu.Parsed("confirm_action", 1.0)
            else:
                parsed = nlu.classify(ctx["message"], today())
                trace.add("intent", intent=parsed.intent, confidence=parsed.confidence, slots=parsed.slots,
                          flags=parsed.flags, sub_questions=parsed.sub_questions)
                if mode == "react" and self.llm.available and parsed.intent not in ("hr_case_sensitive", "out_of_scope"):
                    return await self._react(ctx, trace, parsed)
                plan = await self._dispatch(ctx, trace, parsed)
        except Exception as exc:  # last-resort graceful failure
            log.exception("agent failure")
            trace.add("error", message=f"{type(exc).__name__}: {exc}")
            parsed = nlu.Parsed("error", 0.0)
            plan = Plan("error", short_answer="Sorry, something went wrong while processing this request. Please try again, "
                        "or contact the People Team at people@lumora.example.", use_llm=False)
        return await self._finish(ctx, trace, parsed, plan, mode="planner")

    async def _dispatch(self, ctx, trace, parsed: nlu.Parsed) -> Plan:
        intent = parsed.intent
        if nlu.is_vague(ctx["message"]) and not nlu.OFF_TOPIC_RE.search(ctx["message"]) and intent != "hr_case_sensitive":
            trace.add("clarification", reason="request too vague to route", missing=["topic"])
            return Plan("clarify", use_llm=False, clarification=(
                "Could you tell me a bit more about what you need? For example: \"Can I take 3 days of PTO next week?\", "
                "\"Can I work from Spain for two weeks?\", \"Is a home office chair reimbursable?\", or \"Am I eligible for the HSA?\""))
        if intent == "out_of_scope":
            trace.add("guardrail", rule="out_of_scope", decision="refuse_and_redirect", tools_called=False)
            return Plan(intent, refused=True, use_llm=False, short_answer=(
                "That question is outside what I can help with. I'm the Lumora HR assistant: I can answer questions about "
                "Lumora HR policies (PTO, holidays, remote work, expenses, benefits, leave, security, onboarding, conduct) "
                "and help with HR workflows like PTO requests or work-from-abroad checks."))
        if intent == "hr_case_sensitive":
            return await self._wf_sensitive(ctx, trace, parsed)
        personal_intents = {"pto_request", "remote_work", "expense", "benefits", "onboarding"}
        if intent in personal_intents and not ctx["employee_id"]:
            trace.add("clarification", reason="personal workflow requires an employee ID", missing=["employee_id"])
            general = await self._wf_policy_qa(ctx, trace, parsed, query_override=INTENT_QUERIES.get(intent))
            general.clarification = ("To check this against your own records, please select or provide your employee ID "
                                     "(for example E1001). Meanwhile, here is what the policy says.")
            return general
        if intent in personal_intents and parsed.flags.get("mentioned_employee_ids"):
            other = [e for e in parsed.flags["mentioned_employee_ids"] if e != ctx["employee_id"]]
            if other:
                prof = await self.tool(trace, "lookup_employee_profile", {"employee_id": other[0], "requester_id": ctx["employee_id"]},
                                       "user referenced another employee; privacy check")
                if prof.get("restricted"):
                    trace.add("guardrail", rule="privacy", decision="refuse_other_employee_data")
                    return Plan(intent, refused=True, use_llm=False, short_answer=(
                        f"I can't share personal HR information about {prof.get('name', other[0])}. The HR assistant only "
                        "discloses personal data to the employee themselves or their manager [POL-012 §6.2]."),
                        recommendations=["Ask the employee directly, or contact the People Team if you have a legitimate business need."])
                ctx["employee_id"] = other[0]
        handler = {"pto_request": self._wf_pto, "remote_work": self._wf_remote, "expense": self._wf_expense,
                   "benefits": self._wf_benefits, "onboarding": self._wf_onboarding}.get(intent, self._wf_policy_qa)
        trace.add("plan", workflow=handler.__name__.replace("_wf_", ""),
                  rag_only=handler == self._wf_policy_qa, reason="general policy question: retrieval is sufficient"
                  if handler == self._wf_policy_qa else "personal request: needs employee data + policy evidence + compliance check")
        return await handler(ctx, trace, parsed)

    # ---------------------------------------------------------------- evidence helpers
    async def search(self, ctx, trace, query: str, k: int | None = None, doc_ids: list[str] | None = None, why: str = "") -> list[dict]:
        args = {"query": query, "k": k or self.k}
        if doc_ids:
            args["doc_ids"] = doc_ids
        out = await self.tool(trace, "search_policy_documents", args, why or "retrieve policy evidence")
        hits = out.get("hits", []) if "error" not in out else []
        for h in hits:
            ctx["hits"].setdefault(h["citation"], h)
        return hits

    async def fetch_refs(self, ctx, trace, refs: list[tuple[str, str]], query: str, limit: int = 3) -> None:
        """Retrieve exact text of decisive policy sections cited by compliance checks (if not already retrieved)."""
        have = set(ctx["hits"]) | set(ctx["sections"])
        todo = [r for r in refs if f"{r[0]} §{r[1]}" not in have][:limit]
        for doc_id, sec in todo:
            out = await self.tool(trace, "get_policy_section", {"doc_id": doc_id, "section_id": sec},
                                  "fetch exact text of a policy section cited by a compliance rule")
            if "error" in out:
                continue
            text = "\n".join(p["text"] for p in out.get("parts", []))
            ctx["sections"][f"{doc_id} §{sec}"] = {
                "doc_id": doc_id, "title": out.get("title"), "section_id": sec,
                "section_title": out["parts"][0]["section_title"] if out.get("parts") else "",
                "citation": f"{doc_id} §{sec}", "snippet": best_snippet(text, query), "text": text}

    @staticmethod
    def _cite_points(hits: list[dict], query: str, n: int = 3) -> list[str]:
        pts = []
        for h in hits[:n]:
            pts.append(f"{best_snippet(h['text'], query, 300)} [{h['citation']}]")
        return pts

    @staticmethod
    def _checks_md(comp: dict) -> str:
        rows = []
        for c in comp.get("checks", []):
            ref = c.get("ref")
            ref = ", ".join(ref) if isinstance(ref, list) else ref
            rows.append(f"| {STATUS_ICON.get(c['status'], '')} {c['check'].replace('_', ' ')} | {c['detail']} | {ref or ''} |")
        if not rows:
            return ""
        return "**Policy compliance check** (`check_policy_compliance`)\n\n| Check | Result | Policy |\n|---|---|---|\n" + "\n".join(rows)

    # ---------------------------------------------------------------- workflows
    async def _wf_policy_qa(self, ctx, trace, parsed, query_override: tuple | None = None) -> Plan:
        q, doc_filter = query_override if query_override else (ctx["message"], None)
        queries = ([q] if query_override else parsed.sub_questions) or [q]
        per_q: list[tuple[str, list[dict], float]] = []
        for sq in queries:
            args = {"query": sq, "k": self.k if len(queries) == 1 else 3}
            if doc_filter:
                args["doc_ids"] = doc_filter
            out = await self.tool(trace, "search_policy_documents", args,
                                  "sub-question of a multi-part question" if len(queries) > 1 else "retrieve policy evidence")
            if "error" in out:
                return self._error_plan(out)
            hits = out.get("hits", [])
            cov = out.get("evidence_coverage", 0.0) if hits else 0.0
            if cov < COVERAGE_MIN and len(queries) > 1 and sq != queries[0] and sq != queries[1]:
                # reformulation retry: a follow-up sub-question may depend on the topic of the first part
                topic = re.sub(r"^(if|when|how (many|much)|what|which|can|do|does|is|are)\b\s*", "", queries[1], flags=re.I)
                retry = await self.tool(trace, "search_policy_documents", {**args, "query": f"{sq} {topic}"},
                                        "low evidence coverage: retry sub-question with the topic of the first part")
                if "error" not in retry and retry.get("evidence_coverage", 0) > cov:
                    hits, cov = retry.get("hits", []), retry.get("evidence_coverage", 0.0)
            for h in hits:
                ctx["hits"].setdefault(h["citation"], h)
            per_q.append((sq, hits, cov))
        covered = [(sq, hits, cov) for sq, hits, cov in per_q if hits and cov >= COVERAGE_MIN]
        uncovered = [sq for sq, hits, cov in per_q if not hits or cov < COVERAGE_MIN]
        trace.add("evidence_check", coverage={sq[:60]: cov for sq, _, cov in per_q}, threshold=COVERAGE_MIN,
                  decision="sufficient" if covered and not uncovered else ("partial" if covered else "insufficient"))
        if not covered:
            trace.add("guardrail", rule="no_policy_evidence", decision="refuse_and_redirect")
            return Plan("policy_qa", refused=True, use_llm=False, short_answer=(
                "I couldn't find this in Lumora's HR policies, so I can't give you a policy answer. Please contact the "
                "People Team at people@lumora.example or open a PeopleHub case."))
        # evidence selection: top passage per covered (sub-)question, then other passages that are relevant
        # to the full question; passages with weak overlap are dropped to keep the answer focused
        def rel(h, query):
            return len(set(tokenize(query)) & set(tokenize(best_snippet(h["text"], query, 400))))
        chosen: list[tuple[str, dict]] = []
        seen = set()
        for sq, hits, _ in covered:
            # answer-bearing passage: best sentence-level match among the top 3 (small prior for retrieval rank)
            cands = [h for h in hits[:3] if h["chunk_id"] not in seen]
            if cands:
                top = max(cands, key=lambda h: snippet_score(h["text"], sq) - 0.25 * hits.index(h))
                chosen.append((sq, top))
                seen.add(top["chunk_id"])
        best_rel = max(rel(h, q) for _, h in chosen) if chosen else 0
        single = len(queries) == 1
        lead_docs = {h["doc_id"] for _, h in chosen}
        for sq, hits, _ in covered:
            for h in hits[1:3]:
                # single questions: supporting passages must come from the same policy as the answer
                if single and h["doc_id"] not in lead_docs:
                    continue
                if h["chunk_id"] not in seen and rel(h, sq) >= max(2, 0.6 * best_rel):
                    chosen.append((sq, h))
                    seen.add(h["chunk_id"])
        chosen = chosen[: max(3, len(covered) + 1)]
        lead_q, lead = chosen[0]
        plan = Plan("policy_qa", short_answer=f"{best_snippet(lead['text'], lead_q, 260)} [{lead['citation']}]",
                    policy_points=[f"{best_snippet(h['text'], sq, 280)} [{h['citation']}]" for sq, h in chosen[1:]])
        if uncovered:
            plan.policy_points.append("I couldn't find policy evidence for: " + "; ".join(f'"{u}"' for u in uncovered)
                                      + ". Please contact the People Team for that part.")
        docs = sorted({h["doc_id"] for _, h in chosen})
        if len(docs) > 1:
            trace.add("synthesis_note", multi_document=True, documents=docs)
        return plan

    async def _wf_pto(self, ctx, trace, parsed) -> Plan:
        eid, s = ctx["employee_id"], dict(parsed.slots)
        prof = await self.tool(trace, "lookup_employee_profile", {"employee_id": eid, "requester_id": eid}, "who is asking (eligibility, department, manager)")
        if "error" in prof:
            return self._error_plan(prof)
        bal = await self.tool(trace, "check_pto_balance", {"employee_id": eid}, "current PTO balance")
        hits = await self.search(ctx, trace, "PTO request advance notice business days manager approval PeopleHub", k=4, doc_ids=["POL-002"])
        if "error" in bal:
            return self._error_plan(bal)
        plan = Plan("pto_request")
        if not bal.get("eligible", True):
            plan.short_answer = f"{bal.get('ineligible_reason', 'You are not eligible for PTO.')}"
            plan.situation = [f"{prof['name']} — {prof['title']} ({prof['employment_type']})."]
            plan.recommendations = ["Contractors should arrange time away with their agency or according to their contract [POL-002 §2]."]
            return plan
        plan.situation = [f"PTO balance: **{bal['current_balance']} days** (entitlement {bal['annual_entitlement_days']} days/year; "
                          f"used {bal['used_ytd']} this year; {bal.get('floating_holidays_remaining', 0)} floating holiday(s) left).",
                          f"Manager: {prof['manager']['name']}. Department: {prof['department']}."]
        if s.get("balance_only") or not s.get("start_date"):
            plan.policy_points = self._cite_points(hits, "notice approval PeopleHub", 2)
            if s.get("balance_only"):
                plan.short_answer = f"You have {bal['current_balance']} days of PTO available."
            else:
                trace.add("clarification", reason="PTO request without dates", missing=["start_date", "number_of_days"])
                plan.short_answer = f"You currently have {bal['current_balance']} days of PTO."
                plan.clarification = "Which dates (or how many days, starting when) would you like to take? I'll then check notice, balance and approval requirements."
            return plan
        details = {k: s[k] for k in ("start_date", "end_date", "num_days") if k in s}
        comp = await self.tool(trace, "check_policy_compliance", {"employee_id": eid, "request_type": "pto_request", "details": details},
                               "check balance, notice, blackout and approval rules for these dates")
        if "error" in comp:
            return self._error_plan(comp)
        await self.fetch_refs(ctx, trace, [r for c in comp.get("checks", []) if c["status"] in ("fail", "warning", "needs_approval") for r in parse_refs(c.get("ref"))], "notice approval", 2)
        req = comp.get("request", {})
        days = req.get("workdays_requested")
        verdict = comp["verdict"]
        notice = next((c for c in comp["checks"] if c["check"] == "advance_notice"), None)
        balance_chk = next((c for c in comp["checks"] if c["check"] == "balance"), None)
        if verdict == "not_compliant":
            plan.short_answer = f"Not as requested: {balance_chk['detail'] if balance_chk and balance_chk['status'] == 'fail' else 'the request does not meet policy'} [{balance_chk['ref'] if balance_chk else 'POL-002 §8'}]."
        else:
            sn = notice and notice["status"] == "warning"
            plan.short_answer = (f"Yes, you have enough PTO for {days:g} day(s) ({req['start_date']} to {req['end_date']}), but it needs your manager's approval in PeopleHub"
                                 + (", and it is a **short-notice request**: policy requires " + re.search(r"(\d+) required", notice["detail"]).group(1)
                                    + " business days' notice, so approval is at your manager's discretion" if sn and re.search(r"(\d+) required", notice["detail"]) else "")
                                 + " [POL-002 §6.1, POL-002 §6.2].")
        if req.get("company_holidays_in_range"):
            plan.situation.append(f"Company holidays in that range are not deducted from PTO: {', '.join(req['company_holidays_in_range'])} [POL-002 §10].")
        plan.policy_points = self._cite_points(hits, "advance notice consecutive workdays business days manager approval PeopleHub", 2)
        plan.extra_md = self._checks_md(comp)
        plan.approvals = comp.get("required_approvals", [])
        if verdict != "not_compliant":
            email = await self.tool(trace, "draft_hr_email", {
                "employee_id": eid, "recipient": "manager",
                "purpose": f"request PTO from {req['start_date']} to {req['end_date']} ({days:g} day(s))",
                "key_points": [f"Dates: {req['start_date']} to {req['end_date']} ({days:g} workday(s))",
                               f"My PTO balance is {bal['current_balance']} days, leaving {round(bal['current_balance'] - days, 2)} after this request",
                               "I will submit the request in PeopleHub and arrange coverage for my work"]
                + (["I know this is short notice and appreciate your flexibility"] if notice and notice["status"] == "warning" else [])},
                "draft a message to the manager (not sent)")
            if "error" not in email:
                plan.extra_md += f"\n\n**Draft message to your manager** _(not sent)_\n\n> **To:** {email['to']}  \n> **Subject:** {email['subject']}\n>\n" + "\n".join(f"> {ln}" for ln in email["body"].splitlines())
            plan.recommendations = ["Submit the request in PeopleHub (the assistant cannot submit or approve PTO) [POL-002 §13].",
                                    "Send your manager the draft above and add the time to your calendar with an out-of-office message [POL-002 §6.1]."]
            if notice and notice["status"] == "warning":
                plan.recommendations.insert(1, "Because this is short notice, include a brief note explaining the circumstances [POL-002 §6.2].")
            ticket = await self.tool(trace, "create_mock_hr_ticket", {
                "employee_id": eid, "category": "time_off", "severity": 3,
                "summary": f"PTO request {req['start_date']} to {req['end_date']} ({days:g} days); short notice: {bool(notice and notice['status'] == 'warning')}",
                "policy_refs": ["POL-002 §6.1", "POL-002 §6.2"], "confirm": False}, "prepare optional mock HR record (preview only, nothing created)")
            if ticket.get("status") == "confirmation_required":
                plan.pending_action = {"tool": "create_mock_hr_ticket", "arguments": {**{k: v for k, v in ticket["preview"].items() if k in ("employee_id", "category", "severity", "summary", "policy_refs")}},
                                       "label": "Log a mock PTO request record", "preview": ticket["preview"]}
        else:
            plan.recommendations = ["Choose fewer days or later dates, or discuss options with your manager.",
                                    "Contact the People Team if you believe your balance is wrong."]
        return plan

    async def _wf_remote(self, ctx, trace, parsed) -> Plan:
        eid, s = ctx["employee_id"], dict(parsed.slots)
        prof = await self.tool(trace, "lookup_employee_profile", {"employee_id": eid, "requester_id": eid}, "role, tenure, rating, data access, home country")
        if "error" in prof:
            return self._error_plan(prof)
        home = {"US": "United States", "CA": "Canada", "UK": "United Kingdom"}.get(prof["country"], prof["country"])
        dest = s.get("destination_country")
        international = bool(dest) and dest not in (home, "UK" if home == "United Kingdom" else "", "England" if home == "United Kingdom" else "")
        hits = await self.search(ctx, trace, ("temporary remote work from another country allowance 20 workdays approvals notice"
                                              if international or not dest else "temporary remote work another US state 20 workdays payroll tax"),
                                 k=4, doc_ids=["POL-005"] if international else ["POL-004", "POL-005"], why="remote/cross-border work policy")
        if international:
            await self.search(ctx, trace, "security requirements working abroad VPN managed device restricted data", k=2,
                              doc_ids=["POL-007", "POL-005"], why="security requirements while abroad")
        missing = [m for m in ("destination_country", "workdays") if not s.get(m)]
        plan = Plan("remote_work")
        plan.situation = [f"{prof['name']} — {prof['title']}, {prof['department']}; {prof['work_arrangement']} in {prof['office']} ({home}); "
                          f"tenure {prof['tenure_years']} years; rating '{prof['performance_rating']}'; data access '{prof['data_access_level']}'."]
        if missing:
            trace.add("clarification", reason="remote work request missing details", missing=missing)
            plan.policy_points = self._cite_points(hits, "allowance workdays approvals notice", 3)
            plan.short_answer = "It depends on where and for how long."
            plan.clarification = "Where would you like to work from (country, or US state), for how long, and starting when?"
            return plan
        details = {"destination_country": dest, "workdays": s["workdays"]}
        for k in ("destination_state", "start_date", "duration_weeks"):
            if s.get(k):
                details[k] = s[k]
        comp = await self.tool(trace, "check_policy_compliance", {"employee_id": eid, "request_type": "temporary_remote_work", "details": details},
                               "evaluate eligibility, destination, duration, notice and data access rules")
        if "error" in comp:
            return self._error_plan(comp)
        decisive = [r for c in comp.get("checks", []) if c["status"] in ("fail", "needs_approval", "warning") for r in parse_refs(c.get("ref"))]
        await self.fetch_refs(ctx, trace, decisive, "exception allowance workdays approval", 2)
        verdict = comp["verdict"]
        fails = [c for c in comp["checks"] if c["status"] == "fail"]
        dur_fail = any(c["check"] == "duration" for c in fails)
        weeks = s.get("duration_weeks")
        span = f"{weeks:g} weeks (~{s['workdays']} workdays)" if weeks else f"{s['workdays']} workdays"
        where = dest if international else f"{s.get('destination_state', 'another state')}, {dest}"
        if verdict == "not_compliant" and dur_fail and len(fails) == 1:
            plan.short_answer = (f"Not as a standard request. Working from {where} for {span} exceeds the "
                                 f"{comp['request'].get('allowance_workdays', 20)}-workday allowance in any rolling 12 months [POL-005 §4.1]; "
                                 "anything longer needs a Global Mobility exception, which is rarely granted [POL-005 §7]. "
                                 "You meet the other eligibility requirements.")
        elif verdict == "not_compliant":
            plan.short_answer = f"No — this request does not meet policy: " + "; ".join(f"{c['detail']} [{', '.join(c['ref']) if isinstance(c['ref'], list) else c['ref']}]" for c in fails)
        elif verdict == "allowed_with_approvals":
            plan.short_answer = f"Yes, this can be requested: working from {where} for {span} is within policy limits, subject to the approvals below."
        else:
            plan.short_answer = f"Yes — working from {where} for {span} is within policy."
        # keep only passages for the relevant scope (international: POL-005 §4-§7; domestic: §3 / POL-004)
        scoped = [h for h in hits if (not h["citation"].startswith("POL-005 §3")) == international] or hits
        plan.policy_points = self._cite_points(scoped, "allowance workdays approvals notice exception", 3)
        plan.extra_md = self._checks_md(comp)
        plan.approvals = comp.get("required_approvals", [])
        recs = []
        if dur_fail and international:
            pto = await self.tool(trace, "check_pto_balance", {"employee_id": eid}, "can PTO cover the days beyond the allowance?")
            over = s["workdays"] - comp["request"].get("allowance_workdays", 20) + comp["request"].get("workdays_already_used_12m", 0)
            bal = pto.get("current_balance", 0) if "error" not in pto else 0
            plan.situation.append(f"PTO balance: {bal} days; days beyond the allowance: {over}.")
            if bal >= over:
                recs.append(f"Option A (often approved): work up to {s['workdays'] - over} workdays from {dest} and take {over} days of PTO for the rest, so no work is done beyond the allowance [POL-005 §7]. Your PTO balance ({bal} days) covers this.")
            else:
                recs.append(f"Option A: work up to {s['workdays'] - over} workdays abroad and take PTO for the remaining days [POL-005 §7] — note your PTO balance ({bal} days) does not cover all {over} days.")
            recs.append("Option B: request a Global Mobility exception with a written business justification; it needs department head, VP of People and Legal approval and a 4–6 week tax/immigration assessment [POL-005 §7].")
        if international:
            recs.append("Submit the Work From Abroad form in PeopleHub at least 30 calendar days before your first workday abroad; approvals go Manager → Global Mobility → Information Security [POL-005 §4.3].")
            recs.append("While abroad use only your Lumora-managed laptop with the VPN, and don't download Confidential data locally [POL-005 §6.1, POL-005 §6.2].")
            recs.append("Travel and accommodation for temporary remote work are not reimbursed [POL-005 §9]. Confirm your own entry/visa requirements — I can't give immigration or tax advice [POL-005 §8].")
        else:
            recs.append("Request manager approval in PeopleHub at least 10 business days in advance [POL-005 §3.2].")
        plan.recommendations = recs
        email = await self.tool(trace, "draft_hr_email", {
            "employee_id": eid, "recipient": "manager",
            "purpose": f"discuss working temporarily from {where} for {span}",
            "key_points": [f"Proposed location: {where}", f"Duration: {span}"]
            + (["This exceeds the 20-workday allowance; I'd propose working 20 days and using PTO for the rest, or requesting a Global Mobility exception"] if dur_fail else [])
            + ["I'll use my Lumora laptop and VPN and remain available during core hours"]}, "draft a message to the manager (not sent)")
        if "error" not in email:
            plan.extra_md += f"\n\n**Draft message to your manager** _(not sent)_\n\n> **To:** {email['to']}  \n> **Subject:** {email['subject']}\n>\n" + "\n".join(f"> {ln}" for ln in email["body"].splitlines())
        if international and dur_fail:
            ticket = await self.tool(trace, "create_mock_hr_ticket", {
                "employee_id": eid, "category": "work_arrangement", "severity": 2,
                "summary": f"Work From Abroad: {dest}, {span}; exceeds 20-workday allowance; employee requests guidance on Global Mobility exception vs. PTO split",
                "policy_refs": ["POL-005 §4.1", "POL-005 §7"], "confirm": False}, "prepare mock Global Mobility case (preview only, nothing created)")
            if ticket.get("status") == "confirmation_required":
                plan.pending_action = {"tool": "create_mock_hr_ticket", "arguments": {k: v for k, v in ticket["preview"].items() if k in ("employee_id", "category", "severity", "summary", "policy_refs")},
                                       "label": "Open a mock Global Mobility case", "preview": ticket["preview"]}
                plan.escalation = {"escalate": True, "to": "People Team (Global Mobility)", "severity": 2,
                                   "reason": "Request exceeds policy limit; an exception decision is outside the assistant's authority [POL-012 §7]."}
        return plan

    async def _wf_expense(self, ctx, trace, parsed) -> Plan:
        eid, s = ctx["employee_id"], parsed.slots
        prof = await self.tool(trace, "lookup_employee_profile", {"employee_id": eid, "requester_id": eid}, "work arrangement determines stipend eligibility")
        if "error" in prof:
            return self._error_plan(prof)
        hits = await self.search(ctx, trace, s["expense_description"], k=4, doc_ids=["POL-006", "POL-011", "POL-008", "POL-005"], why="expense policy evidence")
        details = {"category": s["expense_category"], "description": s["expense_description"]}
        for k in ("amount", "city"):
            if k in s:
                details[k] = s[k]
        comp = await self.tool(trace, "check_policy_compliance", {"employee_id": eid, "request_type": "expense", "details": details},
                               "apply expense/stipend rules for this employee")
        if "error" in comp:
            return self._error_plan(comp)
        await self.fetch_refs(ctx, trace, [r for c in comp.get("checks", []) for r in parse_refs(c.get("ref"))], s["expense_description"], 2)
        item = s["expense_category"].replace("_", " ")
        v = comp["verdict"]
        first = comp["checks"][0] if comp.get("checks") else {"detail": "", "ref": ""}
        ref = ", ".join(first["ref"]) if isinstance(first.get("ref"), list) else first.get("ref")
        if v == "not_compliant":
            plan = Plan("expense", short_answer=f"No — a {item} is not reimbursable for you: {first['detail']} [{ref}]")
        elif s["expense_category"] in ("office_chair", "desk"):
            plan = Plan("expense", short_answer=f"Not as a regular expense, but yes through the home office stipend: {first['detail']} [{ref}]")
        else:
            plan = Plan("expense", short_answer=f"Yes, within limits: {first['detail']} [{ref}]")
        plan.situation = [f"{prof['name']}: {prof['work_arrangement']} employee, {prof['employment_type'].replace('_', '-')}, {prof['office']}."]
        plan.policy_points = self._cite_points(hits, s["expense_description"], 2)
        plan.extra_md = self._checks_md(comp)
        plan.recommendations = ["Submit claims in ExpenseIQ within 30 days with itemized receipts for $25 or more [POL-006 §4]."] if v != "not_compliant" else \
            (["Request equipment from IT through the IT Service Portal instead [POL-011 §2]."] if s["expense_category"] in ("laptop", "monitor", "phone") else [])
        return plan

    async def _wf_benefits(self, ctx, trace, parsed) -> Plan:
        eid, s = ctx["employee_id"], parsed.slots
        prof = await self.tool(trace, "lookup_employee_profile", {"employee_id": eid, "requester_id": eid}, "employment type/FTE drive eligibility")
        if "error" in prof:
            return self._error_plan(prof)
        ben = await self.tool(trace, "lookup_benefits_status", {"employee_id": eid}, "current elections")
        hits = await self.search(ctx, trace, f"{s['benefit'].replace('_', ' ')} eligibility", k=4, doc_ids=["POL-008", "POL-009", "POL-011"])
        comp = await self.tool(trace, "check_policy_compliance", {"employee_id": eid, "request_type": "benefits_eligibility", "details": {"benefit": s["benefit"]}},
                               "eligibility rule for this benefit")
        if "error" in comp:
            return self._error_plan(comp)
        chk = comp["checks"][0] if comp.get("checks") else {"status": "info", "detail": "", "ref": ""}
        ok = chk["status"] == "pass"
        name = s["benefit"].replace("_", " ")
        plan = Plan("benefits", short_answer=f"{'Yes' if ok else 'No'} — {chk['detail']} [{chk['ref']}]")
        plan.situation = [f"{prof['name']}: {prof['employment_type'].replace('_', '-')} (FTE {prof['fte']}), {prof['country']}."]
        if "error" not in ben:
            med = ben.get("medical") or {}
            plan.situation.append(f"Enrollment status: {ben.get('enrollment_status')}; medical plan: {med.get('plan') if isinstance(med, dict) else med}; "
                                  f"open enrollment for 2027: {ben['open_enrollment']['start']} to {ben['open_enrollment']['end']} [POL-008 §12.1].")
        plan.policy_points = self._cite_points(hits, f"{name} eligibility", 2)
        plan.extra_md = self._checks_md(comp)
        if not ok or (ben.get("enrollment_status") not in ("enrolled",) and "error" not in ben):
            plan.recommendations = ["For questions about your specific situation, contact the Benefits team (benefits@lumora.example) or open a PeopleHub case [POL-008 §16]."]
            plan.escalation = {"escalate": True, "to": "Benefits team", "severity": 3, "reason": "Eligibility edge case; confirm with Benefits [POL-008 §16]."} if not ok else None
        else:
            plan.recommendations = ["Review or change elections during open enrollment or within 30 days of a qualifying life event [POL-008 §12]."]
        return plan

    async def _wf_onboarding(self, ctx, trace, parsed) -> Plan:
        eid = ctx["employee_id"]
        prof = await self.tool(trace, "lookup_employee_profile", {"employee_id": eid, "requester_id": eid}, "start date, type, location")
        if "error" in prof:
            return self._error_plan(prof)
        cl = await self.tool(trace, "generate_onboarding_checklist", {"employee_id": eid}, "personalised checklist with due dates")
        hits = await self.search(ctx, trace, "onboarding first day first week first 30 days requirements", k=3, doc_ids=["POL-010"])
        if "error" in cl:
            return self._error_plan(cl)
        rows = "\n".join(f"| {i['due']} | {i['task']} | {i['ref']} |" for i in cl["checklist"])
        plan = Plan("onboarding", short_answer=f"Here is your personalized onboarding checklist, starting {cl['start_date']} ({cl['work_arrangement']}).",
                    extra_md=f"| Due | Task | Policy |\n|---|---|---|\n{rows}",
                    policy_points=self._cite_points(hits, "first day first week deadlines", 2),
                    recommendations=["Track official completion in PeopleHub; this checklist is guidance only [POL-010 §11]."])
        plan.situation = [f"{prof['name']} — {prof['title']}, manager {prof['manager']['name']}."]
        return plan

    async def _wf_sensitive(self, ctx, trace, parsed) -> Plan:
        trace.add("guardrail", rule="sensitive_case", decision="escalate_severity_1",
                  note="Assistant must not investigate or assess Severity 1 matters (POL-012 §6.2)")
        hits = await self.search(ctx, trace, "how to report harassment discrimination retaliation concern channels Employee Relations",
                                 k=4, doc_ids=["POL-001", "POL-012"], why="reporting channels and escalation rules")
        await self.fetch_refs(ctx, trace, [("POL-012", "6.2"), ("POL-001", "6")], "Severity 1 escalate retaliation", 2)
        self_harm = parsed.flags.get("self_harm")
        plan = Plan("hr_case_sensitive", use_llm=False)
        plan.short_answer = ("I'm sorry you're dealing with this. This is a serious concern that should go to Lumora's Employee Relations "
                             "team, who handle these matters confidentially; I can't investigate or decide on it myself [POL-012 §6.2].")
        if self_harm:
            plan.short_answer = ("I'm really sorry you're going through this. If you are in immediate danger, please call emergency services "
                                 "(911 in the US/Canada, 999 in the UK) now. In the US you can also call or text 988 (Suicide & Crisis Lifeline). "
                                 "Lumora's Employee Assistance Program offers free, confidential counseling 24/7 [POL-008 §9]. " + plan.short_answer)
        plan.policy_points = [
            "You can report to your manager, the People Team (people@lumora.example or a PeopleHub case), Employee Relations (er@lumora.example), "
            "or the 24/7 Lumora Ethics Line, which accepts anonymous reports where permitted [POL-001 §12].",
            "Retaliation against anyone who reports a concern in good faith is strictly prohibited [POL-001 §6].",
            "Severity 1 cases receive an initial response within 4 business hours, with Employee Relations engaged within 24 hours [POL-012 §4].",
            "If anyone is in immediate danger, call emergency services first [POL-001 §7]."]
        plan.recommendations = ["Write down what happened, when, where, who was involved and any witnesses — you don't need proof to report [POL-001 §12].",
                                "If you'd like, I can prepare a confidential mock case for Employee Relations; nothing is created unless you confirm."]
        eid = ctx["employee_id"]
        plan.escalation = {"escalate": True, "to": "Employee Relations", "severity": 1,
                           "reason": "Possible harassment/discrimination/retaliation/safety matter — must be escalated, not resolved by the assistant [POL-012 §6.2, POL-012 §7]."}
        if eid:
            ticket = await self.tool(trace, "create_mock_hr_ticket", {
                "employee_id": eid, "category": "employee_relations", "severity": 1,
                "summary": "Employee reported a sensitive workplace concern via the HR assistant; requests Employee Relations contact. Details withheld from the assistant summary for confidentiality.",
                "policy_refs": ["POL-001 §12", "POL-012 §4", "POL-012 §6.2"], "confirm": False}, "prepare confidential Severity 1 case (preview only)")
            if ticket.get("status") == "confirmation_required":
                plan.pending_action = {"tool": "create_mock_hr_ticket", "arguments": {k: v for k, v in ticket["preview"].items() if k in ("employee_id", "category", "severity", "summary", "policy_refs")},
                                       "label": "Create confidential mock case for Employee Relations", "preview": ticket["preview"]}
        trace.add("escalation", **plan.escalation)
        return plan

    async def _confirm(self, ctx, trace, action: dict) -> Plan:
        tool, args = action.get("tool"), dict(action.get("arguments") or {})
        trace.add("user_confirmation", tool=tool, confirmed=True)
        if tool not in CONFIRMABLE_TOOLS:
            trace.add("guardrail", rule="confirmable_tools_allowlist", decision="blocked", tool=tool)
            return Plan("confirm_action", refused=True, use_llm=False, short_answer=f"The action '{tool}' cannot be executed by the assistant.")
        if ctx["employee_id"] and args.get("employee_id") and args["employee_id"] != ctx["employee_id"]:
            trace.add("guardrail", rule="action_owner_mismatch", decision="blocked")
            return Plan("confirm_action", refused=True, use_llm=False, short_answer="That action belongs to a different employee and was not executed.")
        args["confirm"] = True
        out = await self.tool(trace, tool, args, "user explicitly confirmed the mock action")
        if out.get("created"):
            t = out["ticket"]
            return Plan("confirm_action", use_llm=False, short_answer=(
                f"Done — mock ticket **{t['ticket_id']}** was created (severity {t['severity']}, routed to {t['routed_to']}). "
                "This is a demo record only; no real HR system was updated."))
        return Plan("confirm_action", use_llm=False, short_answer=f"The action could not be completed: {out.get('message', out)}")

    @staticmethod
    def _error_plan(out: dict) -> Plan:
        if out.get("error") in ("employee_not_found", "missing_employee_id"):
            return Plan("clarify", use_llm=False, clarification=f"{out['message']} Please select a valid employee ID (e.g. E1001).")
        return Plan("error", use_llm=False, short_answer=(
            f"I couldn't complete this because a required HR tool is unavailable ({out.get('error')}: {out.get('message', '')}). "
            "Please try again shortly, or contact the People Team at people@lumora.example."))

    # ---------------------------------------------------------------- ReAct mode
    async def _react(self, ctx, trace, parsed) -> dict:
        tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"][:1000],
                                                  "parameters": t["input_schema"]}} for t in self.mcp.tool_schemas()]
        msgs: list[dict] = [{"role": "system", "content": prompts.REACT_SYSTEM.format(today=today().isoformat(), employee_id=ctx["employee_id"] or "unknown (ask for it if needed)")},
                            {"role": "user", "content": ctx["message"]}]
        trace.add("plan", workflow="react", rag_only=False, reason="LLM selects MCP tools via function calling")
        final = None
        for _ in range(int(os.environ.get("REACT_MAX_STEPS", "8"))):
            t = time.perf_counter()
            msg = self.llm.chat(msgs, tools=tools, max_tokens=900)
            trace.add("llm", purpose="react_step", latency_ms=round((time.perf_counter() - t) * 1000),
                      tool_calls=[tc.function.name for tc in (msg.tool_calls or [])])
            if not msg.tool_calls:
                final = msg.content or ""
                break
            msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = tc.function.name
                if name == "create_mock_hr_ticket" and args.get("confirm"):
                    args["confirm"] = False
                    trace.add("guardrail", rule="confirmation_required", decision="forced confirm=false")
                if name == "lookup_employee_profile":
                    args["requester_id"] = ctx["employee_id"]
                if name in ("check_pto_balance", "lookup_benefits_status", "check_policy_compliance", "draft_hr_email",
                            "generate_onboarding_checklist", "create_mock_hr_ticket") and ctx["employee_id"] and \
                        str(args.get("employee_id", "")).upper() not in ("", ctx["employee_id"]):
                    trace.add("guardrail", rule="privacy", decision="employee_id overridden to acting employee")
                    args["employee_id"] = ctx["employee_id"]
                out = await self.tool(trace, name, args, "selected by LLM")
                if name == "search_policy_documents":
                    for h in out.get("hits", []):
                        ctx["hits"].setdefault(h["citation"], h)
                if name == "get_policy_section" and out.get("parts"):
                    text = "\n".join(p["text"] for p in out["parts"])
                    ctx["sections"][out["citation"]] = {"doc_id": out["doc_id"], "title": out["title"], "section_id": out["section_id"],
                                                        "section_title": out["parts"][0]["section_title"], "citation": out["citation"],
                                                        "snippet": best_snippet(text, ctx["message"]), "text": text}
                if name == "create_mock_hr_ticket" and out.get("status") == "confirmation_required":
                    ctx["pending_action"] = {"tool": name, "arguments": {k: v for k, v in out["preview"].items() if k in ("employee_id", "category", "severity", "summary", "policy_refs")},
                                             "label": "Create mock HR ticket", "preview": out["preview"]}
                ctx["tool_outputs"].setdefault(name, out)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)[:6000]})
        if final is None:
            final = "I wasn't able to complete this request within the step limit. Please contact the People Team at people@lumora.example."
        plan = Plan(parsed.intent, use_llm=False, pending_action=ctx.get("pending_action"))
        plan.short_answer = ""
        plan.extra_md = final
        return await self._finish(ctx, trace, parsed, plan, mode="react", synthesized="llm")

    # ---------------------------------------------------------------- synthesis + response
    def _evidence(self, ctx) -> dict[str, dict]:
        ev = dict(ctx["hits"])
        ev.update(ctx["sections"])
        return ev

    async def _finish(self, ctx, trace, parsed, plan: Plan, mode: str, synthesized: str | None = None) -> dict:
        draft = plan.render()
        # citation self-verification: fetch any cited section that was not retrieved or returned by a tool
        known = set(self._evidence(ctx)) | {f"{d} §{s}" for d, s in parse_refs(trace.raw_outputs)}
        missing = [(d, s) for d, s in dict.fromkeys(REF_RE.findall(draft)) if f"{d} §{s}" not in known]
        if missing:
            await self.fetch_refs(ctx, trace, missing, ctx["message"], limit=4)
        evidence = self._evidence(ctx)
        synthesis = synthesized or "template"
        if synthesized is None and plan.use_llm and not plan.refused and self.llm.available and evidence:
            t = time.perf_counter()
            try:
                ev_txt = "\n\n".join(f"[{c}] {e['title']} > {e.get('section_title', '')}\n{e['text'][:900]}" for c, e in list(evidence.items())[:8])
                tool_txt = json.dumps({s.get("tool"): s.get("output") for s in trace.steps if s["type"] == "tool_call"
                                       and s.get("tool") != "search_policy_documents"}, default=str)[:3500]
                user = prompts.SYNTHESIS_USER.format(question=ctx["message"], employee=ctx["employee_id"] or "not provided",
                                                     plan=draft, evidence=ev_txt, tools=tool_txt)
                msg = self.llm.chat([{"role": "system", "content": prompts.SYNTHESIS_SYSTEM}, {"role": "user", "content": user}], max_tokens=700)
                text = (msg.content or "").strip()
                if text:
                    # keep structured extras (tables, drafts) that the LLM must not rewrite
                    draft = text + (("\n\n" + plan.extra_md) if plan.extra_md and plan.extra_md[:40] not in text else "")
                    synthesis = "llm"
                trace.add("llm", purpose="grounded_synthesis", model=self.llm.model, latency_ms=round((time.perf_counter() - t) * 1000))
            except Exception as exc:  # noqa: BLE001
                trace.add("llm", purpose="grounded_synthesis", status="failed_fallback_to_template", error=str(exc)[:200])
        # ---- citation validation (groundedness guardrail)
        allowed = set(evidence)
        tool_refs = {f"{d} §{s}" for d, s in parse_refs(trace.raw_outputs)}
        allowed |= tool_refs
        cited = [f"{d} §{s}" for d, s in REF_RE.findall(draft)]
        unsupported = sorted({c for c in cited if c not in allowed and not any(a.startswith(c + ".") or c.startswith(a + ".") for a in allowed)})
        if unsupported:
            trace.add("guardrail", rule="citation_validation", unsupported=unsupported, decision="flagged")
        used = []
        for c in dict.fromkeys(cited):
            e = evidence.get(c) or next((v for k, v in evidence.items() if k.startswith(c + ".") or c.startswith(k + ".")), None)
            if e and e["citation"] not in {u["citation"] for u in used}:
                used.append({k: e.get(k) for k in ("doc_id", "title", "section_id", "section_title", "citation", "snippet")})
            elif not e and c in tool_refs and c not in {u["citation"] for u in used}:
                d, sec = c.split(" §")
                used.append({"doc_id": d, "title": CATALOG.get(d, d), "section_id": sec, "section_title": "",
                             "citation": c, "snippet": "(cited by an MCP tool's encoded policy rule)"})
        if not used and not plan.refused and plan.intent not in ("out_of_scope", "clarify", "confirm_action", "error"):
            used = [{k: e.get(k) for k in ("doc_id", "title", "section_id", "section_title", "citation", "snippet")} for e in list(evidence.values())[:3]]
        tools_used = [s["tool"] for s in trace.steps if s["type"] == "tool_call"]
        latency = round((time.perf_counter() - trace.t0) * 1000)
        basis = "refusal/redirect (no tools)" if plan.refused and not tools_used else (
            f"{len(used)} policy citation(s) + MCP tool results: {', '.join(dict.fromkeys(t for t in tools_used if t != 'search_policy_documents')) or 'none'}")
        trace.add("final", answer_basis=basis, synthesis=synthesis, grounded=not unsupported and (bool(used) or plan.refused or plan.intent in ("clarify", "confirm_action")),
                  escalation=plan.escalation, latency_ms=latency)
        return {
            "answer": draft or "(no answer)",
            "intent": parsed.intent, "mode": mode, "synthesis": synthesis,
            "citations": used, "tools_used": tools_used, "trace": trace.steps,
            "escalation": plan.escalation, "pending_action": plan.pending_action,
            "clarification_needed": bool(plan.clarification), "refused": plan.refused,
            "grounded": not unsupported, "unsupported_citations": unsupported,
            "employee_id": ctx["employee_id"], "latency_ms": latency,
            "llm": self.llm.describe(),
        }
