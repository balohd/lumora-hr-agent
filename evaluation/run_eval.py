"""Evaluate the agentic RAG system end-to-end (agent -> MCP client -> MCP server -> RAG/mock data).

    python -m evaluation.run_eval                 # full run + ablations -> evaluation/results/
    python -m evaluation.run_eval --quick         # main run only (used in CI)

Metrics
  Answer quality : groundedness (citations verified + sentence-level support), citation accuracy
                   (gold section cited), citation precision (doc-level), keyword partial / exact match.
  Agent behavior : tool selection accuracy, workflow completion rate, behavior accuracy
                   (answer / clarify / refuse / escalate), action-safety pass rate.
  System         : latency p50/p95 (warm), cold start (MCP spawn + index load, first request).
  Ablations      : retrieval k, retrieval mode (hybrid/dense/lexical, rerank), chunk size, tool availability.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HR_TODAY", "2026-10-30")
os.environ.setdefault("RUNTIME_TICKETS_FILE", str(ROOT / "data" / "eval_runtime_tickets.json"))

from agent.llm import LLM  # noqa: E402
from agent.mcp_client import MCPClientManager  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402
from rag.textutil import REF_RE, STOP, tokenize  # noqa: E402

EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"
random.seed(int(os.environ.get("SEED", "42")))
UNSAFE_CLAIM = re.compile(r"\b(I have|I've|I) (submitted|sent|approved|filed) (your|the)\b", re.I)


def load_items() -> list[dict]:
    return [json.loads(ln) for ln in EVAL_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ------------------------------------------------------------------------------------------ helpers
def pct(v: float) -> float:
    return round(100 * v, 1)


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 1)


def cite_match(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def keyword_score(answer: str, groups: list[list[str]]) -> float:
    if not groups:
        return 1.0
    a = answer.lower()
    return sum(any(k.lower() in a for k in g) for g in groups) / len(groups)


class SectionText:
    """Look up cited section text from the index for sentence-level support checks."""

    def __init__(self):
        from rag.retriever import Retriever

        self.r = Retriever()

    def get(self, doc: str, sec: str) -> str:
        return " ".join(c["text"] for c in self.r.get_section(doc, sec))


def support_ratio(answer: str, sections: SectionText, tool_text: str = "") -> tuple[float, int]:
    """Share of cited sentences whose content tokens are >=50% present in the cited section text(s)
    plus the MCP tool results the agent received (tool facts such as balances are evidence too)."""
    segs = []
    for line in answer.splitlines():
        if line.strip().startswith("|"):
            continue  # compliance tables cite in a column; checked via tool refs
        segs.extend(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z*])", line) if REF_RE.search(s))
    if not segs:
        return 1.0, 0
    ok = 0
    for s in segs:
        refs = REF_RE.findall(s)
        text = " ".join(sections.get(d, sec) for d, sec in refs) + " " + tool_text
        st = {t for t in tokenize(re.sub(REF_RE, " ", re.sub(r"[*_`>\[\]]", " ", s))) if t not in STOP}
        ct = set(tokenize(text))
        if st and len(st & ct) / len(st) >= 0.5:
            ok += 1
    return ok / len(segs), len(segs)


def behavior_of(r: dict) -> str:
    if r["refused"]:
        return "refuse"
    if r["intent"] == "hr_case_sensitive" or (r.get("escalation") and r["escalation"].get("severity") == 1):
        return "escalate"
    if r["clarification_needed"]:
        return "clarify"
    return "answer"


def score_item(item: dict, r: dict, sections: SectionText) -> dict:
    used = r["tools_used"]
    cites = [c["citation"] for c in r["citations"]]
    answer = r["answer"]
    gold = item.get("gold_citations", [])
    behavior = behavior_of(r)
    tool_ok = set(item.get("expected_tools", [])) <= set(used) and not (set(item.get("forbidden_tools", [])) & set(used))
    tool_errors = [s for s in r["trace"] if s["type"] == "tool_call" and s["status"] == "error"]
    kw = keyword_score(answer, item.get("gold_keywords", []))
    tool_text = " ".join(json.dumps(s.get("output", ""), default=str) for s in r["trace"]
                         if s["type"] == "tool_call" and s.get("tool") != "search_policy_documents")
    sup, n_cited = support_ratio(answer, sections, tool_text)
    is_answer = item["expected_behavior"] == "answer"
    confirmed_actions = [s for s in r["trace"] if s.get("tool") == "create_mock_hr_ticket" and s.get("arguments", {}).get("confirm")]
    safety = (not confirmed_actions and not UNSAFE_CLAIM.search(answer)
              and not any(x in answer for x in item.get("must_not_contain", []))
              and not (set(item.get("forbidden_tools", [])) & set(used)))
    res = {
        "id": item["id"], "category": item["category"], "intent": r["intent"], "latency_ms": r["latency_ms"],
        "behavior_expected": item["expected_behavior"], "behavior_actual": behavior,
        "behavior_correct": behavior == item["expected_behavior"],
        "tools_used": used, "tool_selection_correct": tool_ok, "tool_errors": len(tool_errors),
        "citations": cites, "keyword_score": round(kw, 3), "exact_match": kw == 1.0,
        "verified_citations": r["grounded"], "support_ratio": round(sup, 3), "cited_sentences": n_cited,
        "safety_pass": safety, "synthesis": r["synthesis"],
    }
    if is_answer:
        res["grounded"] = bool(r["grounded"] and cites and sup >= 0.8)
    if gold:
        res["citation_hit"] = any(cite_match(c, g) for c in cites for g in gold)
        gold_docs = {g.split(" §")[0] for g in gold}
        res["citation_precision_doc"] = round(sum(c.split(" §")[0] in gold_docs for c in cites) / len(cites), 3) if cites else 0.0
    if item.get("multi_doc_required"):
        res["multi_doc_covered"] = set(item["multi_doc_required"]) <= {c.split(" §")[0] for c in cites}
    if item["category"] == "tool_workflow":
        res["workflow_complete"] = bool(tool_ok and not tool_errors and behavior == "answer" and kw >= 0.5
                                        and (r["pending_action"] is not None or not item.get("expect_pending_action"))
                                        and (bool(r["escalation"]) or not item.get("expect_escalation")))
    return res


def aggregate(rows: list[dict]) -> dict:
    def mean(key, subset=None):
        vals = [r[key] for r in (subset or rows) if key in r]
        return pct(sum(bool(v) if isinstance(v, bool) else v for v in vals) / len(vals)) if vals else None

    lat = [r["latency_ms"] for r in rows]
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    return {
        "n": len(rows),
        "answer_quality": {
            "groundedness_pct": mean("grounded"), "mean_support_ratio_pct": mean("support_ratio"),
            "citation_accuracy_pct": mean("citation_hit"), "citation_precision_doc_pct": mean("citation_precision_doc"),
            "keyword_partial_match_pct": mean("keyword_score", [r for r in rows if r["behavior_expected"] == "answer"]),
            "exact_match_pct": mean("exact_match", [r for r in rows if r["behavior_expected"] == "answer"]),
            "multi_doc_coverage_pct": mean("multi_doc_covered"),
        },
        "agent_behavior": {
            "tool_selection_accuracy_pct": mean("tool_selection_correct"),
            "workflow_completion_pct": mean("workflow_complete"),
            "behavior_accuracy_pct": mean("behavior_correct"),
            "clarification_accuracy_pct": mean("behavior_correct", by_cat.get("ambiguous")),
            "escalation_accuracy_pct": mean("behavior_correct", by_cat.get("sensitive")),
            "refusal_accuracy_pct": mean("behavior_correct", [r for r in rows if r["behavior_expected"] == "refuse"]),
            "action_safety_pass_pct": mean("safety_pass"),
            "avg_tool_calls": round(statistics.mean(len(r["tools_used"]) for r in rows), 2),
        },
        "latency_ms": {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95), "mean": round(statistics.mean(lat), 1),
                       "max": max(lat)},
        "by_category": {c: {"n": len(v), "behavior_accuracy_pct": mean("behavior_correct", v),
                            "keyword_partial_match_pct": mean("keyword_score", v)} for c, v in by_cat.items()},
    }


# ------------------------------------------------------------------------------------------ runners
async def run_items(items: list[dict], env: dict | None = None, k: int | None = None, mode: str | None = None,
                    disabled: set[str] | None = None, sections: SectionText | None = None, repeat_latency: int = 0) -> dict:
    old = {key: os.environ.get(key) for key in (env or {})}
    os.environ.update(env or {})
    try:
        t0 = time.perf_counter()
        mcp = MCPClientManager(server_url="", disabled_tools=disabled or set())
        await mcp.connect()
        connect_ms = round((time.perf_counter() - t0) * 1000)
        agent = Agent(mcp, LLM(), mode=mode)
        if k:
            agent.k = k
        rows, responses = [], []
        first_ms = None
        for it in items:
            r = await agent.run(it["question"], it.get("employee_id"))
            first_ms = first_ms if first_ms is not None else r["latency_ms"]
            rows.append(score_item(it, r, sections))
            responses.append({"id": it["id"], "question": it["question"], "employee_id": it.get("employee_id"),
                              "answer": r["answer"], "citations": r["citations"], "tools_used": r["tools_used"],
                              "trace": [{k2: v for k2, v in s.items() if k2 != "output"} for s in r["trace"]]})
        warm = []
        for _ in range(repeat_latency):
            for it in items:
                warm.append((await agent.run(it["question"], it.get("employee_id")))["latency_ms"])
        await mcp.close()
        return {"rows": rows, "responses": responses, "connect_ms": connect_ms, "first_request_ms": first_ms, "warm": warm}
    finally:
        for key, v in old.items():
            if v is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = v


def build_chunk_index(max_words: int, overlap: int) -> str:
    from rag.ingest import build_index

    d = ROOT / "data" / f"index_c{max_words}"
    if not (d / "manifest.json").exists():
        build_index(index_dir=d, provider="lsa", max_words=max_words, overlap=overlap)
    return str(d)


async def main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items()
    sections = SectionText()
    llm = LLM().describe()
    print(f"Evaluating {len(items)} items | LLM: {llm} | embeddings: {sections.r.manifest['embedding']}")

    main_run = await run_items(items, sections=sections, repeat_latency=0 if args.quick else 2)
    summary = aggregate(main_run["rows"])
    warm = main_run["warm"] or [r["latency_ms"] for r in main_run["rows"]]
    summary["latency_ms"].update({"warm_p50": percentile(warm, 0.5), "warm_p95": percentile(warm, 0.95), "warm_samples": len(warm)})
    summary["cold_start"] = {"mcp_spawn_and_index_load_ms": main_run["connect_ms"], "first_request_ms": main_run["first_request_ms"],
                             "note": "In-process measurement. For the deployed free-tier service use evaluation/latency_http.py "
                                     "(cold start after idle spin-down is dominated by the platform)."}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "llm": llm,
              "embedding": sections.r.manifest["embedding"], "hr_today": os.environ.get("HR_TODAY"), "main": summary}

    if not args.quick:
        policy_items = [i for i in items if i["category"] in ("policy_simple", "multi_doc")]
        workflow_items = [i for i in items if i["category"] == "tool_workflow"]
        abl: dict = {"retrieval_k": {}, "retrieval_mode": {}, "chunk_size": {}, "tool_availability": {}}
        for k in (1, 3, 5, 8):
            rr = await run_items(policy_items, k=k, sections=sections)
            abl["retrieval_k"][f"k={k}"] = aggregate(rr["rows"])["answer_quality"] | {"p50_ms": aggregate(rr["rows"])["latency_ms"]["p50"]}
        for name, env in (("hybrid+rerank (default)", {}), ("hybrid, no rerank", {"RERANK": "0"}),
                          ("dense only", {"RETRIEVAL_MODE": "dense", "RERANK": "0"}), ("lexical (BM25) only", {"RETRIEVAL_MODE": "lexical", "RERANK": "0"}),
                          ("hybrid, no query rewrite", {"QUERY_REWRITE": "0"})):
            rr = await run_items(policy_items, env=env, sections=sections)
            abl["retrieval_mode"][name] = aggregate(rr["rows"])["answer_quality"]
        for mw, ov in ((120, 30), (220, 40), (400, 60)):
            idx = build_chunk_index(mw, ov)
            rr = await run_items(policy_items, env={"INDEX_DIR": idx}, sections=sections)
            m = json.loads((Path(idx) / "manifest.json").read_text())
            abl["chunk_size"][f"{mw} words / {ov} overlap ({m['num_chunks']} chunks)"] = aggregate(rr["rows"])["answer_quality"]
        for name, disabled in (("all tools", set()), ("without check_policy_compliance", {"check_policy_compliance"}),
                               ("without check_pto_balance", {"check_pto_balance"}),
                               ("without search_policy_documents", {"search_policy_documents"})):
            rr = await run_items(workflow_items + policy_items[:4], disabled=disabled, sections=sections)
            agg = aggregate(rr["rows"])
            abl["tool_availability"][name] = {**{k2: agg["agent_behavior"][k2] for k2 in ("workflow_completion_pct", "tool_selection_accuracy_pct", "action_safety_pass_pct")},
                                              **{k2: agg["answer_quality"][k2] for k2 in ("groundedness_pct", "keyword_partial_match_pct", "citation_accuracy_pct")}}
        report["ablations"] = abl

    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out / "details.jsonl", "w", encoding="utf-8") as f:
        for row in main_run["rows"]:
            f.write(json.dumps(row) + "\n")
    with open(out / "responses.jsonl", "w", encoding="utf-8") as f:
        for row in main_run["responses"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(render_md(report, main_run["rows"]), encoding="utf-8")
    print(json.dumps(report["main"], indent=2))
    print(f"\nWrote {out}/summary.json, details.jsonl, responses.jsonl, report.md")
    Path(os.environ["RUNTIME_TICKETS_FILE"]).unlink(missing_ok=True)


def render_md(rep: dict, rows: list[dict]) -> str:
    m = rep["main"]
    L = [f"# Evaluation results\n\nGenerated {rep['generated_at']} · LLM: `{rep['llm']['provider']}` · embeddings: `{rep['embedding']['provider']} / {rep['embedding']['model']}` · HR_TODAY={rep['hr_today']} · n={m['n']}\n",
         "## Answer quality\n", "| Metric | Value |", "|---|---|"]
    L += [f"| {k.replace('_pct', ' (%)').replace('_', ' ')} | {v} |" for k, v in m["answer_quality"].items()]
    L += ["\n## Agent behavior\n", "| Metric | Value |", "|---|---|"]
    L += [f"| {k.replace('_pct', ' (%)').replace('_', ' ')} | {v} |" for k, v in m["agent_behavior"].items()]
    L += ["\n## System latency (ms)\n", "| Metric | Value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in m["latency_ms"].items()]
    L += [f"| cold start: MCP spawn + index load | {m['cold_start']['mcp_spawn_and_index_load_ms']} |",
          f"| cold start: first request | {m['cold_start']['first_request_ms']} |"]
    L += ["\n## Per-item results\n", "| ID | Category | Behavior (exp → act) | Tools OK | Keywords | Gold cite | Grounded | Safe | ms |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['id']} | {r['category']} | {r['behavior_expected']} → {r['behavior_actual']} | {'✅' if r['tool_selection_correct'] else '❌'} | "
                 f"{r['keyword_score']:.2f} | {'—' if 'citation_hit' not in r else ('✅' if r['citation_hit'] else '❌')} | "
                 f"{'—' if 'grounded' not in r else ('✅' if r['grounded'] else '❌')} | {'✅' if r['safety_pass'] else '❌'} | {r['latency_ms']} |")
    if "ablations" in rep:
        L.append("\n## Ablations\n")
        for name, table in rep["ablations"].items():
            L.append(f"### {name.replace('_', ' ').title()}\n")
            cols = list(next(iter(table.values())).keys())
            L.append("| Config | " + " | ".join(c.replace("_pct", " %").replace("_", " ") for c in cols) + " |")
            L.append("|---" * (len(cols) + 1) + "|")
            for cfg, vals in table.items():
                L.append(f"| {cfg} | " + " | ".join(str(vals.get(c)) for c in cols) + " |")
            L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="main run only (no ablations, no latency repeats)")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    asyncio.run(main(ap.parse_args()))
