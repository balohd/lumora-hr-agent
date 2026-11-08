"""Render design-and-evaluation.md from the template + evaluation set + latest results (keeps numbers in sync).

    python scripts/render_design_doc.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tpl = (ROOT / "scripts" / "design_doc_template.md").read_text(encoding="utf-8")
items = [json.loads(ln) for ln in (ROOT / "evaluation" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
rep = json.loads((ROOT / "evaluation" / "results" / "summary.json").read_text(encoding="utf-8"))
report_md = (ROOT / "evaluation" / "results" / "report.md").read_text(encoding="utf-8")
m = rep["main"]

rows = ["| ID | Category | Employee | Question | Expected answer / behavior | Gold citations | Expected tools |", "|---|---|---|---|---|---|---|"]
for it in items:
    rows.append(f"| {it['id']} | {it['category']} | {it.get('employee_id') or '—'} | {it['question']} | {it['gold_answer']} ({it['expected_behavior']}) | "
                f"{', '.join(it['gold_citations']) or '—'} | {', '.join(it.get('expected_tools', [])) or '—'} |")


def table(d):
    return "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k.replace('_pct', ' (%)').replace('_', ' ')} | {v} |" for k, v in d.items())


results = ("**Answer quality**\n\n" + table(m["answer_quality"]) + "\n\n**Agent behavior**\n\n" + table(m["agent_behavior"])
           + "\n\n**System latency (ms)**\n\n" + table(m["latency_ms"]))
items_md = report_md.split("## Per-item results")[1].split("## Ablations")[0].strip()
abl_md = report_md.split("## Ablations")[1].strip() if "## Ablations" in report_md else "_Run without --quick to produce ablations._"
abl_md = abl_md.replace("### ", "#### ")
cfg = f"embeddings {rep['embedding']['provider']} ({rep['embedding']['model']}), LLM {rep['llm']['provider']}, HR_TODAY={rep['hr_today']}, generated {rep['generated_at']}"

out = (tpl.replace("{{EVAL_TABLE}}", "\n".join(rows)).replace("{{RESULTS}}", results).replace("{{ITEMS}}", items_md)
       .replace("{{ABLATIONS}}", abl_md).replace("{{CONFIG}}", cfg)
       .replace("{{P50}}", str(m["latency_ms"]["warm_p50"])).replace("{{P95}}", str(m["latency_ms"]["warm_p95"]))
       .replace("{{COLD}}", str(m["cold_start"]["mcp_spawn_and_index_load_ms"])).replace("{{FIRST}}", str(m["cold_start"]["first_request_ms"])))
(ROOT / "design-and-evaluation.md").write_text(out, encoding="utf-8")
print("wrote design-and-evaluation.md")
