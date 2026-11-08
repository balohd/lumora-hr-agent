# Evaluation

| File | Purpose |
|---|---|
| `eval_set.jsonl` | 29 questions/tasks with gold answers, gold keyword groups, gold citations, expected behavior and expected/forbidden tools |
| `run_eval.py` | Runs every item through the real stack (agent → MCP client → MCP server over stdio → RAG + mock data), scores it and runs the ablations |
| `latency_http.py` | Cold/warm HTTP latency against a running deployment |
| `results/summary.json` | Aggregated metrics + ablations (machine-readable) |
| `results/report.md` | Human-readable report: metrics, per-item table, ablations |
| `results/details.jsonl` | Per-item scores |
| `results/responses.jsonl` | Full answers, citations, tools and traces (without bulky outputs) for every item |
| `results/latency_http_local.json` | HTTP latency measured against a local uvicorn instance |

```bash
python -m evaluation.run_eval                 # main run + ablations (≈1–2 min offline)
python -m evaluation.run_eval --quick         # main run only (CI)
GROQ_API_KEY=... python -m evaluation.run_eval --out evaluation/results_llm   # evaluate the LLM configuration
python -m evaluation.latency_http --url https://<your-service>.onrender.com
python scripts/render_design_doc.py           # refresh numbers in design-and-evaluation.md
```

`HR_TODAY` is pinned to `2026-10-30` by default so relative dates ("next week") and notice calculations are reproducible.
