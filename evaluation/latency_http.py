"""Measure end-to-end HTTP latency of a running deployment (local or Render), separating cold and warm starts.

    python -m evaluation.latency_http --url https://<your-app>.onrender.com
    python -m evaluation.latency_http --url http://localhost:8000 --repeats 3

Cold start: the first request after the service has been idle (on Render's free tier the instance spins
down after ~15 minutes without traffic; the first request then waits for the container to boot, install
nothing, start uvicorn, spawn the MCP server and load the index). Run this script after >15 min of idle
time to capture a cold start; subsequent requests are warm.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--n", type=int, default=15, help="number of representative queries (10-20)")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "latency_http.json"))
    args = ap.parse_args()
    items = [json.loads(ln) for ln in (ROOT / "evaluation" / "eval_set.jsonl").read_text().splitlines() if ln.strip()]
    reps = [i for i in items if i["category"] in ("policy_simple", "multi_doc", "tool_workflow", "ambiguous", "sensitive")][: args.n]

    with httpx.Client(timeout=180) as c:
        t = time.perf_counter()
        first = c.get(f"{args.url}/health")
        cold_health_ms = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        r = c.post(f"{args.url}/chat", json={"message": reps[0]["question"], "employee_id": reps[0].get("employee_id")})
        first_chat_ms = round((time.perf_counter() - t) * 1000, 1)
        warm, server = [], []
        for _ in range(args.repeats):
            for it in reps:
                t = time.perf_counter()
                resp = c.post(f"{args.url}/chat", json={"message": it["question"], "employee_id": it.get("employee_id")})
                warm.append(round((time.perf_counter() - t) * 1000, 1))
                server.append(resp.json().get("latency_ms", 0))
    res = {"url": args.url, "first_health_ms (cold if idle)": cold_health_ms, "health_status": first.status_code,
           "first_chat_ms": first_chat_ms, "first_chat_status": r.status_code,
           "warm_http_ms": {"n": len(warm), "p50": percentile(warm, .5), "p95": percentile(warm, .95), "mean": round(statistics.mean(warm), 1)},
           "warm_server_side_ms": {"p50": percentile(server, .5), "p95": percentile(server, .95)}}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
