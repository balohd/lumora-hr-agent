# Lumora HR Agent — Agentic RAG + MCP HR assistant

An agentic AI system that helps employees of **Lumora Systems, Inc.** (a fictional company) with HR policy questions and multi-step HR workflows. It combines:

- **Policy RAG** over 13 synthetic HR policy documents in four formats (Markdown, HTML, TXT, PDF; ~32 pages), with heading-aware chunking, local embeddings, a FAISS index, hybrid retrieval and cited answers.
- **An MCP server** (`mcp_server/`) exposing **10 tools**: policy search, section lookup, employee profile, PTO balance, benefits status, policy compliance checks, and *mock*, confirmation-gated actions (HR tickets, email drafts, onboarding checklists).
- **An agent orchestrator** (`agent/`) that interprets intent, decides whether RAG alone is enough, plans and executes MCP tool calls, applies safety guardrails and synthesizes grounded answers, with a visible operational trace.
- **A FastAPI web app** (`app/`) with a chat UI, `/chat`, `/health`, tool discovery and one-click demo tasks.

> All people, policies and records are **synthetic**. No real or private data is used.

**Deployed app:** see [`deployed.md`](deployed.md) · **Design & evaluation:** [`design-and-evaluation.md`](design-and-evaluation.md) · **AI tooling:** [`ai-tooling.md`](ai-tooling.md)

---

## Architecture (one service, free-tier friendly)

```
Browser (chat UI) ──HTTP──► FastAPI app (/chat, /health, /tools, /demo-tasks)
                                 │
                                 ▼
                       Agent orchestrator (agent/)
          intent + guardrails ─► workflow planner ─► grounded synthesis (LLM or template)
                                 │  MCP client (tools/list, tools/call)
                                 ▼  stdio (default)  or  Streamable HTTP (MCP_SERVER_URL)
                       MCP server "lumora-hr" (mcp_server/) ── 10 tools
                          │                        │
                          ▼                        ▼
               RAG index (rag/, FAISS +     Synthetic data (mock_data/*.json)
               BM25, data/index/)            + mock ticket store (data/runtime_tickets.json)
                                 
LLM provider (Groq / OpenRouter / OpenAI-compatible; optional) ◄── synthesis / ReAct mode
```

A detailed diagram, tool schemas and design rationale are in [`design-and-evaluation.md`](design-and-evaluation.md).

## Repository layout

| Path | Contents |
|---|---|
| `corpus/` | 13 policy documents (`.md`, `.html`, `.txt`, `.pdf`) + `catalog.json` metadata |
| `corpus_src/` | Markdown source of the PDF policy (`scripts/build_pdf_policy.py` renders it) |
| `mock_data/` | Synthetic employees, PTO balances, benefits, offices, holidays, seed tickets, machine-readable policy rules |
| `rag/` | Parsers, chunker, embeddings (fastembed / LSA), FAISS store, hybrid retriever, ingestion CLI |
| `mcp_server/` | MCP server (`server.py`), compliance engine, data access layer, exported `tool_schemas.json` (named `mcp_server/` rather than `mcp/` so it doesn't shadow the official `mcp` Python SDK) |
| `agent/` | Orchestrator, MCP client, intent/slot extraction (NLU), LLM adapter, prompts |
| `app/` | FastAPI app + single-file chat UI (`app/static/index.html`) |
| `evaluation/` | 29-item evaluation set with gold answers, runner, HTTP latency script, results |
| `tests/` | Unit, MCP, end-to-end and LLM-path tests (pytest) |
| `.github/workflows/ci.yml` | CI/CD: install → import check → index build → MCP smoke test → app start → tests → eval → deploy |
| `render.yaml`, `Dockerfile` | Render blueprint (native Python) and a container image for Railway / any Docker host |

## Local setup

Requires **Python 3.13** (3.12 also works).

```bash
git clone <your-repo-url> lumora-hr-agent && cd lumora-hr-agent
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                       # optional: add GROQ_API_KEY for LLM synthesis
python -m rag.ingest                                       # build the index (bge-small via fastembed; falls back to offline LSA)
uvicorn app.main:app --reload --port 8000                  # open http://localhost:8000
```

- `python -m rag.ingest --provider lsa` builds a fully offline index (no model download).
- The app spawns the MCP server automatically over **stdio**. To run the MCP server as a separate service instead:
  ```bash
  python -m mcp_server.server --transport streamable-http   # serves http://0.0.0.0:8765/mcp
  MCP_SERVER_URL=http://localhost:8765/mcp uvicorn app.main:app --port 8000
  ```
- Without an LLM key the system runs in **deterministic offline mode** (template synthesis from verified evidence). With `GROQ_API_KEY` (free tier) or `OPENROUTER_API_KEY` set, answers are rewritten by the LLM, with citations still validated. `AGENT_MODE=react` lets the LLM choose MCP tools itself via function calling.
- Secrets are read only from environment variables; `.env` is git-ignored.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `GROQ_API_KEY` / `OPENROUTER_API_KEY` / `LLM_MODEL` | auto / – | LLM for synthesis (optional) |
| `AGENT_MODE` | `planner` | `planner` (deterministic workflows) or `react` (LLM tool calling) |
| `EMBEDDING_PROVIDER` | `auto` | `fastembed` (BAAI/bge-small-en-v1.5), `lsa` (offline), `auto` |
| `RETRIEVAL_K`, `RETRIEVAL_MODE`, `RERANK`, `QUERY_REWRITE` | 5, hybrid, 1, 1 | Retrieval settings |
| `CHUNK_MAX_WORDS`, `CHUNK_OVERLAP_WORDS` | 220, 40 | Chunking |
| `EVIDENCE_COVERAGE_MIN` | 0.5 | Out-of-corpus guardrail threshold |
| `MCP_SERVER_URL` | – | Use a remote MCP server (Streamable HTTP) instead of stdio |
| `MCP_DISABLED_TOOLS` | – | Comma-separated tools to disable (failure simulation / ablation) |
| `HR_TODAY` | today | Pin the reference date for reproducible demos and evaluation |
| `SEED` | 42 | Fixed seed (SVD, numpy/random, LLM seed where supported) |

## Reproducing the two agentic demo tasks

**From the UI:** open the app and click **Demo 1** or **Demo 2** in the left panel. The acting employee is selected automatically. The right-hand **Agent trace** panel shows the intent, the plan, every MCP tool call (name, arguments, output, latency), guardrail decisions and the answer basis. When the agent proposes a mock action, a **Confirm & create** button appears; nothing is created until you click it.

**From an API client:**

```bash
# Demo 1 — Remote work eligibility (Priya, E1001: Portugal, six weeks)
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"I want to work remotely from Portugal for six weeks starting in January. Am I allowed to do that, and what do I need to do?","employee_id":"E1001"}' | jq '{answer, tools_used, citations, escalation, pending_action}'

# Demo 2 — PTO request guidance (Marcus, E1002: three days next week)
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Can I take three days of PTO next week?","employee_id":"E1002"}' | jq '{answer, tools_used, pending_action}'

# Confirm the proposed mock action (send back the pending_action exactly as returned)
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"","employee_id":"E1002","confirm_action":<pending_action JSON>}'

curl -s localhost:8000/health        # app + MCP connectivity, tool count, LLM config
curl -s localhost:8000/tools         # MCP tool discovery result (names, descriptions, JSON schemas)
curl -s localhost:8000/demo-tasks    # all demo tasks
```

Interactive API docs are available at `/docs`.

## Tests

```bash
pytest -q                     # 28 tests: parsing (4 formats), chunking, retrieval, compliance rules,
                              # MCP discovery + calls over stdio, app start-up + /health, both demo workflows,
                              # guardrails (out-of-scope, sensitive escalation, privacy, confirmation gate),
                              # graceful failures (disabled tool, unreachable MCP server), LLM paths with a stub LLM
python scripts/mcp_smoke.py   # standalone MCP tool discovery + tool call check (used in CI)
```

## Evaluation

```bash
python -m evaluation.run_eval            # 29 items + ablations -> evaluation/results/{summary.json,report.md,...}
python -m evaluation.run_eval --quick    # main run only (CI)
python -m evaluation.latency_http --url https://<your-app>.onrender.com   # deployed cold/warm latency
```

Headline results (offline configuration: LSA embeddings, template synthesis, n=29): groundedness 100%, citation accuracy 100%, keyword partial match 95%, tool selection 100%, workflow completion 100%, behavior (clarify/refuse/escalate) accuracy 100%, action safety 100%, latency p50 21 ms / p95 57 ms in-process. See [`evaluation/results/report.md`](evaluation/results/report.md) and [`design-and-evaluation.md`](design-and-evaluation.md#9-evaluation) for the full results, ablations and limitations.

## Deployment (Render free tier)

1. Push this repo to GitHub and share it with the `quantic-grader` account.
2. In Render: **New → Blueprint** → select the repo (uses `render.yaml`: one free web service, native Python runtime, `buildCommand` installs dependencies and builds the index, `startCommand` runs uvicorn, health check on `/health`).
3. In the Render service's **Environment**, set `GROQ_API_KEY` (optional but recommended).
4. In the Render service's **Settings → Deploy Hook**, copy the hook URL. In GitHub go to **Settings → Secrets and variables → Actions** and add:
   - secret `RENDER_DEPLOY_HOOK_URL` = the hook URL
   - variable `DEPLOY_URL` = `https://<your-service>.onrender.com` (used to verify the deploy is healthy)
5. Push to `main`. GitHub Actions runs the tests and calls the deploy hook **only if every test step passes** (`autoDeploy: false` in `render.yaml` prevents Render from deploying untested commits).
6. Record the URL in [`deployed.md`](deployed.md).

**Cold starts:** Render's free tier spins the service down after ~15 minutes of inactivity. The first request afterwards waits roughly 30–60 seconds while the instance boots, uvicorn starts, the MCP server process is spawned and the index and embedding model load. After that, requests are warm. Open `/health` before a demo to wake the service.

**Alternatives:** the `Dockerfile` runs the same single service on Railway, Fly.io or any Docker host (`docker build -t lumora-hr . && docker run -p 8000:8000 -e GROQ_API_KEY=... lumora-hr`). To deploy the MCP server separately, run a second service with `python -m mcp_server.server --transport streamable-http` and set `MCP_SERVER_URL` on the web service.
