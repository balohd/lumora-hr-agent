# Deployment

| Item | Value |
|---|---|
| Platform | Render (free web service, single service: web app + agent + MCP server over stdio + FAISS index + synthetic data) |
| Deployed URL | `https://<your-service>.onrender.com`  ← replace after the first deploy |
| Health endpoint | `https://<your-service>.onrender.com/health` (add `?deep=1` to include the RAG index manifest) |
| MCP tool discovery | `https://<your-service>.onrender.com/tools` |
| API docs | `https://<your-service>.onrender.com/docs` |
| Demo tasks | `https://<your-service>.onrender.com/demo-tasks` (or the buttons in the UI) |
| Deploy trigger | GitHub Actions `deploy` job → Render deploy hook, **only after the `test` job passes** |

## Expected `/health` response

```json
{"status": "ok", "app": "lumora-hr-agent", "version": "1.0.0",
 "mcp": {"connected": true, "transport": "stdio", "tool_count": 10, "ping": true, "tools": ["check_policy_compliance", "..."]},
 "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile"}, "agent_mode": "planner"}
```

`status` is `"degraded"` (HTTP 503) if the MCP server cannot be reached; the app retries the connection on the next request.

## Free-tier cold starts

- Render's free web services **spin down after about 15 minutes without traffic**.
- The next request triggers a **cold start of roughly 30–60 seconds**: the container boots, uvicorn starts, the app spawns the MCP server process, and the MCP server loads the FAISS index and the bge-small embedding model (≈1.4 s of that is the app's own start-up, measured in-process).
- After that the service is **warm**. Offline-mode requests take tens of milliseconds server-side; with Groq LLM synthesis, expect about 1–3 seconds per request.
- **Before a demo**, open `/health` once and wait for `"status": "ok"`.
- Cold and warm latency against the live URL can be measured with:
  ```bash
  python -m evaluation.latency_http --url https://<your-service>.onrender.com
  ```
  Record the result here after deploying:

| Measurement | Value |
|---|---|
| First request after >15 min idle (cold) | _fill in_ |
| Warm `/chat` p50 / p95 (15 queries) | _fill in_ |

## Notes

- Memory: the app process is ≈65 MB and the MCP server process ≈200 MB with LSA (more with bge-small), within the 512 MB free-tier limit. If the instance runs out of memory, set `EMBEDDING_PROVIDER=lsa` and redeploy.
- Mock tickets are written to `data/runtime_tickets.json` on the instance's ephemeral disk and disappear on redeploy or restart. This is intentional: no persistent HR records are created.
