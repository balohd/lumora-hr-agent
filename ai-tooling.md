# AI Tooling Used

> Personalize this file before submitting: add your own experience of reviewing, running, deploying and recording the project. The sections below describe how the initial implementation was produced.

## Tools

| Tool | Used for |
|---|---|
| **Claude (Anthropic)**, Opus model, in the Claude app's agentic workspace (a Claude Code–based coding agent with a shell, file editing and a headless browser) | Turning the project brief into an architecture; writing the synthetic policy corpus and mock data; implementing the RAG pipeline, MCP server, agent orchestrator, web UI, tests, CI/CD and deployment config; designing and running the evaluation; drafting the documentation |
| Groq (Llama 3.3 70B), optional at runtime | Grounded answer synthesis and ReAct tool-calling mode inside the application (not used to write code) |

## How the AI assistant was used

1. **Requirements → plan.** The full project brief was given to Claude, which broke it into a tracked task list (corpus, data, RAG, MCP, agent, web app, CI/CD, evaluation, docs, verification) and worked through it step by step.
2. **Synthetic content generation.** Claude wrote 13 interlinked policy documents (about 16k words) in four formats, deliberately cross-referenced so that multi-document questions are realistic. It also wrote mock employee, PTO, benefits and holiday data designed to exercise edge cases: part-time, contractor, intern, PIP, Restricted data access, Customer Support blackout, and a new hire.
3. **Code generation with immediate verification.** Each component was run as soon as it was written. Examples: parsing all four formats and printing the sections; probing retrieval with sample queries; calling the MCP server over a real stdio session; running both demo workflows end-to-end; taking Playwright screenshots of the UI to check the layout.
4. **Evaluation-driven iteration.** The first evaluation runs exposed concrete weaknesses, which were fixed with general mechanisms rather than by special-casing test questions:
   - Intent misroutes: "How many PTO days can I carry over?" was treated as a balance lookup, and general multi-doc questions were routed to personal workflows. Fixed with policy-vs-request rules.
   - Out-of-corpus questions (e.g. "pets in the office") were answered from loosely related passages. Fixed with the idf-weighted evidence-coverage guardrail.
   - Follow-up sub-questions lost their topic ("…and when must I use *them*?"). Fixed with context carry-over and a reformulation retry.
   - Snippets pulled table headers and missed numeric answers. Fixed with table-aware snippets and answer-type scoring.
   - Tokenization mismatches ("3-day", "violates" vs "violations"). Fixed with hyphen splitting and light stemming.

   Keyword partial match on answer items rose from about 72% to 95% and citation accuracy reached 100%, while tool-selection, behavior and safety metrics stayed at 100%.
5. **Tests as a safety net.** A pytest suite (28 tests) covers parsing, retrieval, compliance rules, MCP discovery and calls, app start-up, both demos, guardrails and graceful failures, plus LLM code paths via a stub LLM. It was re-run after every change.

## What worked well

- **Fast scaffolding of a complete, coherent system.** All layers (corpus → index → MCP → agent → UI → CI → eval) were produced in a single session with consistent interfaces.
- **Verification loops.** Running code, MCP sessions and screenshots right away caught integration bugs early. One example: tool outputs were truncated in the trace, which hid citations from the validator.
- **Systematic evaluation.** Writing the evaluation harness first made improvements measurable, and the ablations produced honest, non-obvious findings (BM25 ≥ hybrid with LSA; chunk size irrelevant under heading-aware chunking).
- **Safety design.** Confirmation gating, privacy checks and Severity-1 escalation were designed in from the start rather than bolted on.

## What did not work well / required correction

- **Library version drift.** The latest `mcp` SDK (2.x) had renamed `FastMCP` and changed client APIs relative to the assistant's knowledge. This was detected by inspecting the installed package; the project pins the stable `mcp==1.30.0` line.
- **Sandbox network limits.** Hugging Face was blocked in the development sandbox, so the bge-small (fastembed) embedding path could not be executed there. An offline LSA embedder with automatic fallback was added; it is what the reported evaluation uses. **The fastembed path should be verified on the first Render build** (check `/health?deep=1` → `rag_index.embedding.provider`).
- **No LLM key during development.** LLM synthesis and ReAct mode were tested only with a stub LLM (unit tests). Run the evaluation again with `GROQ_API_KEY` set to measure the deployed configuration.
- **First-pass heuristics were brittle.** The initial NLU and snippet rules needed several iterations (see step 4). Rule-based components need test coverage to stay reliable.
- **Risk of overfitting the evaluation.** Because the same assistant wrote the system and the evaluation set, fixes were kept general and the remaining partial answers (M2, M4) were left as reported. A blind test set written by someone else would be a stronger check.

## Human responsibilities

- Review all generated code and documents. Run the app, tests and evaluation locally.
- Create the GitHub repo, the Render service, the `GROQ_API_KEY` secret and the deploy hook; confirm the deployed URL.
- Record the demo and explain the MCP tool calls in your own words.
