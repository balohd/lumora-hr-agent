import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HR_TODAY", "2026-10-30")          # deterministic dates for tests
os.environ.setdefault("EMBEDDING_PROVIDER", "lsa")      # offline, no model download in CI
os.environ.setdefault("RUNTIME_TICKETS_FILE", str(ROOT / "data" / "test_runtime_tickets.json"))
for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
    os.environ.pop(k, None)                              # tests never call a real LLM
os.environ["LLM_PROVIDER"] = "none"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def built_index():
    from rag import config
    from rag.ingest import build_index

    if not (config.INDEX_DIR / "manifest.json").exists():
        build_index(provider=os.environ["EMBEDDING_PROVIDER"])
    yield config.INDEX_DIR
    Path(os.environ["RUNTIME_TICKETS_FILE"]).unlink(missing_ok=True)
