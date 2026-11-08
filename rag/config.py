"""Central configuration for the RAG pipeline. All values can be overridden by env vars."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # optional .env support for local development
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


SEED = int(_env("SEED", "42"))
CORPUS_DIR = Path(_env("CORPUS_DIR", str(ROOT / "corpus")))
INDEX_DIR = Path(_env("INDEX_DIR", str(ROOT / "data" / "index")))
MOCK_DATA_DIR = Path(_env("MOCK_DATA_DIR", str(ROOT / "mock_data")))

# Embeddings: "fastembed" (BAAI/bge-small-en-v1.5, local ONNX, free) or "lsa" (TF-IDF + SVD, fully offline).
# "auto" tries fastembed and falls back to lsa if the model cannot be loaded.
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "auto").lower()
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
FASTEMBED_CACHE = _env("FASTEMBED_CACHE", str(ROOT / "data" / "models"))
LSA_DIM = int(_env("LSA_DIM", "256"))

# Chunking: heading-aware sections, split into word windows when a section is long.
CHUNK_MAX_WORDS = int(_env("CHUNK_MAX_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(_env("CHUNK_OVERLAP_WORDS", "40"))

# Retrieval
RETRIEVAL_K = int(_env("RETRIEVAL_K", "5"))
RETRIEVAL_MODE = _env("RETRIEVAL_MODE", "hybrid").lower()  # hybrid | dense | lexical
QUERY_REWRITE = _env("QUERY_REWRITE", "1") == "1"
