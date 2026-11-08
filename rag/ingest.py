"""Build the policy RAG index.

    python -m rag.ingest                    # provider from EMBEDDING_PROVIDER (default: auto)
    python -m rag.ingest --provider lsa     # fully offline
    python -m rag.ingest --max-words 120    # chunk-size ablation
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .chunker import chunk_sections, corpus_fingerprint
from .embeddings import create_embedder
from .parsers import parse_corpus
from .store import VectorStore

log = logging.getLogger("rag.ingest")


def build_index(corpus_dir: Path = config.CORPUS_DIR, index_dir: Path = config.INDEX_DIR,
                provider: str | None = None, max_words: int = config.CHUNK_MAX_WORDS,
                overlap: int = config.CHUNK_OVERLAP_WORDS) -> dict:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    t0 = time.time()
    sections = parse_corpus(corpus_dir)
    chunks = chunk_sections(sections, max_words, overlap)
    embedder = create_embedder(provider)
    texts = [c.embed_text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.embed_documents(texts)
    index_dir.mkdir(parents=True, exist_ok=True)
    embedder.save(index_dir)
    formats = sorted({c.source_format for c in chunks})
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding": embedder.describe(),
        "chunking": {"strategy": "heading-aware sections + word windows", "max_words": max_words, "overlap_words": overlap},
        "num_documents": len({c.doc_id for c in chunks}),
        "num_sections": len(sections),
        "num_chunks": len(chunks),
        "source_formats": formats,
        "corpus_fingerprint": corpus_fingerprint(chunks),
        "seed": config.SEED,
        "build_seconds": round(time.time() - t0, 2),
    }
    VectorStore.build(vectors, [c.to_dict() for c in chunks], manifest, index_dir)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="fastembed | lsa | auto")
    ap.add_argument("--index-dir", default=str(config.INDEX_DIR))
    ap.add_argument("--max-words", type=int, default=config.CHUNK_MAX_WORDS)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP_WORDS)
    args = ap.parse_args()
    m = build_index(index_dir=Path(args.index_dir), provider=args.provider, max_words=args.max_words, overlap=args.overlap)
    log.info("Index built: %s", m)


if __name__ == "__main__":
    main()
