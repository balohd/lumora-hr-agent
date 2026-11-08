"""Lightweight vector store: FAISS (IndexFlatIP) for vectors + JSONL for chunk metadata,
plus a small in-memory BM25 index for hybrid lexical/dense retrieval."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

from .textutil import tokenize


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.lens = np.array([len(t) for t in self.toks], dtype="float32")
        self.avgdl = float(self.lens.mean()) if len(self.lens) else 0.0
        self.tf = [Counter(t) for t in self.toks]
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.toks), dtype="float32")
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
            out[i] = s
        return out


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[dict], manifest: dict):
        self.index = index
        self.chunks = chunks
        self.manifest = manifest
        self.bm25 = BM25([c["embed_text"] for c in chunks])

    @staticmethod
    def build(vectors: np.ndarray, chunks: list[dict], manifest: dict, index_dir: Path) -> "VectorStore":
        index_dir.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype("float32"))
        faiss.write_index(index, str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return VectorStore(index, chunks, manifest)

    @staticmethod
    def load(index_dir: Path) -> "VectorStore":
        index = faiss.read_index(str(index_dir / "faiss.index"))
        chunks = [json.loads(ln) for ln in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln]
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        return VectorStore(index, chunks, manifest)

    def dense_search(self, qvec: np.ndarray, n: int) -> list[tuple[int, float]]:
        scores, idx = self.index.search(qvec.astype("float32"), min(n, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def lexical_search(self, query: str, n: int) -> list[tuple[int, float]]:
        s = self.bm25.scores(query)
        order = np.argsort(-s)[:n]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]
