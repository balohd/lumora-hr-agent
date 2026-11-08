"""Embedding providers.

- FastEmbedEmbedder: BAAI/bge-small-en-v1.5 (384-d) running locally via ONNX Runtime
  (the `fastembed` package). Free, no API key, ~130 MB download on first use.
- LSAEmbedder: TF-IDF (1-2 grams) + TruncatedSVD ("latent semantic analysis") fitted on the
  corpus. Fully offline and deterministic (fixed random_state). Used in CI, in sandboxed
  environments without model downloads, and as an automatic fallback.

Both return L2-normalised float32 vectors so inner product == cosine similarity.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, cache_dir: str = config.FASTEMBED_CACHE):
        from fastembed import TextEmbedding  # imported lazily: heavy

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = len(next(iter(self.model.embed(["dimension probe"]))))

    def fit(self, texts: list[str]) -> None:  # pretrained: nothing to fit
        return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack(list(self.model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.stack(list(self.model.query_embed([text]))))

    def save(self, index_dir: Path) -> None:
        return None

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model_name, "dim": self.dim}


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dim: int = config.LSA_DIM, seed: int = config.SEED):
        self.target_dim = dim
        self.seed = seed
        self.vectorizer = None
        self.svd = None
        self.dim = dim

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english",
                                          min_df=1, max_df=0.9)
        x = self.vectorizer.fit_transform(texts)
        self.dim = int(min(self.target_dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=self.dim, random_state=self.seed, n_iter=10)
        self.svd.fit(x)

    def _transform(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.svd.transform(self.vectorizer.transform(texts)))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])

    def save(self, index_dir: Path) -> None:
        with open(index_dir / "lsa_model.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd, "dim": self.dim}, f)

    @classmethod
    def load(cls, index_dir: Path) -> "LSAEmbedder":
        with open(index_dir / "lsa_model.pkl", "rb") as f:
            d = pickle.load(f)
        obj = cls(dim=d["dim"])
        obj.vectorizer, obj.svd, obj.dim = d["vectorizer"], d["svd"], d["dim"]
        return obj

    def describe(self) -> dict:
        return {"provider": self.name, "model": f"tfidf(1-2gram)+svd{self.dim}", "dim": self.dim}


def create_embedder(provider: str | None = None):
    """Create an embedder for *building* an index. 'auto' tries fastembed then falls back to LSA."""
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    if provider in ("fastembed", "auto"):
        try:
            return FastEmbedEmbedder()
        except Exception as exc:  # network blocked, model unavailable, etc.
            if provider == "fastembed":
                raise
            log.warning("fastembed unavailable (%s); falling back to LSA embeddings", exc)
    return LSAEmbedder()


def load_embedder(manifest: dict, index_dir: Path):
    """Load the same embedder that built the index (required for query/index compatibility)."""
    if manifest["embedding"]["provider"] == "fastembed":
        return FastEmbedEmbedder(model_name=manifest["embedding"]["model"])
    return LSAEmbedder.load(index_dir)
