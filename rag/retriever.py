"""Top-k retrieval with optional query rewriting, doc filtering and hybrid (RRF) fusion."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from . import config
from .embeddings import load_embedder
from .store import VectorStore
from .textutil import best_snippet, tokenize

GENERIC_TERMS = set(tokenize(
    "lumora policy policies company employee employees rule rules allowed allow many much get need want know tell "
    "please like what can could would does do the my our i me is are say says about regarding happen happens "
    "happened someone somebody anyone anybody something thing things okay ok allowed actually really just "
    "going want wondering question ask"))
RERANK = __import__('os').environ.get('RERANK', '1') == '1'

# Domain query rewriting: expand employee vocabulary into policy vocabulary.
SYNONYMS: dict[str, str] = {
    r"\bvacation|time off|day(s)? off|holiday leave\b": "PTO paid time off",
    r"\bwfh|work(ing)? from home\b": "remote work hybrid",
    r"\babroad|another country|overseas|international(ly)?|portugal|spain|france|italy|germany|japan|mexico\b":
        "international temporary remote work from abroad approved country allowance workdays",
    r"\banother state|other state|out of state\b": "temporary remote work another US state payroll tax",
    r"\b401k|401\(k\)|retirement\b": "401(k) retirement match vesting",
    r"\bmaternity|paternity|baby|newborn|adoption\b": "parental leave",
    r"\breimburs\w*|expense\w*|pay me back\b": "expense reimbursement ExpenseIQ",
    r"\bchair|desk|furniture\b": "home office stipend ergonomic chair",
    r"\blaptop|computer|macbook\b": "laptop equipment IT provided not reimbursable",
    r"\bharass\w*|bully\w*|discriminat\w*|retaliat\w*|hostile\b":
        "harassment report Employee Relations Severity 1 escalation",
    r"\bfuneral|passed away|death|died\b": "bereavement leave",
    r"\bsick|ill|doctor\b": "paid sick time",
    r"\bhealth insurance|medical plan|dental|vision\b": "benefits medical dental vision eligibility",
    r"\bnew hire|first day|onboard\w*\b": "onboarding checklist first day first week",
    r"\btuition|mba|degree|course\w*|certification\w*|class(es)?\b": "learning and development stipend courses certifications",
    r"\bdog|pet\b": "",
    r"\bpaycheck|payday|paid on\b": "pay schedule payroll",
}


def expansions(query: str) -> dict[str, str]:
    """Map each query term that triggered a rewrite rule to its expansion text."""
    out: dict[str, str] = {}
    for pat, exp in SYNONYMS.items():
        for m in re.finditer(pat, query, re.I):
            for t in tokenize(m.group(0)):
                if exp:
                    out[t] = exp
    return out


def rewrite_query(query: str) -> str:
    extra = [exp for pat, exp in SYNONYMS.items() if exp and re.search(pat, query, re.I)]
    return query if not extra else f"{query} ({'; '.join(extra)})"


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        self.store = VectorStore.load(index_dir)
        self.embedder = load_embedder(self.store.manifest, index_dir)

    @property
    def manifest(self) -> dict:
        return self.store.manifest

    def search(self, query: str, k: int = config.RETRIEVAL_K, doc_ids: list[str] | None = None,
               mode: str = config.RETRIEVAL_MODE, rewrite: bool = config.QUERY_REWRITE, rerank: bool | None = None) -> dict:
        rerank = RERANK if rerank is None else rerank
        q = rewrite_query(query) if rewrite else query
        pool = max(k * 6, 30)
        ranked: dict[int, float] = {}
        dense, lexical = {}, {}
        if mode in ("hybrid", "dense"):
            for rank, (i, s) in enumerate(self.store.dense_search(self.embedder.embed_query(q), pool)):
                dense[i] = s
                ranked[i] = ranked.get(i, 0.0) + 1.0 / (60 + rank)
        if mode in ("hybrid", "lexical"):
            for rank, (i, s) in enumerate(self.store.lexical_search(q, pool)):
                lexical[i] = s
                ranked[i] = ranked.get(i, 0.0) + 1.0 / (60 + rank)
        order = sorted(ranked, key=lambda i: -ranked[i])
        q_terms = [t for t in dict.fromkeys(tokenize(query)) if t not in GENERIC_TERMS]
        exp_terms = [t for t in dict.fromkeys(tokenize(q)) if t not in GENERIC_TERMS and t not in q_terms]
        if rerank and q_terms and order:
            # lexical-overlap rerank of the fused candidate pool (cheap cross-check of the original query)
            top_rrf = ranked[order[0]]
            def rr(i):
                toks = set(tokenize(self.store.chunks[i]["embed_text"]))
                w = {**{t: 0.5 * self._idf(t) for t in exp_terms}, **{t: self._idf(t) for t in q_terms}}
                overlap = sum(v for t, v in w.items() if t in toks) / sum(w.values())
                return ranked[i] / top_rrf + 0.6 * overlap
            order = sorted(order[: max(k * 4, 20)], key=lambda i: -rr(i)) + order[max(k * 4, 20):]
        if doc_ids:
            wanted = {d.upper() for d in doc_ids}
            order = [i for i in order if self.store.chunks[i]["doc_id"] in wanted]
        hits = []
        for i in order[:k]:
            c = self.store.chunks[i]
            hits.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"],
                "section_id": c["section_id"], "section_title": c["section_title"],
                "heading_path": c["heading_path"], "citation": f"{c['doc_id']} §{c['section_id']}",
                "score": round(ranked[i], 5), "dense_score": round(dense.get(i, 0.0), 4),
                "lexical_score": round(lexical.get(i, 0.0), 3),
                "snippet": best_snippet(c["text"], query), "text": c["text"],
            })
        # evidence coverage: idf-weighted share of the query's content terms found in the returned passages
        hit_toks = set()
        for h in hits:
            hit_toks |= set(tokenize(h["title"] + " " + h["heading_path"] + " " + h["text"]))
        exp = expansions(query) if rewrite else {}

        def is_covered(t: str) -> bool:
            if t in hit_toks:
                return True
            et = set(tokenize(exp.get(t, "")))
            return bool(et) and len(et & hit_toks) / len(et) >= 0.5

        total = sum(self._idf(t) for t in q_terms) or 1.0
        coverage = sum(self._idf(t) for t in q_terms if is_covered(t)) / total if q_terms else 1.0
        return {"query": query, "rewritten_query": q if q != query else None, "mode": mode, "k": k, "reranked": bool(rerank),
                "evidence_coverage": round(coverage, 3), "uncovered_terms": [t for t in q_terms if not is_covered(t)], "hits": hits}

    def _idf(self, term: str) -> float:
        idf = self.store.bm25.idf
        return idf.get(term, max(idf.values()) if idf else 1.0)

    def get_section(self, doc_id: str, section_id: str | None = None) -> list[dict]:
        doc_id = doc_id.upper().strip()
        out = []
        for c in self.store.chunks:
            if c["doc_id"] != doc_id:
                continue
            if section_id is None or c["section_id"] == section_id or c["section_id"].startswith(section_id + "."):
                out.append(c)
        return out

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.store.chunks:
            d = docs.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "title": c["title"], "format": c["source_format"],
                                              "file": c["source_file"], "sections": []})
            label = f"{c['section_id']} {c['section_title']}"
            if label not in d["sections"]:
                d["sections"].append(label)
        return list(docs.values())


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
