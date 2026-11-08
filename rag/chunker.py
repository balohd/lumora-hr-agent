"""Heading-aware chunking.

Strategy (justified in design-and-evaluation.md):
1. Each policy section (from the parsers) is the natural unit of meaning and of citation,
   so a chunk never crosses a section boundary.
2. Sections longer than CHUNK_MAX_WORDS are split into word windows with
   CHUNK_OVERLAP_WORDS overlap, preferring paragraph boundaries.
3. Each chunk is prefixed with a contextual header ("POL-002 Paid Time Off Policy > 6.2 Advance
   notice requirements") for embedding, so short chunks still carry their topic.
Chunking is fully deterministic (no randomness), so chunk IDs are stable across builds.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .parsers import Section


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section_id: str
    section_title: str
    heading_path: str
    text: str
    embed_text: str
    source_file: str
    source_format: str
    word_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _windows(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    if len(words) <= max_words:
        return [words]
    step = max(1, max_words - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + max_words])
        if start + max_words >= len(words):
            break
    return out


def chunk_section(sec: Section, max_words: int, overlap: int) -> list[Chunk]:
    paragraphs = [p.strip() for p in sec.text.split("\n\n") if p.strip()]
    # greedily pack paragraphs; fall back to word windows for very long paragraphs
    groups: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        plen = len(p.split())
        if cur and cur_len + plen > max_words:
            groups.append("\n\n".join(cur))
            # carry a small overlap: the tail of the previous group
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur, cur_len = ([tail] if tail else []), len(tail.split())
        cur.append(p)
        cur_len += plen
    if cur:
        groups.append("\n\n".join(cur))

    pieces: list[str] = []
    for g in groups:
        words = g.split()
        if len(words) > max_words * 1.3:
            pieces.extend(" ".join(w) for w in _windows(words, max_words, overlap))
        else:
            pieces.append(g)

    chunks = []
    heading = sec.heading_path or sec.section_title
    for i, piece in enumerate(pieces):
        header = f"{sec.doc_id} {sec.title} > {sec.section_id} {heading}"
        cid = f"{sec.doc_id}::{sec.section_id}::{i}"
        chunks.append(Chunk(
            chunk_id=cid, doc_id=sec.doc_id, title=sec.title, section_id=sec.section_id,
            section_title=sec.section_title, heading_path=heading, text=piece,
            embed_text=f"{header}\n{piece}", source_file=sec.source_file, source_format=sec.source_format,
            word_count=len(piece.split()),
        ))
    return chunks


def chunk_sections(sections: list[Section], max_words: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for sec in sections:
        for c in chunk_section(sec, max_words, overlap):
            # guarantee unique ids if a doc repeats a section number
            if c.chunk_id in seen:
                seen[c.chunk_id] += 1
                c.chunk_id = f"{c.chunk_id}.{seen[c.chunk_id]}"
            else:
                seen[c.chunk_id] = 0
            chunks.append(c)
    return chunks


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]
