"""Ingestion, chunking and retrieval tests."""
from pathlib import Path

from rag import config
from rag.chunker import chunk_sections
from rag.parsers import parse_corpus
from rag.retriever import Retriever, rewrite_query
from rag.textutil import best_snippet


def test_parses_all_four_formats_with_citation_metadata():
    sections = parse_corpus(config.CORPUS_DIR)
    formats = {s.source_format for s in sections}
    assert {"markdown", "html", "txt", "pdf"} <= formats
    assert len({s.doc_id for s in sections}) >= 12
    for s in sections:
        assert s.doc_id.startswith("POL-") and s.title and s.section_id and s.text
    # PDF and HTML sections keep their numbered headings
    pdf = [s for s in sections if s.source_format == "pdf"]
    assert any(s.section_id == "5.1" and "Eligibility" in s.section_title for s in pdf)


def test_chunking_is_deterministic_and_section_bounded():
    sections = parse_corpus(config.CORPUS_DIR)
    a = chunk_sections(sections, 120, 20)
    b = chunk_sections(sections, 120, 20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert all(c.word_count <= 120 * 1.3 + 20 for c in a)
    assert all(c.embed_text.startswith(c.doc_id) for c in a)


def test_top_k_retrieval_finds_expected_sections():
    r = Retriever()
    cases = {
        "How much notice do I need for 3 days of PTO?": "POL-002 §6.2",
        "Can I work from Portugal for six weeks?": "POL-005 §4.1",
        "What is the HSA employer contribution?": "POL-008 §4.2",
        "bereavement leave for a grandparent": "POL-009 §7.1",
        "Is an ergonomic chair covered by the home office stipend?": "POL-011 §5.2",
    }
    for q, expected in cases.items():
        cites = [h["citation"] for h in r.search(q, k=5)["hits"]]
        assert expected in cites, (q, cites)


def test_doc_filter_and_metadata():
    r = Retriever()
    res = r.search("approval requirements", k=4, doc_ids=["POL-006"])
    assert res["hits"] and all(h["doc_id"] == "POL-006" for h in res["hits"])
    h = res["hits"][0]
    for key in ("doc_id", "title", "section_id", "section_title", "citation", "snippet", "text"):
        assert h[key]


def test_out_of_corpus_query_has_low_evidence_coverage():
    r = Retriever()
    assert r.search("What is the policy on bringing pets to the office?")["evidence_coverage"] < 0.5
    assert r.search("What is the 401k match?")["evidence_coverage"] >= 0.5


def test_query_rewrite_and_snippet():
    assert "PTO" in rewrite_query("how much vacation do I have")
    assert "20" in best_snippet("Intro sentence. You may work up to 20 workdays abroad. Unrelated.", "workdays abroad")


def test_section_lookup():
    r = Retriever()
    parts = r.get_section("POL-005", "7")
    assert parts and parts[0]["section_title"].startswith("Exceptions")
    assert Path(config.INDEX_DIR / "faiss.index").exists()
