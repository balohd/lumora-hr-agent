"""Parse policy documents (Markdown, HTML, TXT, PDF) into heading-aware sections.

Every parser returns a list of `Section` objects carrying the metadata needed for
citations: doc_id, document title, section id (e.g. "6.2"), section title and text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")
SYNTHETIC_NOTE_RE = re.compile(r"fictional company.*synthetic", re.I)


@dataclass
class Section:
    doc_id: str
    title: str
    section_id: str
    section_title: str
    text: str
    source_file: str
    source_format: str
    parent_title: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def heading_path(self) -> str:
        parts = [p for p in (self.parent_title, self.section_title) if p]
        return " > ".join(parts)


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_heading(heading: str, fallback_id: str) -> tuple[str, str]:
    m = SECTION_NUM_RE.match(heading.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return fallback_id, heading.strip()


def _drop_note_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not SYNTHETIC_NOTE_RE.search(ln)]


# --------------------------------------------------------------------------- markdown
def parse_markdown(path: Path, doc_id: str) -> list[Section]:
    lines = _drop_note_lines(path.read_text(encoding="utf-8").splitlines())
    title, sections = path.stem, []
    cur_id, cur_title, parent, buf = "0", "Document information", "", []
    top_title = ""

    def flush():
        body = clean_text("\n".join(buf))
        if body:
            sections.append(Section(doc_id, title, cur_id, cur_title, body, path.name, "markdown", parent))

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            sid, stitle = _split_heading(m.group(2), cur_id)
            if level == 2:
                parent = ""
                top_title = stitle
            cur_id, cur_title = sid, stitle
            if level > 2:
                parent = top_title
            continue
        buf.append(line.lstrip("> ").rstrip())
    flush()
    return sections


# --------------------------------------------------------------------------- html
def parse_html(path: Path, doc_id: str) -> list[Section]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    title = (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else soup.title.get_text(strip=True))
    root = soup.find("main") or soup.body or soup
    sections: list[Section] = []
    cur_id, cur_title, parent, buf = "0", "Document information", "", []
    top_title = ""

    def flush():
        body = clean_text("\n".join(_drop_note_lines(buf)))
        if body:
            sections.append(Section(doc_id, title, cur_id, cur_title, body, path.name, "html", parent))

    for el in root.find_all(["h1", "h2", "h3", "p", "li", "tr"]):
        name = el.name
        if name == "h1":
            continue
        if name in ("h2", "h3"):
            flush()
            buf = []
            sid, stitle = _split_heading(el.get_text(" ", strip=True), cur_id)
            if name == "h2":
                parent, top_title = "", stitle
            else:
                parent = top_title
            cur_id, cur_title = sid, stitle
        elif name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            buf.append(" | ".join(cells))
        elif name == "li":
            buf.append("- " + el.get_text(" ", strip=True))
        elif name == "p" and el.find_parent("li") is None:
            buf.append(el.get_text("\n", strip=True))
    flush()
    return sections


# --------------------------------------------------------------------------- txt
TXT_HEADING_RE = re.compile(r"^(\d+)\.\s+([A-Z][A-Z0-9 ,()'&/\-]+)$")


def parse_txt(path: Path, doc_id: str) -> list[Section]:
    lines = _drop_note_lines(path.read_text(encoding="utf-8").splitlines())
    title = next((ln.strip() for ln in lines if ln.strip()), path.stem).title()
    sections: list[Section] = []
    cur_id, cur_title, buf = "0", "Document information", []

    def flush():
        # split into subsections where paragraphs start with "N.M "
        body = clean_text("\n".join(buf))
        if not body:
            return
        paras = re.split(r"\n\s*\n", body)
        sub_id, sub_buf = None, []
        groups: list[tuple[str | None, list[str]]] = []
        for p in paras:
            m = re.match(r"^(\d+\.\d+)\s", p)
            if m:
                if sub_buf:
                    groups.append((sub_id, sub_buf))
                sub_id, sub_buf = m.group(1), [p]
            else:
                sub_buf.append(p)
        if sub_buf:
            groups.append((sub_id, sub_buf))
        for gid, gparas in groups:
            sid = gid or cur_id
            sub_title = cur_title
            if gid:
                lm = re.match(r"^\d+\.\d+\s+([A-Z][^.]{2,40})\.", gparas[0])
                sub_title = lm.group(1).strip() if lm else f"{cur_title} ({gid})"
            sections.append(Section(doc_id, title, sid, sub_title, "\n\n".join(gparas), path.name, "txt",
                                    cur_title if gid else ""))

    for i, line in enumerate(lines):
        if i == 0:
            continue  # title line
        m = TXT_HEADING_RE.match(line.strip())
        if m:
            flush()
            buf = []
            cur_id, cur_title = m.group(1), m.group(2).strip().title()
        else:
            buf.append(line)
    flush()
    return sections


# --------------------------------------------------------------------------- pdf
PDF_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)?)\.?\s+([A-Z][A-Za-z ,()&'/\-]{2,60})$")


def parse_pdf(path: Path, doc_id: str) -> list[Section]:
    reader = PdfReader(str(path))
    raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    lines = _drop_note_lines([ln.rstrip() for ln in raw.splitlines()])
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    title = lines[0].strip() if lines else meta_title or path.stem
    sections: list[Section] = []
    cur_id, cur_title, parent, buf, top_title = "0", "Document information", "", [], ""

    def flush():
        body = clean_text("\n".join(buf))
        if body:
            sections.append(Section(doc_id, title, cur_id, cur_title, body, path.name, "pdf", parent))

    for line in lines[1:]:
        m = PDF_HEADING_RE.match(line.strip())
        if m and not line.strip().endswith("."):
            flush()
            buf = []
            sid, stitle = m.group(1), m.group(2).strip()
            if "." in sid:
                parent = top_title
            else:
                parent, top_title = "", stitle
            cur_id, cur_title = sid, stitle
        else:
            buf.append(line)
    flush()
    return sections


PARSERS = {".md": parse_markdown, ".markdown": parse_markdown, ".html": parse_html, ".htm": parse_html,
           ".txt": parse_txt, ".pdf": parse_pdf}


def load_catalog(corpus_dir: Path) -> dict[str, dict]:
    cat_path = corpus_dir / "catalog.json"
    if not cat_path.exists():
        return {}
    data = json.loads(cat_path.read_text(encoding="utf-8"))
    return {d["file"]: d for d in data["documents"]}


def parse_corpus(corpus_dir: Path) -> list[Section]:
    catalog = load_catalog(corpus_dir)
    out: list[Section] = []
    for path in sorted(corpus_dir.iterdir()):
        parser = PARSERS.get(path.suffix.lower())
        if parser is None:
            continue
        meta = catalog.get(path.name, {})
        doc_id = meta.get("doc_id") or path.stem.split("_")[0]
        secs = parser(path, doc_id)
        if meta.get("title"):
            for s in secs:
                s.title = meta["title"]
        out.extend(secs)
    return out
