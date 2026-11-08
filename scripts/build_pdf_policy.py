"""Render corpus_src/*.md policy sources to PDF files in corpus/.

This gives the corpus a genuine PDF source format (in addition to Markdown,
HTML and TXT) so the ingestion pipeline exercises PDF parsing.

Usage:  python scripts/build_pdf_policy.py
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "corpus_src"
OUT = ROOT / "corpus"


def md_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def render(md_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    story, bullets = [], []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(b, styles["BodyText"])) for b in bullets],
                                      bulletType="bullet", leftIndent=14))
            bullets = []

    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line:
            flush_bullets()
            continue
        if line.startswith("# "):
            flush_bullets(); story.append(Paragraph(md_inline(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            flush_bullets(); story.append(Paragraph(md_inline(line[3:]), styles["Heading2"]))
        elif line.startswith("### "):
            flush_bullets(); story.append(Paragraph(md_inline(line[4:]), styles["Heading3"]))
        elif line.startswith("- "):
            bullets.append(md_inline(line[2:]))
        else:
            flush_bullets(); story.append(Paragraph(md_inline(line), styles["BodyText"]))
            story.append(Spacer(1, 4))
    flush_bullets()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER, leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch, title=md_path.stem,
                            author="Lumora Systems People Team (synthetic)")
    doc.build(story)


if __name__ == "__main__":
    for src in sorted(SRC.glob("*.md")):
        out = OUT / (src.stem + ".pdf")
        render(src, out)
        print(f"wrote {out.relative_to(ROOT)}")
