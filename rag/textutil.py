"""Dependency-free text helpers shared by the RAG index and the agent."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
STOP = set("""a an and are as at be by can do does for from has have how i if in is it its me my of on or our
should so that the their them then there these they this to was we what when where which who will with you your
""".split())
REF_RE = re.compile(r"\b(POL-\d{3})\s*§\s*(\d+(?:\.\d+)*)")


NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "nine": "9", "ten": "10", "twenty": "20", "thirty": "30"}


ALIASES = {"wifi": "wi", "401k": "401", "email": "email"}


SUFFIXES = ("ments", "ment", "ions", "ion", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "al", "s", "e")


def stem(tok: str) -> str:
    """Very light suffix stripping so 'violates'/'violations', 'approved'/'approval' match."""
    if tok.isdigit() or len(tok) <= 4:
        return tok
    for suf in SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            tok = tok[: -len(suf)]
            break
    if len(tok) > 4 and tok.endswith("e"):
        tok = tok[:-1]
    return tok


def _norm(tok: str) -> str:
    tok = ALIASES.get(tok, NUMBER_WORDS.get(tok, tok))
    return stem(tok)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"'s\b|'", " ", text.lower().replace("\u2019", "'")).replace("wi-fi", "wi fi")
    return [_norm(t) for t in TOKEN_RE.findall(text) if t not in STOP and len(t) > 1 or t.isdigit()]


def _units(text: str) -> list[str]:
    """Split text into sentences; markdown/html table rows become 'a — b' units and header rows are dropped."""
    units: list[str] = []
    prev_table = False
    intro = ""
    for line in text.splitlines():
        ls = line.strip()
        if not ls:
            continue
        item = re.match(r"^(?:[-•]|\d+\.)\s+(.*)$", ls)
        if item and intro:
            units.append(f"{intro} {item.group(1).strip()}")
            continue
        intro = ls if ls.endswith(":") and len(ls) < 200 else ""
        if intro:
            continue
        is_row = ls.count("|") >= 2
        if is_row:
            cells = [c.strip() for c in ls.strip("|").split("|") if c.strip()]
            if not cells or all(re.fullmatch(r"[-: ]+", c) for c in cells):
                continue
            if not prev_table:  # header row
                prev_table = True
                continue
            units.append(" — ".join(cells))
        else:
            prev_table = False
            units.extend(x for x in re.split(r"(?<=[.!?])\s+", ls) if x.strip())
    return units




QUANT_Q = re.compile(r"\b(how (many|much|long|quickly|soon|often|far)|when|what is the (limit|maximum|deadline|rate|amount)|limit|deadline)\b", re.I)
QUANT_S = re.compile(r"\d|\b(immediately|annually|monthly|weekly|daily|hour|day|week|month|year)s?\b", re.I)


def snippet_score(text: str, query: str) -> float:
    """Score of the best sentence in `text` for `query` (same scoring as best_snippet)."""
    return _scored(text, query)[0][0] if _scored(text, query) else 0.0


def _scored(text: str, query: str) -> list[tuple[float, int, str]]:
    sents = _units(text) or [text]
    q = set(tokenize(query))
    proper = {_norm(w.lower()) for w in re.findall(r"(?<!^)(?<![.?!]\s)\b([A-Z][a-z]{2,})\b", query)}
    quant = bool(QUANT_Q.search(query))

    def score(s: str) -> float:
        toks = set(tokenize(s))
        sc = len(q & toks) + len(proper & toks)
        if quant and QUANT_S.search(s):
            sc += 1.5
        return sc

    return sorted(((score(s), i, s.strip()) for i, s in enumerate(sents) if s.strip()), key=lambda x: (-x[0], x[1]))


def best_snippet(text: str, query: str, max_chars: int = 320) -> str:
    sents = _units(text) or [text]
    q = set(tokenize(query))
    # proper nouns in the query (e.g. "London") are strong signals
    proper = {_norm(w.lower()) for w in re.findall(r"(?<!^)(?<![.?!]\s)\b([A-Z][a-z]{2,})\b", query)}
    quant = bool(QUANT_Q.search(query))

    def score(s: str) -> float:
        toks = set(tokenize(s))
        sc = len(q & toks) + len(proper & toks)
        if quant and QUANT_S.search(s):
            sc += 1.5
        return sc

    scored = [(score(s), i, s.strip()) for i, s in enumerate(sents) if s.strip()]
    if not scored:
        return text[:max_chars]
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:2]
    if len(top) == 2 and (top[1][0] == 0 or top[1][0] < top[0][0] / 2):
        top = top[:1]
    def clean(u: str) -> str:  # drop list markers and leading section numbers like "7.1 "
        return re.sub(r"^(?:[-•]\s*|\d+(?:\.\d+)+\s+)", "", u.strip()).strip()

    top = [(sc, i, clean(u)) for sc, i, u in top]
    snippet = " ".join(u for _, _, u in sorted(top, key=lambda x: x[1]))
    if len(snippet) > max_chars and len(top) == 2:  # don't let a weaker sentence push the best one out
        snippet = top[0][2]
    return snippet if len(snippet) <= max_chars else snippet[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def parse_refs(obj) -> list[tuple[str, str]]:
    """Find all 'POL-xxx §n.n' references anywhere inside a (nested) object or string."""
    found: list[tuple[str, str]] = []

    def walk(o):
        if isinstance(o, str):
            found.extend(REF_RE.findall(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    seen, out = set(), []
    for r in found:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out
