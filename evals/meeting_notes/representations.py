"""The four representations under comparison.

    A. raw                 - original meeting notes (control)
    B. en_zh               - the repo's token-aware English/Mandarin hybrid
    C. simple_compression  - generic telegraphic compression (function-word removal)
    D. new_language        - designed notation using single-token CJK operators

Design rule shared by B/C/D: **critical literals are never rewritten.** Names,
numbers, percentages, dates, and money keep their exact surface form. The
savings must come from function words and relation markers, never from the
facts under test - otherwise the comparison measures corruption, not
compression.

D exploits a measured property of this tokenizer: common Chinese characters
cost 1 token, while the English relation words they replace ("is responsible
for", "due by") cost 3-5. Entity names stay in English because English is
already cheaper for proper nouns.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


@dataclass
class Rendering:
    """One representation of one note."""

    method: str
    text: str
    #: Legend/grammar that must be sent for the text to be interpretable.
    #: Counted separately so gross and net compression can both be reported.
    grammar: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Literal protection
# ---------------------------------------------------------------------------

# Anything matching these must survive byte-identical in every representation.
_PROTECTED = re.compile(
    r"\$?\d[\d,.]*\s*(?:%|M\b|K\b|B\b|minutes?|hours?|days?)?"   # numbers/money/percent
    r"|\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
    r"|\b[A-Z][a-z]+\b"                                          # capitalised names
)


def protected_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PROTECTED.finditer(text)]


# ---------------------------------------------------------------------------
# A. Raw
# ---------------------------------------------------------------------------


def render_raw(text: str) -> Rendering:
    return Rendering(method="raw", text=text, grammar="")


# ---------------------------------------------------------------------------
# B. English/Mandarin hybrid (the repo encoder)
# ---------------------------------------------------------------------------


class EnZhRenderer:
    def __init__(self, dictionary_path: str | Path, include_grammar: bool = True):
        from token_language.encoder.encoder import Encoder

        self._enc = Encoder.from_dictionary(dictionary_path)
        self.include_grammar = include_grammar

    def render(self, text: str) -> Rendering:
        r = self._enc.encode(text, protect=protected_spans(text))
        grammar = self._enc.decoder_preamble(r.replacements) if self.include_grammar else ""
        return Rendering(
            method="en_zh",
            text=r.encoded_text,
            grammar=grammar,
            metadata={"substitutions": len(r.replacements)},
        )


# ---------------------------------------------------------------------------
# C. Simple compression baseline (generic, no grammar needed)
# ---------------------------------------------------------------------------

# Telegraphic compression: drop closed-class words that carry no fact content.
_DROPPABLE = {
    "the", "a", "an", "of", "to", "will", "was", "were", "is", "are", "be",
    "been", "that", "this", "with", "by", "for", "on", "at", "in", "and",
    "as", "it", "its", "their", "there", "has", "have", "had", "about",
    "once", "afterwards", "beforehand", "please", "then",
}
# Never dropped: negation and quantifier words change the answer.
_NEVER_DROP = {"not", "no", "never", "only", "least", "more", "all", "if",
               "otherwise", "unless", "did", "does", "do"}

_TOKEN = re.compile(r"\w+|[^\w\s]|\n")


def render_simple(text: str) -> Rendering:
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        if low in _NEVER_DROP:
            out.append(tok)
        elif low in _DROPPABLE:
            continue
        else:
            out.append(tok)
    joined = " ".join(out)
    joined = re.sub(r"\s+([,.;:%])", r"\1", joined)
    joined = re.sub(r"\s*\n\s*", "\n", joined)
    return Rendering(method="simple_compression", text=joined.strip(), grammar="")


# ---------------------------------------------------------------------------
# D. New token-efficient language
# ---------------------------------------------------------------------------

# Single CJK characters chosen because each costs exactly 1 token on the
# DeepSeek V4 Flash tokenizer, replacing 2-5 token English relation phrases.
OPERATORS: dict[str, str] = {
    "责": "is responsible for / owns / will do",
    "期": "due by (deadline)",
    "变": "changed from X to Y",
    "阻": "is blocked on",
    "决": "decision",
    "同": "agreed with",
    "否": "NOT (negates the statement)",
    "若": "if (condition)",
    "则": "then (consequence)",
    "说": "said / stated that",
    "问": "open question",
}

GRAMMAR = (
    "Notation used below. Each symbol replaces the English phrase shown:\n"
    + "\n".join(f"{k} = {v}" for k, v in OPERATORS.items())
    + "\nNames, numbers, dates and percentages are written normally and mean exactly what they say."
)

# Ordered rewrite rules. Longest/most specific first.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdid not agree with\b", re.I), "否同"),
    (re.compile(r"\bdoes not agree with\b", re.I), "否同"),
    (re.compile(r"\bagreed to\b", re.I), "同"),
    (re.compile(r"\bagreed with\b", re.I), "同"),
    (re.compile(r"\bis not blocked on\b", re.I), "否阻"),
    (re.compile(r"\bis blocked on\b", re.I), "阻"),
    (re.compile(r"\bmoved from\b", re.I), "变"),
    (re.compile(r"\bcorrected this to\b", re.I), "变"),
    (re.compile(r"\brose from\b", re.I), "变"),
    (re.compile(r"\bis owned by\b", re.I), "责"),
    (re.compile(r"\bowned by\b", re.I), "责"),
    (re.compile(r"\bowns\b", re.I), "责"),
    (re.compile(r"\bwill finish\b", re.I), "责"),
    (re.compile(r"\bwill complete\b", re.I), "责"),
    (re.compile(r"\bwill review\b", re.I), "责rev"),
    (re.compile(r"\bwill send\b", re.I), "责send"),
    (re.compile(r"\bis due\b", re.I), "期"),
    (re.compile(r"\bby the deadline of\b", re.I), "期"),
    (re.compile(r"\bdue\b", re.I), "期"),
    (re.compile(r"\bsaid that\b", re.I), "说"),
    (re.compile(r"\bsaid\b", re.I), "说"),
    (re.compile(r"\bdecision\b", re.I), "决"),
    (re.compile(r"\botherwise\b", re.I), "则否"),
    (re.compile(r"\bonly if\b", re.I), "若仅"),   # distinct from plain "if"
    (re.compile(r"\bif\b", re.I), "若"),
    (re.compile(r"\bdid not\b", re.I), "否"),
    (re.compile(r"\bdoes not\b", re.I), "否"),
    (re.compile(r"\bis not\b", re.I), "否"),
    (re.compile(r"\bnot\b", re.I), "否"),
    (re.compile(r"\bby\b(?=\s+(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day)", re.I), "期"),
]


def render_new_language(text: str, include_grammar: bool = True) -> Rendering:
    """Apply notation rules outside protected literal spans."""
    spans = protected_spans(text)

    def is_protected(a: int, b: int) -> bool:
        return any(a < e and s < b for s, e in spans)

    out = text
    applied = 0
    for pattern, symbol in _RULES:
        result, last, pieces = [], 0, 0
        for m in pattern.finditer(out):
            if is_protected(m.start(), m.end()):
                continue
            result.append(out[last : m.start()])
            result.append(symbol)
            last = m.end()
            pieces += 1
        if pieces:
            result.append(out[last:])
            out = "".join(result)
            applied += pieces
            spans = protected_spans(out)  # offsets shifted; recompute

    # The notation carries the relation structurally, so articles and a few
    # pure-filler adverbs add tokens without adding facts. Dropped only
    # outside protected spans.
    for filler in (r"\bthe\b", r"\ban?\b", r"\bafterwards\b", r"\bbeforehand\b"):
        pat = re.compile(filler, re.I)
        result, last, pieces = [], 0, 0
        for m in pat.finditer(out):
            if is_protected(m.start(), m.end()):
                continue
            result.append(out[last : m.start()])
            last = m.end()
            pieces += 1
        if pieces:
            result.append(out[last:])
            out = "".join(result)
            spans = protected_spans(out)

    # Collapse the whitespace the substitutions leave behind.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" ([,.;:])", r"\1", out)
    out = re.sub(r"(?<=[一-鿿]) (?=\w)", "", out)
    out = re.sub(r"(?<=\w) (?=[一-鿿])", "", out)

    return Rendering(
        method="new_language",
        text=out.strip(),
        grammar=GRAMMAR if include_grammar else "",
        metadata={"operators_applied": applied},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_renderers(dictionary_path: str | Path, include_grammar: bool = True) -> dict:
    enzh = EnZhRenderer(dictionary_path, include_grammar=include_grammar)
    return {
        "raw": render_raw,
        "en_zh": enzh.render,
        "simple_compression": render_simple,
        "new_language": lambda t: render_new_language(t, include_grammar=include_grammar),
    }


METHODS = ["raw", "en_zh", "simple_compression", "new_language"]
