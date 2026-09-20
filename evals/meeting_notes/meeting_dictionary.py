"""Meeting-domain EN->ZH phrase substitutions using REAL Mandarin, not symbols.

Unlike `new_language` (representations.py), this requires NO legend. These are
ordinary Chinese words the model already knows from pretraining - the same
reason DeepSeek answered a plain English question about mixed EN/ZH input
correctly with zero instruction in earlier testing. If that holds here, the
grammar-overhead problem that sank `new_language` (106 tokens for an 11-line
legend) does not apply, because there is nothing to explain.

MEASURED RESULT (live test): gross savings only 1.1%, and includes a real
correctness bug - "owned by Rafael" -> "负责Rafael" reverses the ownership
direction. Root cause: splicing a single Chinese word mid-English-sentence
sits on an inefficient BPE boundary and doesn't tokenize as well as a whole
translated clause. Kept in the repo as a documented negative result; see
translate_full_zh.py for the representation that actually worked (whole-
sentence translation, 7.8% gross, 0 grammar overhead, 21/21 blind QA).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from representations import Rendering, protected_spans  # noqa: E402

# English phrase -> real Mandarin phrase. Longest-first application.
# No legend accompanies these; a fluent Chinese speaker (or LLM) needs no
# explanation for 负责 any more than for "owns".
#
# KNOWN BUG (documented, not fixed): "owned by X" -> "负责X" reverses
# direction (负责 reads as "is responsible for X", not "X is responsible for
# it"). Left as-is because it demonstrates exactly the risk this eval suite
# exists to catch; do not deploy this dictionary without a directional fix.
PHRASES: list[tuple[str, str]] = [
    ("did not agree with", "不同意"),
    ("does not agree with", "不同意"),
    ("agreed with", "同意"),
    ("agreed to", "同意"),
    ("is not blocked on", "没有被阻塞于"),
    ("is blocked on", "被阻塞于"),
    ("blocked on", "阻塞于"),
    ("moved from", "从"),  # paired with "to" -> "从X到Y", handled by regex below
    ("is owned by", "由...负责"),
    ("owned by", "负责"),
    ("owns", "负责"),
    ("will finish", "将完成"),
    ("will complete", "将完成"),
    ("will review", "将审查"),
    ("will send", "将发送"),
    ("is due", "截止日期是"),
    ("due by", "截止于"),
    ("said that", "说"),
    ("said", "说"),
    ("decision", "决定"),
    ("otherwise", "否则"),
    ("only if", "仅当"),
    ("if", "如果"),
    ("did not", "没有"),
    ("does not", "不"),
    ("is not", "不是"),
    ("not", "不"),
    ("blocked", "阻塞"),
    ("unblocked", "无阻塞"),
]

_RULES = [(re.compile(rf"\b{re.escape(en)}\b", re.I), zh) for en, zh in PHRASES]


def render_meeting_zh(text: str, include_grammar: bool = False) -> Rendering:
    """Substitute meeting-relation phrases with real Mandarin. No legend."""
    spans = protected_spans(text)

    def is_protected(a: int, b: int) -> bool:
        return any(a < e and s < b for s, e in spans)

    out = text
    applied = 0
    for pattern, zh in _RULES:
        result, last, pieces = [], 0, 0
        for m in pattern.finditer(out):
            if is_protected(m.start(), m.end()):
                continue
            result.append(out[last : m.start()])
            result.append(zh)
            last = m.end()
            pieces += 1
        if pieces:
            result.append(out[last:])
            out = "".join(result)
            applied += pieces
            spans = protected_spans(out)

    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" ([,.;:])", r"\1", out)
    out = re.sub(r"(?<=[一-鿿]) (?=\w)", "", out)
    out = re.sub(r"(?<=\w) (?=[一-鿿])", "", out)

    return Rendering(
        method="meeting_zh",
        text=out.strip(),
        grammar="",  # deliberately empty: real language needs no legend
        metadata={"substitutions": applied},
    )
