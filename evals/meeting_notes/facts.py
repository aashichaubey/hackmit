"""Layer 1: deterministic factual checks.

Verifies that each *planted* fact remains recoverable from a representation.
Deliberately mechanical - no model is involved - because the whole point of
this layer is to catch cases an LLM judge would wave through.

Recoverability rule: a fact value is recoverable if its surface form appears
in the representation, OR it is reachable through the representation's
declared grammar/legend. Numbers, dates, names and percentages must match
exactly; `13` does not satisfy `30`, and `Alice` does not satisfy `Alicia`.

Negation is checked separately and structurally: a representation that drops
or adds a negation is marked corrupt even when every other value survives,
because that is the failure mode most likely to pass unnoticed.

LIMITATION (found during live testing): this checker does literal ENGLISH
substring matching. It cannot validate a representation that is genuinely
translated into another language (e.g. a full-sentence Mandarin translation),
because a translated fact will not substring-match its English planted value.
For such representations, treat this layer's score as a lower bound and cross-
check with downstream QA (Layer 2/4), which is language-agnostic because it
grades the model's actual answer rather than the representation's text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Values of these fact fields must survive verbatim.
_VALUE_FIELDS = (
    "value", "person", "object", "deadline", "action", "from_value",
    "to_value", "between", "and_", "topic", "blocked_on", "depends_on",
    "condition", "consequence", "speaker", "about", "project",
    "statement",
)
# NOTE: `metric` is deliberately absent. It is a human-readable *label* for
# what a number measures ("outage duration initial"), not a span that appears
# in the note. Requiring it made the raw control score 84.9% instead of 100%,
# which would have silently inflated every representation's apparent loss.

# Fact types where an exact-match failure is a severe corruption.
CRITICAL_TYPES = {
    "number", "percentage", "date", "deadline", "person", "owner",
    "negation", "status_change", "action_item",
}

_NEG_EN = re.compile(
    r"\b(?:not|no|never|didn't|doesn't|isn't|won't|cannot|can't)\b", re.I
)
# Notation / Mandarin negation markers used by the compact representations.
_NEG_SYM = re.compile(r"[否不没无未]")


def _normalize(s: str) -> str:
    """Lowercase and collapse whitespace. Does NOT alter digits or letters."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _resolve(text: str, grammar: str) -> str:
    """Text plus anything the grammar makes reachable.

    The legend is appended so a value spelled out only in the legend
    (e.g. `责 = is responsible for`) still counts as recoverable.
    """
    return _normalize(text + "\n" + grammar)


@dataclass
class FactResult:
    fact: dict
    recovered: bool
    missing: list[str] = field(default_factory=list)
    critical: bool = False


@dataclass
class FactReport:
    total: int
    recovered: int
    results: list[FactResult]
    negation_preserved: bool
    negation_detail: str = ""

    @property
    def accuracy(self) -> float:
        return self.recovered / self.total if self.total else 0.0

    @property
    def critical_failures(self) -> int:
        return sum(1 for r in self.results if not r.recovered and r.critical)


def check_facts(facts: list[dict], text: str, grammar: str = "",
                original: str | None = None) -> FactReport:
    """Check every planted fact against one representation."""
    haystack = _resolve(text, grammar)
    results: list[FactResult] = []

    for fact in facts:
        missing: list[str] = []
        for key in _VALUE_FIELDS:
            val = fact.get(key)
            if not val or not isinstance(val, str):
                continue
            # A multi-word value counts as recovered if every content word
            # survives; word order may legitimately change under compression.
            words = [w for w in re.findall(r"[\w$%.]+", val.lower()) if w]
            if not words:
                continue
            for w in words:
                # Exact token match, so 13 never satisfies 30 and
                # Alice never satisfies Alicia.
                if not re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", haystack):
                    missing.append(f"{key}:{val}")
                    break
        results.append(FactResult(
            fact=fact,
            recovered=not missing,
            missing=sorted(set(missing)),
            critical=fact.get("type") in CRITICAL_TYPES,
        ))

    neg_ok, detail = True, ""
    if original is not None:
        neg_ok, detail = _check_negation(original, text, grammar)

    return FactReport(
        total=len(results),
        recovered=sum(r.recovered for r in results),
        results=results,
        negation_preserved=neg_ok,
        negation_detail=detail,
    )


def _check_negation(original: str, text: str, grammar: str) -> tuple[bool, str]:
    """Negation count must not change between original and representation."""
    orig_n = len(_NEG_EN.findall(original))
    # The compact forms may encode negation as a symbol rather than a word.
    rep_n = len(_NEG_EN.findall(text)) + len(_NEG_SYM.findall(text))
    if orig_n == 0 and rep_n == 0:
        return True, ""
    if orig_n > 0 and rep_n == 0:
        return False, f"negation DROPPED (original had {orig_n}, representation has 0)"
    if orig_n == 0 and rep_n > 0:
        return False, f"negation INTRODUCED (original had 0, representation has {rep_n})"
    return True, f"negation present in both ({orig_n} -> {rep_n})"
