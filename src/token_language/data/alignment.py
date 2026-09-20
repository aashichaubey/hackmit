"""Alignment quality filtering.

Parallel corpora contain misaligned rows, boilerplate, and rows that are not
actually bilingual. Admitting these into the dictionary would produce confident
substitutions that silently change meaning, so filtering is a safety mechanism
rather than a tidiness one.

These are *cheap structural* checks. Semantic equivalence is handled separately
in `semantics/` - the two are complementary: structural checks catch "this row
is garbage", semantic checks catch "this row is fluent but says something else".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalization import cjk_ratio

# Ratio of English chars to Chinese chars for a faithful translation. Chinese
# is far denser per character, so a legitimate pair sits roughly in 1.2-6.0.
# Outside that band the rows are almost always mismatched in length.
MIN_CHAR_RATIO = 1.0
MAX_CHAR_RATIO = 8.0

_URL = re.compile(r"https?://\S+|www\.\S+")
_PLACEHOLDER = re.compile(r"%[sdif@]|\{\d*\}|\$\{[^}]+\}|@[A-Z_]+@|&[a-z]+;")
# KDE/GNOME metadata rows that are not natural language at all.
_UI_METADATA = re.compile(
    r"ROLES_OF_TRANSLATORS|EMAIL OF TRANSLATORS|CREDIT_FOR_TRANSLATORS|"
    r"Your names|Your emails|^@[a-z]+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FilterResult:
    keep: bool
    reason: str = ""
    score: float = 1.0


def check_pair(
    english: str,
    mandarin: str,
    min_en_chars: int = 15,
    max_en_chars: int = 400,
    min_cjk: float = 0.25,
    max_en_cjk: float = 0.05,
) -> FilterResult:
    """Structural validity check for one aligned pair.

    Returns the first failure reason, which makes the rejection histogram in
    the dataset report directly interpretable.
    """
    en, zh = english.strip(), mandarin.strip()

    if not en or not zh:
        return FilterResult(False, "empty_side")

    if len(en) < min_en_chars:
        return FilterResult(False, "too_short")
    if len(en) > max_en_chars:
        return FilterResult(False, "too_long")

    # The Mandarin column must actually be Chinese, and the English column must
    # not be. Catches column swaps and untranslated copy-through rows.
    zh_ratio = cjk_ratio(zh)
    if zh_ratio < min_cjk:
        return FilterResult(False, "mandarin_not_chinese", zh_ratio)
    if cjk_ratio(en) > max_en_cjk:
        return FilterResult(False, "english_contains_chinese")

    # Identical sides mean the row was never translated.
    if en == zh:
        return FilterResult(False, "untranslated")

    if _UI_METADATA.search(en) or _UI_METADATA.search(zh):
        return FilterResult(False, "ui_metadata")

    # Format placeholders must survive translation; a mismatch means the row
    # is not a faithful pair and would teach the encoder to drop them.
    if sorted(_PLACEHOLDER.findall(en)) != sorted(_PLACEHOLDER.findall(zh)):
        return FilterResult(False, "placeholder_mismatch")

    # URLs must be preserved exactly.
    if sorted(_URL.findall(en)) != sorted(_URL.findall(zh)):
        return FilterResult(False, "url_mismatch")

    ratio = len(en) / max(len(zh), 1)
    if not (MIN_CHAR_RATIO <= ratio <= MAX_CHAR_RATIO):
        return FilterResult(False, "length_ratio", ratio)

    # Map the ratio to a soft 0-1 alignment score peaking at the typical ~2.5.
    score = _ratio_score(ratio)
    return FilterResult(True, "", score)


def _ratio_score(ratio: float, ideal: float = 2.5, width: float = 2.0) -> float:
    """Gaussian-ish confidence that a length ratio indicates good alignment."""
    return float(max(0.0, min(1.0, 1.0 - ((ratio - ideal) / width) ** 2 * 0.5)))
