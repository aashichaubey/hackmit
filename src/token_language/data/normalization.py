"""Text normalization.

This module exists because of a measurement hazard, not for cosmetics.

Several OPUS corpora (KDE4 most notably) ship *pre-tokenized* Chinese, with
spaces injected around every CJK character group and punctuation mark:

    "生成图像指纹将花费一段时间 。 您想要哪个 ？"

Feeding that to a BPE tokenizer inflates the Chinese token count substantially,
because each injected space breaks a merge that would otherwise apply. Left
uncorrected it would make Mandarin look artificially expensive and bias the
entire study toward English. Normalization is therefore a *correctness*
requirement for any token measurement, and is applied before counting.

Separately, some corpora (tldr-pages) ship Traditional Chinese while the target
register is Simplified; mixing scripts pollutes both the dictionary and the
token statistics.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# CJK ideographs, plus the fullwidth punctuation block used in Chinese text.
_CJK = r"㐀-䶿一-鿿豈-﫿"
_CJK_PUNCT = r"　-〿＀-￯"

# A space sandwiched between two CJK-ish characters is always an artifact:
# Chinese does not use inter-word spaces.
_SPACE_BETWEEN_CJK = re.compile(f"(?<=[{_CJK}{_CJK_PUNCT}]) +(?=[{_CJK}{_CJK_PUNCT}])")
# A space before fullwidth punctuation ("时间 。") is likewise an artifact.
_SPACE_BEFORE_CJK_PUNCT = re.compile(f" +(?=[{_CJK_PUNCT}])")
# ...as is one after an opening fullwidth bracket/quote.
_SPACE_AFTER_CJK_PUNCT = re.compile(f"(?<=[〈-〛（｛]) +")

_MULTISPACE = re.compile(r"[ \t ]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_cjk_spaces(text: str) -> str:
    """Remove tokenization-artifact spaces from Chinese text.

    Only touches spaces adjacent to CJK characters, so embedded Latin runs
    ("使用 API 密钥") keep their legitimate separating spaces.
    """
    text = _SPACE_BETWEEN_CJK.sub("", text)
    text = _SPACE_BEFORE_CJK_PUNCT.sub("", text)
    text = _SPACE_AFTER_CJK_PUNCT.sub("", text)
    return text


@lru_cache(maxsize=1)
def _converter():
    """Lazily build the OpenCC Traditional->Simplified converter."""
    try:
        import opencc

        return opencc.OpenCC("t2s")
    except ImportError:  # pragma: no cover - optional dependency
        return None


def to_simplified(text: str) -> str:
    """Convert Traditional Chinese to Simplified.

    Returns the input unchanged if OpenCC is unavailable, so the pipeline
    degrades rather than crashing; `has_traditional` lets callers detect and
    drop such rows instead.
    """
    conv = _converter()
    return conv.convert(text) if conv is not None else text


# Characters that exist only in Traditional orthography. Not exhaustive - this
# is a cheap detector used to flag corpora, not to validate individual strings.
_TRADITIONAL_MARKERS = set(
    "啟資料透過設定為個們這來時後開關閉點擊選擇檔輸單" "與對從體當態網絡務準確靜態語當歸總結構響應級"
)


def has_traditional(text: str) -> bool:
    """Heuristic: does this text contain Traditional-only characters?"""
    return any(ch in _TRADITIONAL_MARKERS for ch in text)


def normalize_chinese(text: str, simplify: bool = True) -> str:
    """Full normalization for the Mandarin side.

    Uses NFC, not NFKC. NFKC folds fullwidth punctuation to ASCII ("？" -> "?"),
    but fullwidth punctuation is correct Chinese orthography and is what a real
    user would actually send to the model. Folding it would change the token
    count away from the real-world case we are trying to measure.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = strip_cjk_spaces(text)
    if simplify:
        text = to_simplified(text)
    return _MULTISPACE.sub(" ", text).strip()


def normalize_english(text: str) -> str:
    """Full normalization for the English side.

    NFKC is deliberately *not* applied here: it would rewrite characters like
    the micro sign or ligatures and change token counts for reasons unrelated
    to the hypothesis. We only strip control characters and collapse spacing.
    """
    text = _CONTROL.sub("", text)
    text = _fix_moses_detok(text)
    return _MULTISPACE.sub(" ", text).strip()


# Moses-style corpora separate punctuation with spaces on the English side too
# ("server , so ." -> "server, so."). Same inflation problem, smaller magnitude.
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?%)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{$]) +")
_DETOK_APOSTROPHE = re.compile(r" +' *(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)


def _fix_moses_detok(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _DETOK_APOSTROPHE.sub(r"'\1", text)
    return text


def cjk_ratio(text: str) -> float:
    """Fraction of non-space characters that are CJK ideographs.

    Used to sanity-check that the 'mandarin' column actually contains Chinese
    and the 'english' column does not.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    pattern = re.compile(f"[{_CJK}]")
    return sum(bool(pattern.match(c)) for c in chars) / len(chars)
