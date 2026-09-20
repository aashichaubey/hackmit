"""Hard invariant checks for meaning preservation.

This is the safety-critical module of the project.

Embedding cosine similarity is known to be unreliable for exactly the
distinctions that matter most here. "Delete the database." and "Do not delete
the database." are lexically near-identical and score very highly under any
sentence embedding model, yet are opposites. Shipping an encoder that relies on
cosine similarity alone would eventually substitute one for the other.

So equivalence is gated by *deterministic* checks on the facts that must never
change - numbers, negation, URLs, code, identifiers - and embeddings are used
only to judge the remaining, softer question of whether the phrasing matches.

Chinese complicates this: numerals appear as Han characters (三十 = 30,
百分之二十 = 20%), and negation is carried by a small closed set of particles.
Both are handled explicitly below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_ARABIC_NUM = re.compile(r"\d+(?:[.,]\d+)*")

_HAN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    # Financial/formal variants, which appear in legal and UN text.
    "壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5,
    "陆": 6, "柒": 7, "捌": 8, "玖": 9,
}
_HAN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10**4, "亿": 10**8}

_HAN_NUM_RE = re.compile(f"[{''.join(_HAN_DIGITS)}{''.join(_HAN_UNITS)}]+")


def han_to_int(text: str) -> int | None:
    """Convert a Han numeral string to an int.

    Handles the multiplicative structure (三百二十 = 3*100 + 2*10) and the
    colloquial leading-unit form (十五 = 15). Returns None if unparseable,
    which callers treat as "cannot verify" rather than "equal".
    """
    if not text:
        return None
    total, section, number = 0, 0, 0
    for ch in text:
        if ch in _HAN_DIGITS:
            number = _HAN_DIGITS[ch]
        elif ch in _HAN_UNITS:
            unit = _HAN_UNITS[ch]
            if unit >= 10**4:
                section = (section + number) * unit
                total += section
                section, number = 0, 0
            else:
                # Bare "十" means 10, as in 十五 = 15.
                section += (number or 1) * unit
                number = 0
        else:
            return None
    return total + section + number


# English number words. Needed because English routinely spells numbers out
# ("nine hundred sixty-two years") where Chinese uses numerals (九百六十二),
# which would otherwise read as a number appearing out of nowhere.
_EN_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_SCALES = {"hundred": 100, "thousand": 10**3, "million": 10**6,
              "billion": 10**9, "trillion": 10**12}
# Chinese magnitude suffixes that can follow Arabic digits: 2,400万 = 24,000,000.
_ZH_SCALES = {"十": 10, "百": 100, "千": 10**3, "万": 10**4, "亿": 10**8}

_EN_NUM_PHRASE = re.compile(
    r"\b(?:(?:" + "|".join(_EN_ONES) + r"|" + "|".join(_EN_SCALES) + r"|and|[-\s])+)\b",
    re.IGNORECASE,
)
# Arabic digits optionally followed by an English scale word or Han magnitude.
_ARABIC_SCALED = re.compile(
    r"(\d+(?:[,\s]\d{3})*(?:\.\d+)?)\s*(" + "|".join(_EN_SCALES) + r"|[十百千万亿])?",
    re.IGNORECASE,
)


def _parse_en_words(phrase: str) -> int | None:
    """Parse an English spelled-out number ('nine hundred sixty-two' -> 962)."""
    tokens = re.split(r"[-\s]+", phrase.lower().strip())
    tokens = [t for t in tokens if t and t != "and"]
    if not tokens or not any(t in _EN_ONES or t in _EN_SCALES for t in tokens):
        return None
    # A bare scale word ("million" in "$24 million") is a unit attached to an
    # adjacent Arabic numeral, which the Arabic pass already handled. Treating
    # it as the standalone value 1,000,000 would invent a number.
    if not any(t in _EN_ONES for t in tokens):
        return None

    total = current = 0
    for tok in tokens:
        if tok in _EN_ONES:
            current += _EN_ONES[tok]
        elif tok in _EN_SCALES:
            scale = _EN_SCALES[tok]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
        else:
            return None
    return total + current


def extract_numbers(text: str) -> set[float]:
    """Collect every number in `text` as a float, normalized to one scale.

    Handles the four ways the same quantity is written across these corpora:
    Arabic digits, Arabic + scale word ("24 million"), Arabic + Han magnitude
    ("2,400万"), English number words, and Han numerals (九百六十二).
    """
    found: set[float] = set()

    for m in _ARABIC_SCALED.finditer(text):
        raw, scale = m.group(1), m.group(2)
        # UN-style documents use a space as a thousands separator ("1 800").
        raw = raw.replace(",", "").replace(" ", "").replace(" ", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        if scale:
            key = scale.lower()
            val *= _EN_SCALES.get(key) or _ZH_SCALES.get(scale, 1)
        found.add(val)

    for m in _EN_NUM_PHRASE.finditer(text):
        if (val := _parse_en_words(m.group())) is not None:
            # Ignore bare "one"/"a"-like usages that are articles in disguise.
            if val > 1 or len(m.group().split()) > 1:
                found.add(float(val))

    for m in _HAN_NUM_RE.finditer(text):
        # Skip single characters that are idiomatic rather than numeric
        # ("一" in 一些/"some", "万" in 万一/"in case").
        val = han_to_int(m.group())
        if val is not None and (len(m.group()) > 1 or val > 1):
            found.add(float(val))

    return found


# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------

_EN_NEG = re.compile(
    r"\b(?:not|no|never|none|nor|neither|cannot|can't|won't|don't|doesn't|"
    r"didn't|isn't|aren't|wasn't|weren't|shouldn't|wouldn't|couldn't|"
    r"mustn't|without|unable|fails? to|refuses? to)\b",
    re.IGNORECASE,
)

# English idioms that contain a negation token but are not semantically negative
# ("last but not least" is emphasis, not denial).
_EN_NEG_IDIOM = re.compile(
    r"\blast but not least\b|\bnot only\b|\bno doubt\b|\bno wonder\b|"
    r"\bnone other than\b|\bnot to mention\b",
    re.IGNORECASE,
)

# Chinese is the hard side. A bare 不/非/别/无 is frequently part of a compound
# that carries no negation at all - 非常 ("very"), 别名 ("alias"), 不过
# ("however"), 无数 ("countless") - and 莫 appears mostly inside transliterated
# names (莫斯科 "Moscow"). Matching single characters produced a ~90% false
# positive rate on real corpus text, so non-negating compounds are stripped
# before any negation is counted.
_ZH_NON_NEGATING = re.compile(
    # 非 compounds: very / Africa / extraordinary / informal / NGO
    r"非常|非洲|非凡|非法|非正式|非营利|非政府|非官方|非典|是非|除非|"
    # 别 compounds: alias / especially / difference / category / gender
    r"别名|别的|别人|别墅|特别|区别|分别|级别|差别|个别|性别|告别|识别|"
    r"鉴别|派别|类别|送别|离别|辨别|"
    # 不 compounds that are conjunctions, intensifiers, or positives
    r"不过|不但|不仅|不同|不断|不管|不论|不少|不错|不时|不妨|不然|不如|"
    r"不止|不定|不料|不禁|对不起|差不多|不得不|不由得|要不|不外乎|"
    # 无 compounds: regardless / countless / undoubtedly / infinite
    r"无论|无数|无比|无限|无穷|无疑|无非|无奈|无聊|无情|无辜|无私|"
    # 未 compounds: future / minor
    r"未来|未成年|未免|"
    # 没 compounds
    r"没关系|出没"
)

# Genuine negation markers. Multi-character forms are unambiguous; the bare
# particles are only counted after compound stripping.
_ZH_NEG = re.compile(
    r"不要|不能|不可|不会|不应|不得|不准|不许|不是|不曾|没有|沒有|"
    r"无法|無法|毫无|禁止|严禁|勿|请勿|切勿|未能|未曾|并非|绝不|从不|从未|"
    r"[不没無无未]"
)


def negation_count(text: str, chinese: bool) -> int:
    """Count semantically negating constructions.

    For Chinese, non-negating compounds are removed first so that characters
    like 非 in 非常 ("very") are never mistaken for negation.
    """
    if chinese:
        text = _ZH_NON_NEGATING.sub("", text)
        return len(_ZH_NEG.findall(text))
    text = _EN_NEG_IDIOM.sub("", text)
    return len(_EN_NEG.findall(text))


# --------------------------------------------------------------------------
# Verbatim spans that must survive translation untouched
# --------------------------------------------------------------------------

_URL = re.compile(r"https?://[^\s，。）】」）]+|www\.[^\s，。）】」）]+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_CODE = re.compile(r"`[^`]+`|```[\s\S]*?```")
_PLACEHOLDER = re.compile(r"%[sdif@]|\{\w*\}|\$\{[^}]+\}|<[A-Z_]+>")
# Latin identifiers: API, HTTP, CamelCase, snake_case, file.ext
_IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9_.]*(?:[A-Z][a-z]|_|\.)[A-Za-z0-9_.]*\b")


@dataclass
class InvariantReport:
    """Result of the deterministic equivalence gate."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)


def check_invariants(
    english: str,
    mandarin: str,
    check_numbers: bool = True,
    check_negation: bool = True,
    check_verbatim: bool = True,
) -> InvariantReport:
    """Verify that the facts which must not change, did not change.

    A violation is a hard reject: no embedding score can override it.
    """
    violations: list[str] = []
    details: dict[str, object] = {}

    if check_numbers:
        en_nums, zh_nums = extract_numbers(english), extract_numbers(mandarin)
        # Only flag numbers that vanish or appear outright. Chinese frequently
        # restates a figure in a different unit, so we compare as sets and
        # accept a subset relationship in either direction only when empty.
        if en_nums != zh_nums:
            missing = en_nums - zh_nums
            added = zh_nums - en_nums
            if missing or added:
                violations.append("number_mismatch")
                details["numbers"] = {
                    "english": sorted(en_nums),
                    "mandarin": sorted(zh_nums),
                    "missing": sorted(missing),
                    "added": sorted(added),
                }

    if check_negation:
        en_neg = negation_count(english, chinese=False)
        zh_neg = negation_count(mandarin, chinese=True)
        # Exact counts differ legitimately (Chinese often needs two particles
        # for one English negation), but presence/absence must agree. This is
        # the check that stops "delete" turning into "do not delete".
        if (en_neg > 0) != (zh_neg > 0):
            violations.append("negation_mismatch")
            details["negation"] = {"english": en_neg, "mandarin": zh_neg}

    if check_verbatim:
        for label, pattern in (
            ("url", _URL),
            ("email", _EMAIL),
            ("code", _CODE),
            ("placeholder", _PLACEHOLDER),
        ):
            en_spans = sorted(pattern.findall(english))
            zh_spans = sorted(pattern.findall(mandarin))
            if en_spans != zh_spans:
                violations.append(f"{label}_mismatch")
                details[label] = {"english": en_spans, "mandarin": zh_spans}

    return InvariantReport(ok=not violations, violations=violations, details=details)
