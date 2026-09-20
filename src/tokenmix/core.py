"""Exact phrase matching and full-text token accounting; no translation API."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import tiktoken


@dataclass(frozen=True)
class Entry:
    id: str
    en: str
    zh: str
    domain: str = "general"
    source: str = ""
    enabled: bool = False

    def __post_init__(self) -> None:
        for field in ("id", "en", "zh", "domain"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        if not isinstance(self.source, str) or type(self.enabled) is not bool:
            raise ValueError("source must be a string and enabled must be a boolean")


def load_dictionary(path: str | Path) -> list[Entry]:
    entries: list[Entry] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = Entry(**json.loads(line))
            if entry.id in seen:
                raise ValueError(f"duplicate id {entry.id!r}")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        seen.add(entry.id)
        entries.append(entry)
    return entries


@dataclass(frozen=True)
class Change:
    entry_id: str
    start: int
    end: int
    original: str
    replacement: str
    tokens_saved_at_step: int


@dataclass(frozen=True)
class Result:
    original: str
    text: str
    encoding: str
    original_tokens: int
    text_tokens: int
    payload_tokens: int
    prefix: str
    changes: tuple[Change, ...]

    @property
    def saved_tokens(self) -> int:
        return self.original_tokens - self.payload_tokens

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "saved_tokens": self.saved_tokens,
            "savings_percent": round(100 * self.saved_tokens / self.original_tokens, 2)
            if self.original_tokens else 0.0,
            "payload": self.prefix + self.text,
        }


# Deliberately conservative. This is a Markdown/plain-text heuristic, not a
# parser for code, JSON, HTML, or other structured formats.
_PROTECTED = re.compile(
    r"(?m:^[ \t]*(?:`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:^[ \t]*(?:`{3,}|~{3,})[^\n]*(?:\n|$)|\Z))"
    r"|(?P<ticks>`+)[^`\n]*(?P=ticks)"
    r"|https?://[^\s<>\"']+"
    r'|"[^"\n]*"|“[^”\n]*”|「[^」\n]*」|『[^』\n]*』'
    r"|(?<!\w)'[^'\n]*'(?!\w)"
    r"|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"
    r"|\{\{[^}\n]*\}\}|\$\{[^}\n]*\}|\{[A-Za-z_]\w*\}"
    r"|\b\w*_\w+\b"
    r"|\d+(?:[.,]\d+)*"
)


def protected_spans(text: str, literals: Iterable[str] = ()) -> list[tuple[int, int]]:
    spans = [(match.start(), match.end()) for match in _PROTECTED.finditer(text)]
    for literal in literals:
        if not literal:
            raise ValueError("protected literals must not be empty")
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(literal), text))
    return spans


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < right and end > left for left, right in spans)


def _render(original: str, changes: Iterable[Change]) -> str:
    parts: list[str] = []
    cursor = 0
    for change in sorted(changes, key=lambda c: c.start):
        parts.extend((original[cursor:change.start], change.replacement))
        cursor = change.end
    parts.append(original[cursor:])
    return "".join(parts)


def compress(
    text: str,
    entries: Iterable[Entry],
    *,
    encoding: str = "o200k_base",
    domain: str = "general",
    prefix: str = "",
    protect: Iterable[str] = (),
    max_changes: int = 100,
) -> Result:
    """Greedily choose nonoverlapping replacements that save full-payload tokens.

    Exact, case-sensitive matches only. Dictionary equivalence is assumed, not
    proven. Every trial retokenizes the entire candidate with its optional prefix.
    A final original-text fallback accounts for the prefix's cost. This is a
    bounded greedy search, not a globally optimal decoder or semantic validator.
    """
    if max_changes < 0:
        raise ValueError("max_changes must be nonnegative")
    tokenizer = tiktoken.get_encoding(encoding)

    def count(value: str) -> int:
        return len(tokenizer.encode_ordinary(value))

    original_tokens = count(text)
    protected = protected_spans(text, protect)
    active = [e for e in entries if e.enabled and e.domain in ("general", domain)]

    # If a source has multiple possible translations in the selected domain,
    # abstain. A production disambiguator would need the surrounding context.
    translations: dict[str, set[str]] = {}
    for entry in active:
        for source, target in ((entry.en, entry.zh), (entry.zh, entry.en)):
            translations.setdefault(source, set()).add(target)

    candidates: list[Change] = []
    seen: set[tuple[int, int, str]] = set()
    for entry in active:
        for source, target in ((entry.en, entry.zh), (entry.zh, entry.en)):
            if len(translations[source]) != 1 or source == target:
                continue
            # Latin words cannot match inside larger words/identifiers. Chinese
            # terms are matched literally; lexical ambiguity still needs review.
            pattern = re.escape(source)
            if source[0].isascii() and (source[0].isalnum() or source[0] == "_"):
                pattern = r"(?<!\w)" + pattern
            if source[-1].isascii() and (source[-1].isalnum() or source[-1] == "_"):
                pattern += r"(?!\w)"
            # Never introduce or remove literal-bearing structures via a mapping.
            if protected_spans(target):
                continue
            for match in re.finditer(pattern, text):
                key = (match.start(), match.end(), target)
                if key in seen or _overlaps(match.start(), match.end(), protected):
                    continue
                seen.add(key)
                candidates.append(Change(entry.id, match.start(), match.end(), source, target, 0))

    changes: list[Change] = []
    current_tokens = count(prefix + text)
    for _ in range(max_changes):
        best: Change | None = None
        best_tokens = current_tokens
        occupied = [(c.start, c.end) for c in changes]
        for candidate in candidates:
            if _overlaps(candidate.start, candidate.end, occupied):
                continue
            trial_tokens = count(prefix + _render(text, [*changes, candidate]))
            if trial_tokens < best_tokens:
                best, best_tokens = candidate, trial_tokens
        if best is None:
            break
        changes.append(Change(
            best.entry_id, best.start, best.end, best.original, best.replacement,
            current_tokens - best_tokens,
        ))
        current_tokens = best_tokens

    if not changes or current_tokens >= original_tokens:
        return Result(text, text, encoding, original_tokens, original_tokens, original_tokens, "", ())
    compressed = _render(text, changes)
    return Result(text, compressed, encoding, original_tokens, count(compressed), current_tokens, prefix, tuple(changes))
