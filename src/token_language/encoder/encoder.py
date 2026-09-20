"""Token-aware hybrid encoder.

Replaces English spans with Mandarin ones only when doing so reduces TARGET
tokens. The output is deliberately mixed-language: where English is already
cheaper, English stays.

Matching is greedy longest-first over non-overlapping spans. Each candidate
substitution is re-measured in context rather than trusted from the dictionary,
because BPE merges across span boundaries mean an entry that saves tokens in
isolation can cost tokens in situ.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..schema import DictionaryEntry, EncodingResult, Language, Replacement
from ..tokenizer.base import Tokenizer
from ..tokenizer.hf import HFTokenizer


def _boundary_pattern(phrase: str) -> re.Pattern[str]:
    """Match the phrase on word boundaries, tolerant of whitespace runs."""
    parts = [re.escape(tok) for tok in phrase.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


class Encoder:
    """Greedy, token-verified hybrid encoder."""

    def __init__(self, entries: list[DictionaryEntry], tokenizer: Tokenizer,
                 dictionary_version: str = "unknown") -> None:
        # Longest first: prefer specific multi-word phrases over their parts.
        self.entries = sorted(entries, key=lambda e: -len(e.english))
        self._patterns = [(e, _boundary_pattern(e.english)) for e in self.entries]
        self.tokenizer = tokenizer
        self.dictionary_version = dictionary_version

    @classmethod
    def from_dictionary(cls, path: str | Path, tokenizer: Tokenizer | None = None,
                        tokenizer_model: str = "deepseek-v4-flash") -> Encoder:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"dictionary not found: {p}. Build it with "
                "`python -m scripts.build_dictionary`."
            )
        entries = [
            DictionaryEntry.model_validate_json(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tk = tokenizer or HFTokenizer.from_pretrained(tokenizer_model)

        version = "unknown"
        meta = p.with_suffix(".meta.json")
        if meta.exists():
            version = json.loads(meta.read_text(encoding="utf-8")).get(
                "dictionary_version", "unknown")

        # Only entries measured against THIS tokenizer are trustworthy.
        usable = [e for e in entries if e.tokenizer_id == tk.id]
        return cls(usable, tk, dictionary_version=version)

    def encode(self, text: str, protect: list[tuple[int, int]] | None = None) -> EncodingResult:
        """Encode `text`, optionally leaving `protect` character ranges untouched."""
        original_tokens = self.tokenizer.count_tokens(text)
        protect = protect or []
        taken: list[tuple[int, int]] = list(protect)
        replacements: list[Replacement] = []

        def overlaps(a: int, b: int) -> bool:
            return any(a < e and s < b for s, e in taken)

        # Pass 1: collect non-overlapping candidate spans.
        candidates = []
        for entry, pattern in self._patterns:
            for m in pattern.finditer(text):
                if overlaps(m.start(), m.end()):
                    continue
                taken.append((m.start(), m.end()))
                candidates.append((m.start(), m.end(), entry, m.group()))

        candidates.sort(key=lambda c: c[0])

        # Pass 2: apply, verifying each substitution actually saves tokens in
        # context. Applied right-to-left so earlier offsets stay valid.
        out = text
        for start, end, entry, matched in reversed(candidates):
            trial = out[:start] + entry.mandarin + out[end:]
            if self.tokenizer.count_tokens(trial) >= self.tokenizer.count_tokens(out):
                continue  # no in-context saving; keep the English
            before = self.tokenizer.count_tokens(matched)
            after = self.tokenizer.count_tokens(entry.mandarin)
            out = trial
            replacements.append(Replacement(
                start=start, end=end, original=matched, replacement=entry.mandarin,
                language=Language.MANDARIN, tokens_before=before, tokens_after=after,
                semantic_score=entry.semantic_score, concept_id=entry.concept_id,
            ))

        replacements.reverse()
        confidence = (
            min((r.semantic_score for r in replacements), default=1.0)
        )
        return EncodingResult(
            original_text=text,
            encoded_text=out,
            original_tokens=original_tokens,
            encoded_tokens=self.tokenizer.count_tokens(out),
            tokenizer_id=self.tokenizer.id,
            replacements=replacements,
            semantic_confidence=confidence,
        )

    def decoder_preamble(self, replacements: list[Replacement]) -> str:
        """Codebook covering exactly the entries used. Empty when none were."""
        if not replacements:
            return ""
        seen: dict[str, str] = {}
        by_id = {e.concept_id: e for e in self.entries}
        for r in replacements:
            entry = by_id.get(r.concept_id)
            if entry:
                seen[entry.mandarin] = entry.english
        lines = "\n".join(f"{zh} = {en}" for zh, en in seen.items())
        return (
            "Some phrases below are written in Chinese as shorthand. "
            "They mean exactly the English on the right:\n" + lines
        )
