"""Core data models shared across the pipeline.

Everything that crosses a module boundary is a pydantic model so that the
offline (dictionary-building) and online (encoding) phases cannot silently
disagree about a field name or unit.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Language(str, Enum):
    ENGLISH = "english"
    MANDARIN = "mandarin"


class EquivalenceVerdict(str, Enum):
    """Why a pair was accepted or rejected.

    Kept explicit (rather than a bare bool) because the failure *reason* is the
    interesting research output: an invariant violation is a very different
    signal from merely-low embedding similarity.
    """

    EQUIVALENT = "equivalent"
    LOW_SIMILARITY = "low_similarity"
    INVARIANT_VIOLATION = "invariant_violation"
    JUDGE_REJECTED = "judge_rejected"
    NOT_EVALUATED = "not_evaluated"


class TokenCount(BaseModel):
    """Token counts for one pair under one specific tokenizer.

    `tokenizer_id` is mandatory: a token count is meaningless without knowing
    which tokenizer produced it, and mixing them is the single easiest way to
    fabricate a result.
    """

    tokenizer_id: str
    english_tokens: int = Field(ge=0)
    mandarin_tokens: int = Field(ge=0)

    @property
    def preferred_language(self) -> Language:
        return (
            Language.MANDARIN
            if self.mandarin_tokens < self.english_tokens
            else Language.ENGLISH
        )

    @property
    def token_savings(self) -> int:
        """Tokens saved by picking the cheaper side. Never negative."""
        return abs(self.english_tokens - self.mandarin_tokens)

    @property
    def savings_percent(self) -> float:
        """Savings relative to the English baseline."""
        if self.english_tokens == 0:
            return 0.0
        cheaper = min(self.english_tokens, self.mandarin_tokens)
        return 100.0 * (self.english_tokens - cheaper) / self.english_tokens


class ParallelExample(BaseModel):
    """One aligned English/Mandarin sentence pair."""

    id: str
    source: str
    domain: str
    english: str
    mandarin: str

    alignment_score: float | None = None
    semantic_score: float | None = None
    verdict: EquivalenceVerdict = EquivalenceVerdict.NOT_EVALUATED
    violations: list[str] = Field(default_factory=list)

    # tokenizer_id -> counts. Multiple tokenizers can be measured on one example.
    tokens: dict[str, TokenCount] = Field(default_factory=dict)

    def counts_for(self, tokenizer_id: str) -> TokenCount | None:
        return self.tokens.get(tokenizer_id)


class DictionaryEntry(BaseModel):
    """A validated, token-measured phrase correspondence."""

    concept_id: str
    english: str
    mandarin: str
    english_tokens: int
    mandarin_tokens: int
    tokenizer_id: str
    semantic_score: float
    frequency: int = 1
    domains: list[str] = Field(default_factory=list)

    @property
    def preferred(self) -> Language:
        return (
            Language.MANDARIN
            if self.mandarin_tokens < self.english_tokens
            else Language.ENGLISH
        )

    @property
    def savings(self) -> int:
        """Tokens saved by substituting the cheaper side for the English side.

        Zero when English already wins - such entries are kept in the dictionary
        (they are evidence for research question #3) but are never substituted.
        """
        return max(0, self.english_tokens - self.mandarin_tokens)

    @model_validator(mode="after")
    def _non_empty(self) -> DictionaryEntry:
        if not self.english.strip() or not self.mandarin.strip():
            raise ValueError("dictionary entry sides must both be non-empty")
        return self


class Replacement(BaseModel):
    """A single span substitution made by the encoder."""

    start: int
    end: int
    original: str
    replacement: str
    language: Language
    tokens_before: int
    tokens_after: int
    semantic_score: float
    concept_id: str

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


class EncodingResult(BaseModel):
    """Everything the online phase produces for one input."""

    original_text: str
    encoded_text: str
    original_tokens: int
    encoded_tokens: int
    tokenizer_id: str
    replacements: list[Replacement] = Field(default_factory=list)
    semantic_confidence: float = 1.0

    # Overhead accounting. `instruction_tokens` is the cost of any system
    # preamble required to make the hybrid intelligible to the model.
    instruction_tokens: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.encoded_tokens

    @property
    def savings_percent(self) -> float:
        """Raw savings, ignoring instruction overhead."""
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * self.tokens_saved / self.original_tokens

    @property
    def net_tokens(self) -> int:
        return self.encoded_tokens + self.instruction_tokens

    @property
    def net_savings_percent(self) -> float:
        """Savings after paying for the decoding instruction.

        This is the number that actually matters, and it can be negative.
        Reporting only `savings_percent` would overstate the result.
        """
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * (self.original_tokens - self.net_tokens) / self.original_tokens


Side = Literal["english", "mandarin"]
