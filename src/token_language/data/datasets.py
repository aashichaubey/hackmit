"""Dataset assembly: raw corpora -> normalized, filtered, token-measured records."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import ParallelExample, TokenCount
from ..tokenizer.base import Tokenizer
from .alignment import check_pair
from .loaders import REGISTRY, load_corpus
from .normalization import (
    has_traditional,
    normalize_chinese,
    normalize_english,
)

log = logging.getLogger(__name__)


@dataclass
class BuildStats:
    """Per-corpus accounting so filtering is transparent, not silent."""

    corpus: str
    seen: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)

    @property
    def yield_pct(self) -> float:
        return 100.0 * self.kept / self.seen if self.seen else 0.0

    def summary(self) -> str:
        top = ", ".join(f"{r}={n}" for r, n in self.rejected.most_common(4))
        return (
            f"{self.corpus:<12} seen={self.seen:<7,} kept={self.kept:<6,} "
            f"({self.yield_pct:5.1f}%)  top_rejects: {top or 'none'}"
        )


def _stable_id(english: str, mandarin: str) -> str:
    digest = hashlib.sha1(f"{english}\x00{mandarin}".encode()).hexdigest()
    return digest[:16]


def build_examples(
    corpus: str,
    tokenizers: dict[str, Tokenizer],
    limit: int | None = None,
    scan_limit: int | None = None,
    dedupe: bool = True,
) -> tuple[list[ParallelExample], BuildStats]:
    """Load, normalize, filter and token-measure one corpus.

    `limit` caps *kept* examples; `scan_limit` caps raw rows read. Noisy
    corpora need a much larger scan budget than their target yield.
    """
    spec = REGISTRY[corpus]
    stats = BuildStats(corpus=corpus)
    out: list[ParallelExample] = []
    seen_ids: set[str] = set()

    # Read generously when a target yield is requested, since filters are strict.
    if scan_limit is None:
        scan_limit = limit * 40 if limit else None

    for raw_en, raw_zh in load_corpus(corpus, limit=scan_limit):
        stats.seen += 1

        en = normalize_english(raw_en)
        # Corpora flagged Traditional get converted; others are left alone but
        # still normalized, so a stray Traditional row is converted too.
        zh = normalize_chinese(raw_zh, simplify=spec.traditional or has_traditional(raw_zh))

        verdict = check_pair(en, zh)
        if not verdict.keep:
            stats.rejected[verdict.reason] += 1
            continue

        ex_id = _stable_id(en, zh)
        if dedupe:
            if ex_id in seen_ids:
                stats.rejected["duplicate"] += 1
                continue
            seen_ids.add(ex_id)

        ex = ParallelExample(
            id=f"{corpus}_{ex_id}",
            source=corpus,
            domain=spec.domain,
            english=en,
            mandarin=zh,
            alignment_score=round(verdict.score, 4),
        )
        out.append(ex)
        stats.kept += 1

        if limit is not None and stats.kept >= limit:
            break

    measure_tokens(out, tokenizers)
    return out, stats


def measure_tokens(
    examples: list[ParallelExample], tokenizers: dict[str, Tokenizer]
) -> None:
    """Attach token counts for every configured tokenizer, in place.

    Uses batch encoding, which is markedly faster than per-row calls at
    corpus scale.
    """
    if not examples:
        return
    en_texts = [e.english for e in examples]
    zh_texts = [e.mandarin for e in examples]

    for tid, tk in tokenizers.items():
        en_counts = tk.count_many(en_texts)
        zh_counts = tk.count_many(zh_texts)
        for ex, ec, zc in zip(examples, en_counts, zh_counts):
            ex.tokens[tid] = TokenCount(
                tokenizer_id=tid, english_tokens=ec, mandarin_tokens=zc
            )


def write_jsonl(examples: Iterable[ParallelExample], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[ParallelExample]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield ParallelExample.model_validate_json(line)


def split_by_source_document(
    examples: list[ParallelExample], test_fraction: float = 0.3, seed: int = 13
) -> tuple[list[ParallelExample], list[ParallelExample]]:
    """Split train/test by a hash of the example, grouped to avoid leakage.

    Splitting randomly over sentences would let near-duplicate sentences from
    the same document land on both sides, letting the dictionary memorize the
    test set. Hashing a coarse content key keeps related rows together.
    """
    train, test = [], []
    for ex in examples:
        # Bucket on a stable hash so the split is deterministic and reproducible.
        key = f"{seed}:{ex.source}:{ex.id}".encode()
        bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) / 0xFFFFFFFF
        (test if bucket < test_fraction else train).append(ex)
    return train, test
