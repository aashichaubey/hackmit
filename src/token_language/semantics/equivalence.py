"""Semantic equivalence gate.

Combines the two complementary signals:

  1. `invariants`  - deterministic, catches negation/number/URL corruption.
  2. `embeddings`  - graded, catches "fluent but says something else".

Order matters. Invariants run first and are absolute: a pair that changes a
number is rejected no matter how high it scores under the embedding model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..schema import EquivalenceVerdict, ParallelExample
from .embeddings import EmbeddingModel
from .invariants import check_invariants

log = logging.getLogger(__name__)

# Chosen empirically: LaBSE scores genuine en-zh translation pairs well above
# this, while misaligned OPUS rows fall below it. Configurable per experiment.
DEFAULT_THRESHOLD = 0.75


@dataclass
class EquivalenceConfig:
    threshold: float = DEFAULT_THRESHOLD
    check_numbers: bool = True
    check_negation: bool = True
    check_verbatim: bool = True
    #: Route borderline pairs to an LLM judge (costs API calls).
    use_llm_judge: bool = False
    #: Band around the threshold considered "borderline".
    judge_margin: float = 0.05


def evaluate_examples(
    examples: list[ParallelExample],
    embedder: EmbeddingModel,
    config: EquivalenceConfig | None = None,
    judge=None,
) -> list[ParallelExample]:
    """Score and label every example in place, returning the same list."""
    cfg = config or EquivalenceConfig()
    if not examples:
        return examples

    # Invariants first - cheap, deterministic, and absolute.
    hard_failed: list[int] = []
    for i, ex in enumerate(examples):
        report = check_invariants(
            ex.english,
            ex.mandarin,
            check_numbers=cfg.check_numbers,
            check_negation=cfg.check_negation,
            check_verbatim=cfg.check_verbatim,
        )
        if not report.ok:
            ex.verdict = EquivalenceVerdict.INVARIANT_VIOLATION
            ex.violations = report.violations
            hard_failed.append(i)

    # Embed everything (including hard failures) so the report can show that
    # invariant violations often score *highly* - the key argument for why
    # embeddings alone are insufficient.
    scores = embedder.similarity(
        [e.english for e in examples], [e.mandarin for e in examples]
    )
    for ex, s in zip(examples, scores):
        ex.semantic_score = round(float(s), 4)

    borderline: list[ParallelExample] = []
    for i, ex in enumerate(examples):
        if ex.verdict == EquivalenceVerdict.INVARIANT_VIOLATION:
            continue
        if ex.semantic_score < cfg.threshold:
            ex.verdict = EquivalenceVerdict.LOW_SIMILARITY
        else:
            ex.verdict = EquivalenceVerdict.EQUIVALENT
            if cfg.use_llm_judge and ex.semantic_score < cfg.threshold + cfg.judge_margin:
                borderline.append(ex)

    if borderline and judge is not None:
        log.info("routing %d borderline pairs to the LLM judge", len(borderline))
        for ex, verdict in zip(borderline, judge.judge_batch(borderline)):
            if not verdict.equivalent:
                ex.verdict = EquivalenceVerdict.JUDGE_REJECTED
                ex.violations = list(verdict.issues)

    return examples


def accepted(examples: list[ParallelExample]) -> list[ParallelExample]:
    return [e for e in examples if e.verdict == EquivalenceVerdict.EQUIVALENT]
