"""Phase 1 proof-of-concept report.

Answers the founding question before any encoder is built: for the target
tokenizer, how often and by how much is one language cheaper than the other,
and does the cheaper side actually preserve meaning?

    python -m scripts.report_phase1 --examples data/processed/examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_language.data.datasets import read_jsonl, write_jsonl  # noqa: E402
from token_language.schema import EquivalenceVerdict, ParallelExample  # noqa: E402
from token_language.semantics.embeddings import EmbeddingModel  # noqa: E402
from token_language.semantics.equivalence import (  # noqa: E402
    EquivalenceConfig,
    evaluate_examples,
)

log = logging.getLogger("report_phase1")

# Below this many tokens, a difference is noise rather than a usable signal.
NEGLIGIBLE = 1


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def summarize(examples: list[ParallelExample], tid: str) -> dict:
    """Token statistics for one tokenizer over `examples`."""
    rows = [(e, e.tokens[tid]) for e in examples if tid in e.tokens]
    if not rows:
        return {}

    en = [t.english_tokens for _, t in rows]
    zh = [t.mandarin_tokens for _, t in rows]
    # The hybrid oracle: per sentence, take whichever side is cheaper.
    best = [min(a, b) for a, b in zip(en, zh)]

    zh_wins = sum(b - a > NEGLIGIBLE for a, b in zip(zh, en))
    en_wins = sum(a - b > NEGLIGIBLE for a, b in zip(zh, en))
    ties = len(rows) - zh_wins - en_wins

    def dist(v: list[int]) -> dict:
        s = sorted(v)
        return {
            "total": sum(v),
            "mean": round(st.mean(v), 2),
            "median": round(st.median(v), 1),
            "p90": s[int(0.9 * (len(s) - 1))],
        }

    return {
        "n": len(rows),
        "english": dist(en),
        "mandarin": dist(zh),
        "oracle_hybrid": dist(best),
        # Whole-sentence Mandarin translation (Baseline B).
        "full_mandarin_reduction_pct": round(pct(sum(en) - sum(zh), sum(en)), 2),
        # Per-sentence oracle selection (upper bound for Baseline C).
        "oracle_reduction_pct": round(pct(sum(en) - sum(best), sum(en)), 2),
        "mandarin_wins": zh_wins,
        "english_wins": en_wins,
        "negligible": ties,
        "mandarin_win_pct": round(pct(zh_wins, len(rows)), 1),
        "english_win_pct": round(pct(en_wins, len(rows)), 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples", default="data/processed/examples.jsonl")
    p.add_argument("--primary", default="deepseek-v4-flash")
    p.add_argument("--embedding-model", default="labse")
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--out", default="experiments/results/phase1")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    examples = list(read_jsonl(args.examples))
    if not examples:
        log.error("no examples found at %s", args.examples)
        return 1
    log.info("loaded %d examples", len(examples))

    with EmbeddingModel(args.embedding_model) as embedder:
        evaluate_examples(
            examples, embedder, EquivalenceConfig(threshold=args.threshold)
        )

    tokenizer_ids = sorted({t for e in examples for t in e.tokens})
    kept = [e for e in examples if e.verdict == EquivalenceVerdict.EQUIVALENT]

    report = {
        "n_examples": len(examples),
        "embedding_model": args.embedding_model,
        "semantic_threshold": args.threshold,
        "primary_tokenizer": args.primary,
        "verdicts": {
            v.value: sum(e.verdict == v for e in examples) for v in EquivalenceVerdict
        },
        "n_semantically_valid": len(kept),
        "by_tokenizer": {t: summarize(kept, t) for t in tokenizer_ids},
        "by_domain": {},
        "by_length_bucket": {},
    }

    # Per-domain, primary tokenizer only.
    by_domain = defaultdict(list)
    for e in kept:
        by_domain[e.domain].append(e)
    report["by_domain"] = {
        d: summarize(v, args.primary) for d, v in sorted(by_domain.items())
    }

    # Does savings grow with input length? (Research question #5.)
    buckets = defaultdict(list)
    for e in kept:
        n = e.tokens[args.primary].english_tokens
        label = "1-15" if n <= 15 else "16-40" if n <= 40 else "41-80" if n <= 80 else "81+"
        buckets[label].append(e)
    report["by_length_bucket"] = {
        k: summarize(v, args.primary)
        for k, v in sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+")))
    }

    # Evidence that embeddings alone are unsafe: how do invariant violations score?
    viol = [e for e in examples if e.verdict == EquivalenceVerdict.INVARIANT_VIOLATION]
    if viol:
        scored = [e.semantic_score for e in viol if e.semantic_score is not None]
        report["invariant_violations"] = {
            "count": len(viol),
            "mean_embedding_score": round(st.mean(scored), 4) if scored else None,
            "n_scoring_above_threshold": sum(s >= args.threshold for s in scored),
            "by_type": dict(
                sorted(
                    ((k, sum(k in e.violations for e in viol)) for k in
                     {v for e in viol for v in e.violations}),
                    key=lambda kv: -kv[1],
                )
            ),
        }

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_jsonl(examples, outdir / "scored_examples.jsonl")
    _write_examples(kept, args.primary, outdir / "examples.md")
    _print(report, args.primary)
    print(f"\nwrote {outdir}/report.json, scored_examples.jsonl, examples.md")
    return 0


def _write_examples(kept: list[ParallelExample], tid: str, path: Path) -> None:
    """Dump concrete winners on each side - the qualitative deliverable."""
    rows = [(e, e.tokens[tid]) for e in kept if tid in e.tokens]
    zh_best = sorted(rows, key=lambda r: r[1].english_tokens - r[1].mandarin_tokens)[::-1]
    en_best = sorted(rows, key=lambda r: r[1].mandarin_tokens - r[1].english_tokens)[::-1]
    tied = [r for r in rows if abs(r[1].english_tokens - r[1].mandarin_tokens) <= NEGLIGIBLE]

    def fmt(section: str, items) -> str:
        out = [f"## {section}\n"]
        for e, t in items:
            out.append(
                f"- **EN {t.english_tokens} / ZH {t.mandarin_tokens}** "
                f"(sim {e.semantic_score}, {e.domain})\n"
                f"  - `{e.english}`\n  - `{e.mandarin}`"
            )
        return "\n".join(out) + "\n"

    path.write_text(
        f"# Phase 1 qualitative examples ({tid})\n\n"
        + fmt("Mandarin wins by the largest margin", zh_best[:15])
        + "\n"
        + fmt("English wins by the largest margin", en_best[:15])
        + "\n"
        + fmt("Negligible difference", tied[:10]),
        encoding="utf-8",
    )


def _print(r: dict, primary: str) -> None:
    print("\n" + "=" * 78)
    print("PHASE 1 REPORT - is an English/Mandarin hybrid token-cheaper?")
    print("=" * 78)
    print(f"\nExamples: {r['n_examples']:,}   "
          f"semantically valid: {r['n_semantically_valid']:,} "
          f"({pct(r['n_semantically_valid'], r['n_examples']):.1f}%)")
    print("Verdicts: " + "  ".join(f"{k}={v}" for k, v in r["verdicts"].items() if v))

    print(f"\n--- Token economics by tokenizer (n={r['by_tokenizer'][primary]['n']:,}) ---")
    hdr = f"{'tokenizer':<20}{'EN tot':>9}{'ZH tot':>9}{'full-ZH':>9}{'oracle':>9}{'ZH win%':>9}{'EN win%':>9}"
    print(hdr)
    print("-" * len(hdr))
    for t, s in r["by_tokenizer"].items():
        mark = " *" if t == primary else "  "
        print(f"{t + mark:<20}{s['english']['total']:>9,}{s['mandarin']['total']:>9,}"
              f"{s['full_mandarin_reduction_pct']:>8.1f}%{s['oracle_reduction_pct']:>8.1f}%"
              f"{s['mandarin_win_pct']:>8.1f}%{s['english_win_pct']:>8.1f}%")
    print("\n  full-ZH = translate everything to Mandarin (Baseline B)")
    print("  oracle  = per-sentence pick the cheaper side (upper bound for hybrid)")

    print(f"\n--- By domain [{primary}] ---")
    hdr2 = f"{'domain':<16}{'n':>6}{'full-ZH':>10}{'oracle':>10}{'ZH win%':>10}{'EN win%':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for d, s in r["by_domain"].items():
        if s:
            print(f"{d:<16}{s['n']:>6}{s['full_mandarin_reduction_pct']:>9.1f}%"
                  f"{s['oracle_reduction_pct']:>9.1f}%{s['mandarin_win_pct']:>9.1f}%"
                  f"{s['english_win_pct']:>9.1f}%")

    print(f"\n--- By English length [{primary}] ---")
    hdr3 = f"{'tokens':<16}{'n':>6}{'full-ZH':>10}{'oracle':>10}{'ZH win%':>10}"
    print(hdr3)
    print("-" * len(hdr3))
    for b, s in r["by_length_bucket"].items():
        if s:
            print(f"{b:<16}{s['n']:>6}{s['full_mandarin_reduction_pct']:>9.1f}%"
                  f"{s['oracle_reduction_pct']:>9.1f}%{s['mandarin_win_pct']:>9.1f}%")

    if v := r.get("invariant_violations"):
        print("\n--- Why embeddings alone are not enough ---")
        print(f"  {v['count']} pairs failed a hard invariant check.")
        print(f"  Their mean embedding similarity: {v['mean_embedding_score']}")
        print(f"  {v['n_scoring_above_threshold']} of them scored ABOVE the "
              f"{r['semantic_threshold']} similarity threshold,")
        print("  i.e. an embedding-only gate would have accepted them.")
        print(f"  types: {v['by_type']}")


if __name__ == "__main__":
    raise SystemExit(main())
