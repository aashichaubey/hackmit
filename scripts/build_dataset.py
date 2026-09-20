"""Build the aligned, normalized, token-measured parallel dataset.

    python -m scripts.build_dataset --per-corpus 400
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_language.data.datasets import (  # noqa: E402
    build_examples,
    split_by_source_document,
    write_jsonl,
)
from token_language.data.loaders import REGISTRY  # noqa: E402
from token_language.tokenizer import HFTokenizer, TiktokenTokenizer  # noqa: E402

log = logging.getLogger("build_dataset")

DEFAULT_CORPORA = ["wmt_news", "opus100", "tatoeba", "kde4", "bible", "tldr"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpora", nargs="*", default=DEFAULT_CORPORA)
    p.add_argument("--per-corpus", type=int, default=400, help="target kept examples")
    p.add_argument("--out", default="data/processed/examples.jsonl")
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument(
        "--tokenizers",
        nargs="*",
        default=["deepseek-v4-flash", "o200k_base", "cl100k_base"],
        help="primary target first",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tokenizers = {}
    for tid in args.tokenizers:
        tokenizers[tid] = (
            TiktokenTokenizer(tid)
            if tid.endswith("_base")
            else HFTokenizer.from_pretrained(tid)
        )
    log.info("tokenizers: %s", list(tokenizers))

    all_examples = []
    stats_rows = []
    for corpus in args.corpora:
        if corpus not in REGISTRY:
            log.warning("skipping unknown corpus %r", corpus)
            continue
        try:
            examples, stats = build_examples(corpus, tokenizers, limit=args.per_corpus)
        except Exception as exc:  # network/corpus failures must not kill the run
            log.error("corpus %s failed: %s: %s", corpus, type(exc).__name__, exc)
            continue
        all_examples.extend(examples)
        stats_rows.append(stats)
        print("  " + stats.summary())

    if not all_examples:
        log.error("no examples built")
        return 1

    train, test = split_by_source_document(all_examples, args.test_fraction)
    out = Path(args.out)
    write_jsonl(all_examples, out)
    write_jsonl(train, out.with_name("train.jsonl"))
    write_jsonl(test, out.with_name("test.jsonl"))

    meta = {
        "total": len(all_examples),
        "train": len(train),
        "test": len(test),
        "tokenizers": list(tokenizers),
        "corpora": {
            s.corpus: {
                "seen": s.seen,
                "kept": s.kept,
                "yield_pct": round(s.yield_pct, 2),
                "domain": REGISTRY[s.corpus].domain,
                "license": REGISTRY[s.corpus].license,
                "rejected": dict(s.rejected),
            }
            for s in stats_rows
        },
    }
    out.with_name("dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nwrote {len(all_examples):,} examples -> {out}")
    print(f"  train={len(train):,}  test={len(test):,}  (split is document-grouped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
