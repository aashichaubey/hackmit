from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tokenmix.evaluation.storage import read_jsonl, strict_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "deepseek-streamlake.json"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and benchmark a question-conditioned meeting-note pruner"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Fetch pinned public source repositories")
    fetch.add_argument(
        "--directory", type=Path, default=PROJECT_ROOT / "data/raw/meetings"
    )
    imp = sub.add_parser("import-public")
    imp.add_argument(
        "--qmsum", type=Path, default=PROJECT_ROOT / "data/raw/meetings/QMSum"
    )
    imp.add_argument(
        "--explain",
        type=Path,
        default=PROJECT_ROOT / "data/raw/meetings/ExplainMeetSum",
    )
    imp.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/imported")
    ann = sub.add_parser(
        "annotate",
        help="Prepare source-checked questions for question-independent notes",
    )
    ann.add_argument(
        "--sources",
        type=Path,
        default=PROJECT_ROOT / "data/imported/notes_sources.jsonl",
    )
    ann.add_argument("--output", type=Path, required=True)
    ann.add_argument("--train-meetings", type=int, default=12)
    ann.add_argument("--val-meetings", type=int, default=3)
    ann.add_argument("--test-meetings", type=int, default=3)
    ann.add_argument("--seed", type=int, default=17)
    review = sub.add_parser("apply-review")
    review.add_argument("--input", type=Path, required=True)
    review.add_argument("--review", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    curate = sub.add_parser(
        "curate",
        help="Combine training data with source-versioned authored evaluation questions",
    )
    curate.add_argument("--sources", type=Path, required=True)
    curate.add_argument("--questions", type=Path, required=True)
    curate.add_argument("--train", type=Path, required=True)
    curate.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--train", type=Path, action="append", required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model", default="answerdotai/ModernBERT-base")
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--max-steps", type=int, default=0)
    train.add_argument("--max-examples", type=int, default=0)
    train.add_argument("--accumulation", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--keep-weight", type=float, default=3)
    train.add_argument("--seed", type=int, default=17)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--checkpoint", type=Path, required=True)
    cal.add_argument("--validation", type=Path, required=True)
    cal.add_argument(
        "--recall-targets",
        type=float,
        nargs=3,
        default=(0.99, 0.97, 0.95),
        metavar=("CONSERVATIVE", "BALANCED", "AGGRESSIVE"),
    )
    prune = sub.add_parser("prune")
    prune.add_argument("--checkpoint", type=Path, required=True)
    prune.add_argument("--question", required=True)
    prune.add_argument("--notes", type=Path, required=True)
    prune.add_argument("--threshold", type=float)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--checkpoint", type=Path, required=True)
    bench.add_argument("--dataset", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--max-cases", type=int, default=0)
    bench.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=2)
    bench.add_argument(
        "--no-lingua",
        action="store_true",
        help="Explicitly omit the LLMLingua comparator",
    )
    report = sub.add_parser("report")
    report.add_argument("directory", type=Path)
    audit = sub.add_parser(
        "audit-grades", help="Apply a versioned scoring review into a separate report"
    )
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--review", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    for command in (ann, bench):
        command.add_argument("--target-config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--journal", type=Path)
        command.add_argument(
            "--max-calls", type=int, default=400 if command is bench else 72
        )
        command.add_argument(
            "--max-attempts",
            type=int,
            default=3,
            help="Lifetime attempt cap per request; raising it explicitly retries exhausted transient failures",
        )
    for command in (train, cal, prune, bench):
        command.add_argument(
            "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            args.directory.mkdir(parents=True, exist_ok=True)
            for name, repo, revision in [
                (
                    "QMSum",
                    "Yale-LILY/QMSum",
                    "83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e",
                ),
                (
                    "ExplainMeetSum",
                    "hkim-etri/ExplainMeetSum",
                    "d8459311a6578cd438a785bf19dd2fa757ade1c5",
                ),
            ]:
                dest = args.directory / name
                if not dest.exists():
                    subprocess.run(
                        [
                            "git",
                            "clone",
                            "--no-checkout",
                            f"https://github.com/{repo}.git",
                            str(dest),
                        ],
                        check=True,
                    )
                    subprocess.run(
                        ["git", "-C", str(dest), "checkout", "--detach", revision],
                        check=True,
                    )
                actual = subprocess.check_output(
                    ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
                ).strip()
                if actual != revision:
                    raise ValueError(
                        f"{dest} has a different revision; use a fresh destination"
                    )
            result = {"sources": str(args.directory)}
        elif args.command == "import-public":
            from .data import import_public

            result = import_public(args.qmsum, args.explain, args.output)
        elif args.command == "annotate":
            from .api import JournaledAPI
            from .data import annotate_notes

            limits = {
                "train": args.train_meetings,
                "val": args.val_meetings,
                "test": args.test_meetings,
            }
            if any(v < 0 for v in limits.values()) or not any(limits.values()):
                raise ValueError(
                    "meeting limits must be nonnegative and at least one must be positive"
                )
            api = JournaledAPI(
                args.target_config,
                args.journal or args.output / "api.jsonl",
                max_calls=args.max_calls,
                max_attempts=args.max_attempts,
            )
            result = annotate_notes(
                args.sources, args.output, api, per_split=limits, seed=args.seed
            )
        elif args.command == "apply-review":
            from .data import apply_review

            result = apply_review(args.input, args.review, args.output)
        elif args.command == "curate":
            from .data import curate

            result = curate(args.sources, args.questions, args.train, args.output)
        elif args.command == "train":
            from .training import train as train_model

            result = train_model(
                args.train,
                args.validation,
                args.output,
                model_name=args.model,
                device=args.device,
                max_length=args.max_length,
                epochs=args.epochs,
                max_steps=args.max_steps,
                accumulation=args.accumulation,
                lr=args.learning_rate,
                keep_weight=args.keep_weight,
                seed=args.seed,
                max_examples=args.max_examples,
            )
        elif args.command == "calibrate":
            from .training import calibrate

            result = calibrate(
                args.checkpoint,
                args.validation,
                device=args.device,
                recall_targets=args.recall_targets,
            )
        elif args.command == "prune":
            from .model import ModernBertPruner

            result = (
                ModernBertPruner(args.checkpoint, device=args.device)
                .prune(args.question, args.notes.read_text(), threshold=args.threshold)
                .to_dict()
            )
        elif args.command == "benchmark":
            from .api import JournaledAPI
            from .benchmark import benchmark

            if args.max_cases < 0:
                raise ValueError("max-cases must be nonnegative")
            api = JournaledAPI(
                args.target_config,
                args.journal or args.output / "api.jsonl",
                max_calls=args.max_calls,
                max_attempts=args.max_attempts,
            )
            result = {
                "artifacts": str(
                    benchmark(
                        args.dataset,
                        args.checkpoint,
                        args.output,
                        api,
                        device=args.device,
                        max_cases=args.max_cases,
                        include_lingua=not args.no_lingua,
                        workers=args.workers,
                    )
                )
            }
        elif args.command == "audit-grades":
            from .audit import audit_grades

            result = audit_grades(args.source, args.dataset, args.review, args.output)
        else:
            from .benchmark import write_reports

            manifest = strict_json((args.directory / "manifest.json").read_text())
            rows = read_jsonl(args.directory / "per_call.jsonl")
            result = write_reports(
                args.directory,
                rows,
                manifest,
                complete=len(rows) == manifest["expected_pairs"],
            )
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f"pruning-model: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
