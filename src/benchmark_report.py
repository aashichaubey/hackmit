"""Post-run reporting only; never changes official benchmark scores."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

from .benchmark_stats import write_outputs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "benchmark_v1" / "full_50"


def read_csv(name: str) -> list[dict]:
    return list(csv.DictReader((OUTPUT / name).open(encoding="utf-8")))


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    reps, qas = read_csv("representations.csv"), read_csv("qa_results.csv")
    # Rebuild derived statistics and plots from the immutable representation
    # and QA rows. This never changes model answers or judge verdicts.
    write_outputs(OUTPUT, reps, qas)
    strategies = ("ORIGINAL", "COMPACT_ENGLISH", "TOKEN_OPTIMIZED")
    categories = sorted({row["category"] for row in reps})
    category_rows = []
    for category in categories:
        for strategy in strategies:
            sr = [row for row in reps if row["category"] == category and row["strategy"] == strategy]
            sq = [row for row in qas if row["category"] == category and row["strategy"] == strategy]
            reductions = [float(row["reduction_percent"]) for row in sr]
            category_rows.append({
                "category": category, "strategy": strategy, "contexts": len(sr), "qa_pairs": len(sq),
                "mean_tokens": mean(int(row["compressed_tokens"]) for row in sr),
                "mean_reduction_percent": mean(reductions), "median_reduction_percent": median(reductions),
                "reduction_stddev": pstdev(reductions),
                "deterministic_accuracy": mean(row["deterministic_correct"] == "True" for row in sq),
                "judge_accuracy": mean(row["judge_correct"] == "True" for row in sq),
            })
    write_csv("category_summary.csv", category_rows)

    failures = []
    for row in qas:
        if row["judge_correct"] == "False":
            possible_error = row["deterministic_correct"] == "True"
            failures.append({
                "context_id": row["context_id"], "category": row["category"], "strategy": row["strategy"],
                "question_index": row["question_index"], "question": row["question"],
                "expected_answer": row["expected_answer"], "model_answer": row["model_answer"],
                "judge_reason": row["judge_reason"], "deterministic_correct": row["deterministic_correct"],
                "possible_evaluator_error": possible_error,
                "audit_note": "Automatically flagged because deterministic normalized matching passed while the blinded judge failed; official result unchanged." if possible_error else "Official failure; requires qualitative inspection before attributing cause.",
            })
    write_csv("judge_failure_audit.csv", failures)

    by_key = {(row["context_id"], row["strategy"]): row for row in reps}
    qa_counts = defaultdict(int)
    qa_verdicts = {}
    for row in qas:
        qa_counts[(row["context_id"], row["strategy"])] += row["judge_correct"] == "True"
        qa_verdicts[(row["context_id"], row["strategy"], int(row["question_index"]))] = row["judge_correct"] == "True"
    context_ids = sorted({row["context_id"] for row in reps})
    direct = {
        "compact_wins_compression": 0, "token_optimized_wins_compression": 0, "compression_ties": 0,
        "compact_wins_qa": 0, "token_optimized_wins_qa": 0, "qa_ties": 0,
        "per_context": [],
    }
    important = []
    for cid in context_ids:
        compact_tokens = int(by_key[(cid, "COMPACT_ENGLISH")]["compressed_tokens"])
        optimized_tokens = int(by_key[(cid, "TOKEN_OPTIMIZED")]["compressed_tokens"])
        if compact_tokens < optimized_tokens: direct["compact_wins_compression"] += 1
        elif optimized_tokens < compact_tokens: direct["token_optimized_wins_compression"] += 1
        else: direct["compression_ties"] += 1
        compact_qa = qa_counts[(cid, "COMPACT_ENGLISH")]; optimized_qa = qa_counts[(cid, "TOKEN_OPTIMIZED")]
        if compact_qa > optimized_qa: direct["compact_wins_qa"] += 1
        elif optimized_qa > compact_qa: direct["token_optimized_wins_qa"] += 1
        else: direct["qa_ties"] += 1
        original_qa = qa_counts[(cid, "ORIGINAL")]
        compact_failures = sum(qa_verdicts[(cid, "ORIGINAL", i)] and not qa_verdicts[(cid, "COMPACT_ENGLISH", i)] for i in range(1, 6))
        compact_gains = sum(not qa_verdicts[(cid, "ORIGINAL", i)] and qa_verdicts[(cid, "COMPACT_ENGLISH", i)] for i in range(1, 6))
        optimized_failures = sum(qa_verdicts[(cid, "ORIGINAL", i)] and not qa_verdicts[(cid, "TOKEN_OPTIMIZED", i)] for i in range(1, 6))
        optimized_gains = sum(not qa_verdicts[(cid, "ORIGINAL", i)] and qa_verdicts[(cid, "TOKEN_OPTIMIZED", i)] for i in range(1, 6))
        row = {
            "context_id": cid, "category": by_key[(cid, "ORIGINAL")]["category"],
            "compact_tokens": compact_tokens, "token_optimized_tokens": optimized_tokens,
            "compact_qa": compact_qa, "token_optimized_qa": optimized_qa, "original_qa": original_qa,
            "compact_additional_failures": compact_failures, "compact_gains": compact_gains,
            "token_optimized_additional_failures": optimized_failures, "token_optimized_gains": optimized_gains,
        }
        direct["per_context"].append(row)
        if compact_failures or optimized_failures:
            important.append(row)
    direct["mean_token_difference_token_optimized_minus_compact"] = mean(row["token_optimized_tokens"] - row["compact_tokens"] for row in direct["per_context"])
    direct["mean_qa_difference_token_optimized_minus_compact"] = mean(row["token_optimized_qa"] - row["compact_qa"] for row in direct["per_context"]) / 5
    direct["compact_additional_question_failures"] = sum(row["compact_additional_failures"] for row in direct["per_context"])
    direct["compact_question_gains"] = sum(row["compact_gains"] for row in direct["per_context"])
    direct["compact_contexts_with_additional_failure"] = sum(row["compact_additional_failures"] > 0 for row in direct["per_context"])
    direct["token_optimized_additional_question_failures"] = sum(row["token_optimized_additional_failures"] for row in direct["per_context"])
    direct["token_optimized_question_gains"] = sum(row["token_optimized_gains"] for row in direct["per_context"])
    direct["token_optimized_contexts_with_additional_failure"] = sum(row["token_optimized_additional_failures"] > 0 for row in direct["per_context"])
    direct["important_failure_contexts"] = important
    (OUTPUT / "direct_comparison.json").write_text(json.dumps(direct, indent=2) + "\n")
    print(json.dumps({k: v for k, v in direct.items() if k != "per_context"}, indent=2))
    print(f"official_failures={len(failures)} possible_evaluator_errors={sum(row['possible_evaluator_error'] for row in failures)}")


if __name__ == "__main__":
    main()
