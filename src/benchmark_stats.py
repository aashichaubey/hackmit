"""Statistical summaries, paired analyses, and plots for benchmark_v1."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

BOOTSTRAP_SEED = 1729
BOOTSTRAP_DRAWS = 10_000


def as_bool(value: object) -> bool:
    """Accept in-memory booleans and their CSV representation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise ValueError(f"expected boolean value, got {value!r}")


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index); upper = min(lower + 1, len(ordered)); fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(values: list[float], seed_offset: int = 0) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws = [mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_DRAWS)]
    return percentile(draws, .025), percentile(draws, .975)


def summarize(rep_rows: list[dict], qa_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    strategies = ("ORIGINAL", "COMPACT_ENGLISH", "TOKEN_OPTIMIZED")
    summary, paired = [], []
    original_qa = defaultdict(dict)
    for row in qa_rows:
        if row["strategy"] == "ORIGINAL":
            original_qa[row["context_id"]][int(row["question_index"])] = as_bool(row["judge_correct"])
    for strategy_index, strategy in enumerate(strategies):
        reps = [row for row in rep_rows if row["strategy"] == strategy]
        qas = [row for row in qa_rows if row["strategy"] == strategy]
        reductions = [float(row["reduction_percent"]) for row in reps]
        token_counts = [int(row["compressed_tokens"]) for row in reps]
        judge_values = [as_bool(row["judge_correct"]) for row in qas]
        deterministic_values = [as_bool(row["deterministic_correct"]) for row in qas]
        per_context_diff = []
        transitions = {}
        for context_id in sorted({row["context_id"] for row in qas}):
            current = {
                int(row["question_index"]): as_bool(row["judge_correct"])
                for row in qas if row["context_id"] == context_id
            }
            original = original_qa[context_id]
            if current.keys() != original.keys():
                raise ValueError(f"question mismatch for {context_id} / {strategy}")
            additional_failures = sum(original[index] and not current[index] for index in original)
            gains = sum(not original[index] and current[index] for index in original)
            current_correct = sum(current.values())
            original_correct = sum(original.values())
            transitions[context_id] = {
                "current_correct": current_correct,
                "original_correct": original_correct,
                "additional_failures": additional_failures,
                "gains": gains,
            }
            per_context_diff.append(current_correct / len(current) - original_correct / len(original))
        reduction_ci = bootstrap_ci(reductions, strategy_index)
        accuracy_diff_ci = bootstrap_ci(per_context_diff, 100 + strategy_index)
        summary.append({
            "strategy": strategy, "contexts": len(reps), "qa_pairs": len(qas),
            "mean_tokens": mean(token_counts), "median_tokens": median(token_counts), "token_stddev": pstdev(token_counts),
            "mean_reduction_percent": mean(reductions), "median_reduction_percent": median(reductions),
            "reduction_stddev": pstdev(reductions),
            "reduction_ci_low": reduction_ci[0], "reduction_ci_high": reduction_ci[1],
            "total_tokens_saved": sum(int(row["tokens_saved"]) for row in reps),
            "deterministic_accuracy": mean(deterministic_values), "judge_accuracy": mean(judge_values),
            "qa_diff_vs_original": mean(per_context_diff), "qa_diff_ci_low": accuracy_diff_ci[0], "qa_diff_ci_high": accuracy_diff_ci[1],
            "contexts_smaller": sum(int(row["compressed_tokens"]) < int(row["original_tokens"]) for row in reps),
            "contexts_unchanged": sum(int(row["compressed_tokens"]) == int(row["original_tokens"]) for row in reps),
            "contexts_larger": sum(int(row["compressed_tokens"]) > int(row["original_tokens"]) for row in reps),
            "candidate_zero_selected": sum(as_bool(row.get("candidate_zero_selected", False)) for row in reps),
            "additional_question_failures": sum(value["additional_failures"] for value in transitions.values()),
            "question_gains": sum(value["gains"] for value in transitions.values()),
            "contexts_with_additional_failure": sum(value["additional_failures"] > 0 for value in transitions.values()),
            "contexts_saved_without_additional_qa_loss": sum(
                int(rep["tokens_saved"]) > 0 and transitions[rep["context_id"]]["additional_failures"] == 0
                for rep in reps
            ),
        })
        if strategy != "ORIGINAL":
            for rep in reps:
                cid = rep["context_id"]
                transition = transitions[cid]
                paired.append({
                    "context_id": cid, "category": rep["category"], "strategy": strategy,
                    "reduction_percent": rep["reduction_percent"],
                    "qa_correct": transition["current_correct"], "original_qa_correct": transition["original_correct"],
                    "qa_difference": transition["current_correct"] - transition["original_correct"],
                    "additional_question_failures": transition["additional_failures"],
                    "question_gains": transition["gains"],
                    "saved_without_qa_loss": int(rep["tokens_saved"]) > 0 and transition["additional_failures"] == 0,
                    "caused_failure": transition["additional_failures"] > 0,
                    "candidate_zero_selected": as_bool(rep.get("candidate_zero_selected", False)),
                })
    return summary, paired


def write_outputs(output: Path, rep_rows: list[dict], qa_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summary, paired = summarize(rep_rows, qa_rows)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary.csv", summary), ("paired_analysis.csv", paired)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "statistics.json").write_text(json.dumps({"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "summary": summary, "paired": paired}, indent=2) + "\n")
    make_plots(output, rep_rows, qa_rows, summary)
    return summary, paired


def make_plots(output: Path, rep_rows: list[dict], qa_rows: list[dict], summary: list[dict]) -> None:
    import matplotlib.pyplot as plt
    strategies = [row["strategy"] for row in summary]
    colors = {"ORIGINAL": "#4C78A8", "COMPACT_ENGLISH": "#F58518", "TOKEN_OPTIMIZED": "#54A24B"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary:
        ax.scatter(row["mean_tokens"], row["judge_accuracy"], s=90, color=colors[row["strategy"]])
        ax.annotate(row["strategy"], (row["mean_tokens"], row["judge_accuracy"]), xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Mean representation tokens", ylabel="Blinded-judge QA accuracy", title="Token count vs QA accuracy", ylim=(0, 1.02)); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "token_vs_accuracy.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, strategy in zip(axes, ("COMPACT_ENGLISH", "TOKEN_OPTIMIZED")):
        values = [float(row["reduction_percent"]) * 100 for row in rep_rows if row["strategy"] == strategy]
        ax.hist(values, bins=min(10, max(3, len(values))), color=colors[strategy], edgecolor="white")
        ax.axvline(0, color="black", linewidth=1); ax.set(title=strategy, xlabel="Token reduction (%)", ylabel="Contexts")
    fig.suptitle("Distribution of per-context token reductions"); fig.tight_layout(); fig.savefig(output / "reduction_histograms.png", dpi=160); plt.close(fig)

    category_data = defaultdict(lambda: defaultdict(list))
    for row in rep_rows: category_data[row["category"]][row["strategy"]].append(float(row["reduction_percent"]) * 100)
    categories = sorted(category_data)
    fig, ax = plt.subplots(figsize=(11, 6))
    width = .35; positions = list(range(len(categories)))
    for offset, strategy in ((-.175, "COMPACT_ENGLISH"), (.175, "TOKEN_OPTIMIZED")):
        ax.bar([p + offset for p in positions], [mean(category_data[c][strategy]) for c in categories], width, label=strategy, color=colors[strategy])
    ax.axhline(0, color="black", linewidth=1); ax.set_xticks(positions, categories, rotation=35, ha="right"); ax.set(ylabel="Mean token reduction (%)", title="Compression by category"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "category_compression.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in strategies:
        reps = [row for row in rep_rows if row["strategy"] == strategy]
        for rep in reps:
            qas = [row for row in qa_rows if row["strategy"] == strategy and row["context_id"] == rep["context_id"]]
            ax.scatter(float(rep["reduction_percent"]) * 100, mean(as_bool(q["judge_correct"]) for q in qas), alpha=.7, color=colors[strategy], label=strategy)
    handles, labels = ax.get_legend_handles_labels(); unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys()); ax.set(xlabel="Per-context token reduction (%)", ylabel="Per-context QA accuracy", title="Accuracy–compression tradeoff", ylim=(-.02, 1.02)); ax.axvline(0, color="black", linewidth=1); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / "accuracy_compression_tradeoff.png", dpi=160); plt.close(fig)
