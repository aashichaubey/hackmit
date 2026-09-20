"""Run the question-independent multi-question compression experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from .evaluate import evaluate_question
from .generate_dataset import DEFAULT_OUTPUT, EXAMPLES, validate_examples
from .representations import (
    LLMProvider, SEARCH_COMPARISON_STRATEGIES as STRATEGIES, encode_facts,
    extract_facts, search_token_optimized, validate_semantics,
)
from .tokenizer import TargetTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_examples(path: Path) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(EXAMPLES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    examples = json.loads(path.read_text(encoding="utf-8"))
    validate_examples(examples)
    return examples


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    summary = []
    for strategy in STRATEGIES:
        group = grouped[strategy]
        # Token count repeats for five question rows, so average across unique contexts.
        per_context = {}
        for row in group:
            per_context[row["example_id"]] = (row["token_count"], row["compression_ratio"])
        avg_tokens = sum(v[0] for v in per_context.values()) / len(per_context)
        avg_reduction = sum(v[1] for v in per_context.values()) / len(per_context)
        summary.append({
            "strategy": strategy,
            "average_tokens": round(avg_tokens, 3),
            "reduction": round(avg_reduction, 6),
            "deterministic_correct": sum(bool(r["deterministic_correct"]) for r in group),
            "deterministic_accuracy": round(sum(bool(r["deterministic_correct"]) for r in group) / len(group), 6),
            "judge_correct": sum(bool(r["judge_correct"]) for r in group),
            "judge_accuracy": round(sum(bool(r["judge_correct"]) for r in group) / len(group), 6),
            "semantic_valid_contexts": sum(bool(v) for v in {r["example_id"]: r["semantic_valid"] for r in group}.values()),
            "semantic_validation_rate": round(sum(bool(v) for v in {r["example_id"]: r["semantic_valid"] for r in group}.values()) / len(per_context), 6),
            "questions_total": len(group),
        })
    return summary


def print_summary(summary: list[dict]) -> None:
    print("\nStrategy          Avg tokens   Reduction   Semantic   Deterministic   Judge QA")
    for row in summary:
        total = row["questions_total"]
        print(f"{row['strategy']:<17} {row['average_tokens']:>10.1f}   {row['reduction']:>8.1%}   {row['semantic_validation_rate']:>7.1%}   {row['deterministic_correct']:>2}/{total:<2} ({row['deterministic_accuracy']:>5.1%})   {row['judge_correct']:>2}/{total:<2} ({row['judge_accuracy']:>5.1%})")


def print_context_results(examples: list[dict], rows: list[dict]) -> None:
    print("\nPER-CONTEXT JUDGE RESULTS")
    for example in examples:
        print(f"Context {example['id']}:")
        for strategy in STRATEGIES:
            group = [r for r in rows if r["example_id"] == example["id"] and r["strategy"] == strategy]
            correct = sum(bool(r["judge_correct"]) for r in group)
            print(f"  {strategy:<16} tokens={group[0]['token_count']:<4} reduction={group[0]['compression_ratio']:.1%} judge={correct}/5")


def print_search_analysis(records: list[dict]) -> None:
    print("\nTOKEN_OPTIMIZED SEARCH")
    for record in records:
        print(f"Context {record['example_id']}:")
        for candidate in record["candidates"]:
            status = "valid" if candidate["valid"] else "invalid"
            marker = " <- winner" if candidate["candidate"] == record["winner_candidate"] else ""
            print(f"  Candidate {candidate['candidate']}: {candidate['token_count']} tokens, {status}{marker}")
        print(f"  WINNING REPRESENTATION:\n{record['winner_text']}\n")


def plot_tradeoff(summary: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary:
        ax.scatter(row["average_tokens"], row["judge_accuracy"], s=75)
        ax.annotate(row["strategy"], (row["average_tokens"], row["judge_accuracy"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Average representation tokens", ylabel="LLM-judge QA accuracy", title="Reusable representation: compression / QA tradeoff", ylim=(-0.03, 1.05))
    ax.grid(alpha=.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    examples = load_examples(args.dataset)
    if args.limit is not None:
        if not 1 <= args.limit <= 3:
            parser.error("--limit must be between 1 and 3")
        examples = examples[:args.limit]

    provider = LLMProvider()
    target_model = os.getenv("TARGET_MODEL", provider.answer_model)
    tokenizer = TargetTokenizer(target_model)
    print(f"Provider={provider.provider}; generation={provider.generation_model}; answer={provider.answer_model}; judge={provider.judge_model}; target={target_model}; tokenizer={tokenizer.name}")
    rows, saved_representations, optimization_records, usage_events = [], [], [], []
    for position, example in enumerate(examples, 1):
        print(f"[{position}/{len(examples)}] compiling context {example['id']} (questions withheld)")
        facts_result = extract_facts(example["context"], provider)
        representations = {"ORIGINAL": {"text": example["context"], "usage": {}}, "FACTS": facts_result}
        usage_events.append(("generation", facts_result["usage"]))
        for strategy in ("COMPACT_ENGLISH", "SYMBOLIC", "MANDARIN", "HYBRID"):
            representations[strategy] = encode_facts(facts_result["text"], strategy, provider)
            usage_events.append(("generation", representations[strategy]["usage"]))
        optimization = search_token_optimized(example["context"], facts_result["text"], tokenizer, provider)
        winner = optimization["winner"]
        representations["TOKEN_OPTIMIZED"] = {"text": winner["text"], "usage": winner["generation_usage"]}
        for candidate in optimization["candidates"]:
            usage_events.append(("search_generation", candidate["generation_usage"]))
            usage_events.append(("semantic_validation", candidate["validation_usage"]))
        optimization_records.append({"example_id": example["id"], "winner_candidate": winner["candidate"], "winner_token_count": winner["token_count"], "winner_text": winner["text"], "candidates": optimization["candidates"]})
        semantic_results = {}
        for strategy in STRATEGIES:
            if strategy == "TOKEN_OPTIMIZED":
                semantic_results[strategy] = {"valid": True, "reason": winner["validation_reason"]}
            else:
                semantic_results[strategy] = validate_semantics(facts_result["text"], representations[strategy]["text"], provider)
                usage_events.append(("semantic_validation", semantic_results[strategy]["usage"]))
        original_tokens = tokenizer.count(representations["ORIGINAL"]["text"])
        saved_representations.append({"example_id": example["id"], "representations": {k: v["text"] for k, v in representations.items()}})
        for strategy in STRATEGIES:
            text = representations[strategy]["text"]
            token_count = tokenizer.count(text)
            for question_index, item in enumerate(example["questions"], 1):
                evaluation = evaluate_question(text, item, provider)
                rows.append({
                    "example_id": example["id"], "category": example["category"], "strategy": strategy,
                    "question_index": question_index, "question": item["question"], "expected_answer": item["expected_answer"],
                    "token_count": token_count, "tokens_saved": original_tokens - token_count,
                    "compression_ratio": 1 - token_count / original_tokens, "model_answer": evaluation["answer"],
                    "deterministic_correct": evaluation["deterministic_correct"], "deterministic_method": evaluation["deterministic_method"],
                    "judge_correct": evaluation["judge_correct"], "judge_reason": evaluation["judge_reason"],
                    "semantic_valid": semantic_results[strategy]["valid"], "semantic_validation_reason": semantic_results[strategy]["reason"],
                    "generation_input_tokens": representations[strategy].get("usage", {}).get("input_tokens", 0),
                    "generation_output_tokens": representations[strategy].get("usage", {}).get("output_tokens", 0),
                    "answer_input_tokens": evaluation["answer_usage"].get("input_tokens", 0),
                    "answer_output_tokens": evaluation["answer_usage"].get("output_tokens", 0),
                    "judge_input_tokens": evaluation["judge_usage"].get("input_tokens", 0),
                    "judge_output_tokens": evaluation["judge_usage"].get("output_tokens", 0),
                })
                usage_events.append(("answer", evaluation["answer_usage"]))
                usage_events.append(("qa_judge", evaluation["judge_usage"]))

    fields = ["example_id", "category", "strategy", "question_index", "question", "expected_answer", "token_count", "tokens_saved", "compression_ratio", "semantic_valid", "semantic_validation_reason", "model_answer", "deterministic_correct", "deterministic_method", "judge_correct", "judge_reason", "generation_input_tokens", "generation_output_tokens", "answer_input_tokens", "answer_output_tokens", "judge_input_tokens", "judge_output_tokens"]
    write_csv(ROOT / "results" / "qa_results.csv", rows, fields)
    summary = build_summary(rows)
    write_csv(ROOT / "results" / "summary.csv", summary, ["strategy", "average_tokens", "reduction", "semantic_valid_contexts", "semantic_validation_rate", "deterministic_correct", "deterministic_accuracy", "judge_correct", "judge_accuracy", "questions_total"])
    (ROOT / "results" / "representations.json").write_text(json.dumps(saved_representations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "results" / "optimization_candidates.json").write_text(json.dumps(optimization_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_tradeoff(summary, ROOT / "results" / "tradeoff.png")
    print_summary(summary)
    print_context_results(examples, rows)
    print_search_analysis(optimization_records)
    print("\nAPI USAGE")
    for scope in ("generation", "search_generation", "semantic_validation", "answer", "qa_judge"):
        events = [usage for event_scope, usage in usage_events if event_scope == scope]
        print(f"{scope}={sum(int(u.get('input_tokens', 0)) for u in events)} in/{sum(int(u.get('output_tokens', 0)) for u in events)} out")


if __name__ == "__main__":
    main()
