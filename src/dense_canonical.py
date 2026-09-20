"""DENSE_CANONICAL stability and downstream-QA experiment."""

from __future__ import annotations

import csv
import json
import math
import os
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

from dotenv import load_dotenv

from .evaluate import evaluate_question
from .generate_dataset import DEFAULT_OUTPUT, validate_examples
from .representations import LLMProvider
from .tokenizer import TargetTokenizer

ROOT = Path(__file__).resolve().parents[1]

DENSE_SYSTEM = (
    "Deterministically compile supplied facts into DENSE_CANONICAL v1 for reuse by a capable language model. "
    "Preserve every proposition. Use concise English vocabulary; omit unnecessary function words only when meaning "
    "remains clear; use common abbreviations, numeric forms, 24-hour times, arrows for changes/implications, comparison "
    "operators, compact lists, and semicolon-delimited facts where appropriate. Use stable entity and relation names. "
    "Do not use JSON, XML, Markdown, verbose field names, a decoding dictionary, multilingual variation, commentary, or "
    "stylistic alternatives. Emit exactly one plain-text representation with facts in source order."
)


def generate_dense_canonical(facts: str, provider: LLMProvider) -> dict:
    """Question-independent deterministic compiler; structurally accepts only facts."""
    assert isinstance(facts, str) and facts.strip()
    return provider.complete(DENSE_SYSTEM, f"SOURCE FACTS:\n{facts}", purpose="generation", temperature=0)


def lcp_length(left: list[int], right: list[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def stability_metrics(texts: list[str], tokenizer: TargetTokenizer) -> dict:
    assert len(texts) == 10
    sequences = [tokenizer.encode(text) for text in texts]
    pairs = list(combinations(range(len(texts)), 2))
    text_equal = [texts[i] == texts[j] for i, j in pairs]
    token_equal = [sequences[i] == sequences[j] for i, j in pairs]
    prefix_pct = [lcp_length(sequences[i], sequences[j]) / max(1, min(len(sequences[i]), len(sequences[j]))) for i, j in pairs]
    similarities = [SequenceMatcher(None, sequences[i], sequences[j], autojunk=False).ratio() for i, j in pairs]
    global_lcp = min((lcp_length(sequences[0], sequence) for sequence in sequences[1:]), default=len(sequences[0]))
    counts = [len(sequence) for sequence in sequences]
    return {
        "all_text_identical": all(text == texts[0] for text in texts),
        "pairwise_exact_text_rate": mean(text_equal),
        "all_token_sequences_identical": all(sequence == sequences[0] for sequence in sequences),
        "pairwise_exact_token_rate": mean(token_equal),
        "global_longest_common_token_prefix": global_lcp,
        "global_lcp_percentage": global_lcp / max(1, min(counts)),
        "average_pairwise_lcp_percentage": mean(prefix_pct),
        "average_pairwise_token_similarity": mean(similarities),
        "token_count_mean": mean(counts),
        "token_count_stddev": pstdev(counts),
        "token_counts": counts,
    }


def load_frozen_inputs() -> tuple[list[dict], dict[int, dict]]:
    examples = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    validate_examples(examples)
    saved = json.loads((ROOT / "results" / "representations.json").read_text(encoding="utf-8"))
    by_id = {int(item["example_id"]): item["representations"] for item in saved}
    assert len(examples) == 3 and set(by_id) == {1, 2, 3}
    for example in examples:
        assert by_id[example["id"]]["ORIGINAL"] == example["context"], "saved representation does not match frozen context"
        assert by_id[example["id"]]["FACTS"].strip()
    return examples, by_id


def prior_comparison() -> dict[str, dict]:
    rows = list(csv.DictReader((ROOT / "results" / "qa_results.csv").open(encoding="utf-8")))
    result = {}
    for strategy in ("ORIGINAL", "COMPACT_ENGLISH", "TOKEN_OPTIMIZED"):
        group = [row for row in rows if row["strategy"] == strategy]
        assert len(group) == 15, f"missing frozen prior results for {strategy}"
        contexts = {row["example_id"]: row for row in group}
        result[strategy] = {
            "average_tokens": mean(int(row["token_count"]) for row in contexts.values()),
            "reduction": mean(float(row["compression_ratio"]) for row in contexts.values()),
            "judge_correct": sum(row["judge_correct"] == "True" for row in group),
            "judge_accuracy": sum(row["judge_correct"] == "True" for row in group) / len(group),
            "exact_stability_rate": 1.0 if strategy == "ORIGINAL" else None,
            "average_lcp_percentage": 1.0 if strategy == "ORIGINAL" else None,
        }
    return result


def main() -> None:
    load_dotenv(ROOT / ".env")
    examples, frozen = load_frozen_inputs()
    provider = LLMProvider()
    target_model = os.getenv("TARGET_MODEL", provider.answer_model)
    tokenizer = TargetTokenizer(target_model)
    print(f"Provider={provider.provider}; generation={provider.generation_model}; answer={provider.answer_model}; judge={provider.judge_model}; target={target_model}; tokenizer={tokenizer.name}")
    records, qa_rows, usage_events = [], [], []
    original_counts = {example["id"]: tokenizer.count(example["context"]) for example in examples}

    for example in examples:
        facts = frozen[example["id"]]["FACTS"]
        print(f"Context {example['id']}: generating DENSE_CANONICAL 10× from identical FACTS")
        runs = []
        for run in range(1, 11):
            generated = generate_dense_canonical(facts, provider)
            runs.append({"run": run, "text": generated["text"], "token_count": tokenizer.count(generated["text"]), "usage": generated["usage"]})
            usage_events.append(("generation", generated["usage"]))
        metrics = stability_metrics([run["text"] for run in runs], tokenizer)
        first = runs[0]["text"]
        for question_index, item in enumerate(example["questions"], 1):
            evaluated = evaluate_question(first, item, provider)
            qa_rows.append({
                "example_id": example["id"], "question_index": question_index, "question": item["question"],
                "expected_answer": item["expected_answer"], "model_answer": evaluated["answer"],
                "deterministic_correct": evaluated["deterministic_correct"], "judge_correct": evaluated["judge_correct"],
                "judge_reason": evaluated["judge_reason"],
            })
            usage_events.extend((("answer", evaluated["answer_usage"]), ("judge", evaluated["judge_usage"])))
        records.append({"example_id": example["id"], "source_facts": facts, "runs": runs, "metrics": metrics})

    dense_avg = mean(record["metrics"]["token_count_mean"] for record in records)
    dense_reduction = mean(1 - record["metrics"]["token_count_mean"] / original_counts[record["example_id"]] for record in records)
    dense_correct = sum(row["judge_correct"] for row in qa_rows)
    comparison = prior_comparison()
    comparison["DENSE_CANONICAL"] = {
        "average_tokens": dense_avg, "reduction": dense_reduction,
        "judge_correct": dense_correct, "judge_accuracy": dense_correct / len(qa_rows),
        "exact_stability_rate": mean(record["metrics"]["pairwise_exact_token_rate"] for record in records),
        "average_lcp_percentage": mean(record["metrics"]["average_pairwise_lcp_percentage"] for record in records),
    }

    output = ROOT / "results"
    (output / "dense_canonical_runs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "dense_canonical_qa.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qa_rows[0]))
        writer.writeheader(); writer.writerows(qa_rows)
    with (output / "dense_canonical_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["strategy", "average_tokens", "reduction", "judge_correct", "judge_accuracy", "exact_stability_rate", "average_lcp_percentage"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({"strategy": strategy, **values} for strategy, values in comparison.items())

    print("\nCOMPARISON")
    print("Strategy          Avg tokens   Reduction   Judge QA       Exact stability   Avg pairwise LCP")
    for strategy, values in comparison.items():
        stability = "N/A" if values["exact_stability_rate"] is None else f"{values['exact_stability_rate']:.1%}"
        lcp = "N/A" if values["average_lcp_percentage"] is None else f"{values['average_lcp_percentage']:.1%}"
        print(f"{strategy:<17} {values['average_tokens']:>10.1f}   {values['reduction']:>8.1%}   {values['judge_correct']:>2}/15 ({values['judge_accuracy']:>5.1%})   {stability:>15}   {lcp:>16}")
    print("\nDENSE_CANONICAL STABILITY BY CONTEXT")
    for record in records:
        m = record["metrics"]
        print(f"Context {record['example_id']}: counts={m['token_counts']}; all_text_equal={m['all_text_identical']}; all_tokens_equal={m['all_token_sequences_identical']}; global_lcp={m['global_longest_common_token_prefix']} ({m['global_lcp_percentage']:.1%}); pairwise_similarity={m['average_pairwise_token_similarity']:.1%}; mean={m['token_count_mean']:.2f}; sd={m['token_count_stddev']:.2f}")
        print(f"FIRST REPRESENTATION:\n{record['runs'][0]['text']}\n")
    print("API USAGE")
    for scope in ("generation", "answer", "judge"):
        uses = [usage for event, usage in usage_events if event == scope]
        print(f"{scope}={sum(u.get('input_tokens', 0) for u in uses)} in/{sum(u.get('output_tokens', 0) for u in uses)} out")


if __name__ == "__main__":
    main()
