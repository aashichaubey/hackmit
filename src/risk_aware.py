"""Question-independent RISK_AWARE_COMPACT experiment on frozen failure contexts."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

from dotenv import load_dotenv

from .benchmark import DATASET, HASH_FILE, dataset_hash, validate_benchmark
from .benchmark_evaluate import ApiCache, CachedProvider, evaluate_blinded
from .representations import LLMProvider
from .tokenizer import TargetTokenizer

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "results" / "benchmark_v1" / "full_50"
OUTPUT = ROOT / "results" / "risk_aware_v1"

RISK_CATEGORIES = (
    "NEGATION", "CONDITION", "EXCEPTION", "ORDERING", "REQUIREMENT", "PERMISSION",
    "QUANTITY", "TIME_DATE", "COMPARISON_THRESHOLD", "CAUSAL_DEPENDENCY", "LOCATION_ENTITY",
)

RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "protected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "categories": {"type": "array", "items": {"type": "string", "enum": list(RISK_CATEGORIES)}},
                },
                "required": ["text", "categories"], "additionalProperties": False,
            },
        }
    },
    "required": ["protected"], "additionalProperties": False,
}

RISK_SYSTEM = (
    "Identify semantically fragile information in a context for question-independent compression. Return minimal VERBATIM "
    "spans copied exactly from the context and label applicable categories: NEGATION; CONDITION; EXCEPTION; ORDERING; "
    "REQUIREMENT; PERMISSION; QUANTITY; TIME_DATE; COMPARISON_THRESHOLD; CAUSAL_DEPENDENCY; LOCATION_ENTITY. Protect "
    "named entities or locations only where changing them alters a factual proposition. Include conditions, exceptions, "
    "negations, procedural sequence, obligations, permissions, thresholds, quantities, dates/times, and causal links. "
    "Do not rewrite spans, infer questions, or protect generic discourse merely because it is grammatical."
)

COMPRESS_SYSTEM = (
    "Produce RISK_AWARE_COMPACT: concise natural language preserving all materially useful information for unknown future "
    "questions. Compress aggressively by removing discourse filler, redundant phrasing, unnecessary transitions, verbosity, "
    "and repeated explanation. Preserve the exact semantic content of every protected span. Do not reverse ordering, remove "
    "negations, merge distinct conditions, remove exceptions, change thresholds/quantities/dates/times, turn requirements "
    "into suggestions, turn permissions into requirements, merge nearby distinct facts, or remove meaning-changing causal "
    "dependencies. Do not output JSON, triples, semantic IR, or dense symbolic notation. Primarily use concise natural "
    "language; lightweight abbreviations or symbols are allowed only when unambiguous. Output only the compressed context."
)


def detect_risks(context: str, provider: CachedProvider) -> dict:
    """Question-independent risk detector; accepts only context and provider."""
    assert context.strip()
    result = provider.complete_schema(RISK_SYSTEM, f"CONTEXT:\n{context}", schema_name="risk_spans", schema=RISK_SCHEMA, temperature=0)
    protected = []
    seen = set()
    for item in result["data"]["protected"]:
        text = item["text"].strip()
        categories = sorted(set(item["categories"]))
        if text and text not in context:
            start = context.casefold().find(text.casefold())
            if start >= 0:
                text = context[start:start + len(text)]
            else:
                def terms(value: str) -> set[str]:
                    words = re.findall(r"[a-z0-9]+", value.casefold())
                    return {re.sub(r"(?:ing|ed|es|s)$", "", word) for word in words if len(word) > 2}
                query_terms = terms(text)
                sentences = re.split(r"(?<=[.!?])\s+", context)
                scored = [(len(query_terms & terms(sentence)) / max(1, len(query_terms)), sentence) for sentence in sentences]
                coverage, sentence = max(scored, default=(0, ""), key=lambda pair: pair[0])
                if coverage >= .6:
                    text = sentence
        if not text or text not in context:
            raise ValueError(f"risk detector returned non-verbatim span: {text!r}")
        key = (text, tuple(categories))
        if key not in seen:
            protected.append({"text": text, "categories": categories}); seen.add(key)
    protected.sort(key=lambda item: (context.index(item["text"]), item["text"], item["categories"]))
    return {"protected": protected, "usage": result.get("usage", {}), "cached": result.get("cached", False)}


def compress_risk_aware(context: str, protected: list[dict], provider: CachedProvider) -> dict:
    """Question-independent compressor; accepts no downstream evaluation data."""
    assert context.strip()
    lines = [f"- [{','.join(item['categories'])}] {item['text']}" for item in protected]
    prompt = f"ORIGINAL CONTEXT:\n{context}\n\nPROTECTED INFORMATION (verbatim source spans):\n" + ("\n".join(lines) if lines else "(none detected)")
    return provider.complete(COMPRESS_SYSTEM, prompt, purpose="generation", temperature=0)


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _relevant_sentence(context: str, expected: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", context)
    return next((sentence for sentence in sentences if expected.casefold() in sentence.casefold()), "")


def _categories_for_sentence(sentence: str, protected: list[dict]) -> list[str]:
    categories = set()
    for item in protected:
        if item["text"] in sentence or sentence in item["text"]:
            categories.update(item["categories"])
    return sorted(categories)


def main() -> None:
    load_dotenv(ROOT / ".env")
    digest = dataset_hash()
    frozen = HASH_FILE.read_text(encoding="utf-8").split()[0]
    print(f"FROZEN_BENCHMARK_SHA256 computed={digest} stored={frozen}")
    if digest != frozen:
        raise RuntimeError("benchmark hash mismatch")
    items = json.loads(DATASET.read_text(encoding="utf-8")); validate_benchmark(items)
    by_id = {item["id"]: item for item in items}
    prior_rep_rows = _read_csv(PRIOR / "representations.csv")
    prior_qa_rows = _read_csv(PRIOR / "qa_results.csv")
    for row in prior_qa_rows:
        row["question_index"] = int(row["question_index"])
        row["deterministic_correct"] = row["deterministic_correct"] == "True"
        row["judge_correct"] = row["judge_correct"] == "True"
    # Preserve the historical post-hoc subset exactly: it selected contexts
    # whose Compact English *net score* was lower than Original. The corrected
    # paired report separately identifies any question-level loss and finds 20
    # contexts; this completed exploratory experiment remains the original 16.
    selected_ids = []
    for context_id in sorted(by_id):
        original_correct = sum(row["judge_correct"] for row in prior_qa_rows if row["context_id"] == context_id and row["strategy"] == "ORIGINAL")
        compact_correct = sum(row["judge_correct"] for row in prior_qa_rows if row["context_id"] == context_id and row["strategy"] == "COMPACT_ENGLISH")
        if compact_correct < original_correct:
            selected_ids.append(context_id)
    assert len(selected_ids) == 16
    prior_records = {record["context_id"]: record for record in json.loads((PRIOR / "records.json").read_text(encoding="utf-8"))}

    base = LLMProvider(); provider = CachedProvider(base, ApiCache(OUTPUT / "cache" / "api_cache.json"))
    target_model = os.getenv("TARGET_MODEL", provider.answer_model); tokenizer = TargetTokenizer(target_model)
    print(f"SELECTED_CONTEXTS={len(selected_ids)} MODELS generation={provider.generation_model} answer={provider.answer_model} judge={provider.judge_model} TOKENIZER={tokenizer.name} PLANNED_CALLS={len(selected_ids) * 12}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    risk_records, risk_qa, usage_events = [], [], []
    for position, context_id in enumerate(selected_ids, 1):
        item = by_id[context_id]
        print(f"[{position}/16] {context_id}")
        detected = detect_risks(item["context"], provider)
        usage_events.append({"scope": "risk_detection", "usage": detected["usage"], "cached": detected["cached"]})
        compressed = compress_risk_aware(item["context"], detected["protected"], provider)
        usage_events.append({"scope": "risk_aware_generation", "usage": compressed.get("usage", {}), "cached": compressed.get("cached", False)})
        original_tokens = tokenizer.count(item["context"]); compressed_tokens = tokenizer.count(compressed["text"])
        protected_tokens = sum(tokenizer.count(span["text"]) for span in detected["protected"])
        record = {
            "context_id": context_id, "category": item["category"], "protected": detected["protected"],
            "protected_span_count": len(detected["protected"]), "protected_tokens": protected_tokens,
            "protected_token_fraction": protected_tokens / original_tokens, "original_tokens": original_tokens,
            "risk_aware_tokens": compressed_tokens, "reduction_percent": 1 - compressed_tokens / original_tokens,
            "representation": compressed["text"],
        }
        risk_records.append(record)
        for question_index, question in enumerate(item["questions"], 1):
            evaluated = evaluate_blinded(compressed["text"], question, provider)
            usage_events.extend((
                {"scope": "downstream_qa", "usage": evaluated["answer_usage"], "cached": evaluated["answer_cached"]},
                {"scope": "qa_judging", "usage": evaluated["judge_usage"], "cached": evaluated["judge_cached"]},
            ))
            risk_qa.append({
                "context_id": context_id, "category": item["category"], "strategy": "RISK_AWARE_COMPACT",
                "question_index": question_index, "question": question["question"], "expected_answer": question["expected_answer"],
                "model_answer": evaluated["model_answer"], "deterministic_correct": evaluated["deterministic_correct"],
                "deterministic_method": evaluated["deterministic_method"], "judge_correct": evaluated["judge_correct"],
                "judge_reason": evaluated["judge_reason"],
            })

    strategies = ("ORIGINAL", "COMPACT_ENGLISH", "RISK_AWARE_COMPACT")
    comparison_qa = [row for row in prior_qa_rows if row["context_id"] in selected_ids and row["strategy"] in {"ORIGINAL", "COMPACT_ENGLISH"}] + risk_qa
    rep_lookup = {(row["context_id"], row["strategy"]): row for row in prior_rep_rows}
    risk_lookup = {record["context_id"]: record for record in risk_records}
    summaries = []
    for strategy in strategies:
        if strategy == "RISK_AWARE_COMPACT":
            counts = [r["risk_aware_tokens"] for r in risk_records]; reductions = [r["reduction_percent"] for r in risk_records]
        else:
            rows = [rep_lookup[(cid, strategy)] for cid in selected_ids]
            counts = [int(row["compressed_tokens"]) for row in rows]; reductions = [float(row["reduction_percent"]) for row in rows]
        qas = [row for row in comparison_qa if row["strategy"] == strategy]
        additional = 0
        for row in qas:
            original = next(q for q in comparison_qa if q["context_id"] == row["context_id"] and q["question_index"] == row["question_index"] and q["strategy"] == "ORIGINAL")
            additional += bool(original["judge_correct"]) and not bool(row["judge_correct"])
        summaries.append({
            "strategy": strategy, "contexts": 16, "qa_pairs": len(qas), "mean_tokens": mean(counts),
            "mean_reduction": mean(reductions), "median_reduction": median(reductions), "reduction_stddev": pstdev(reductions),
            "deterministic_accuracy": mean(bool(row["deterministic_correct"]) for row in qas),
            "judge_accuracy": mean(bool(row["judge_correct"]) for row in qas), "additional_failures_vs_original": additional,
        })

    official = {(row["context_id"], row["question_index"], row["strategy"]): row for row in comparison_qa}
    transitions = []
    recovered = new_failures = 0
    for cid in selected_ids:
        record = risk_lookup[cid]; item = by_id[cid]
        for index, question in enumerate(item["questions"], 1):
            compact = official[(cid, index, "COMPACT_ENGLISH")]; risk = official[(cid, index, "RISK_AWARE_COMPACT")]
            compact_ok, risk_ok = bool(compact["judge_correct"]), bool(risk["judge_correct"])
            transition = "recovered" if not compact_ok and risk_ok else "new_failure" if compact_ok and not risk_ok else None
            if transition:
                recovered += transition == "recovered"; new_failures += transition == "new_failure"
                relevant = _relevant_sentence(item["context"], question["expected_answer"])
                transitions.append({
                    "transition": transition, "context_id": cid, "category": item["category"], "question": question["question"],
                    "expected_answer": question["expected_answer"], "original_relevant_information": relevant,
                    "compact_english_representation": prior_records[cid]["representations"]["COMPACT_ENGLISH"],
                    "risk_aware_representation": record["representation"],
                    "compact_model_answer": compact["model_answer"], "risk_aware_model_answer": risk["model_answer"],
                    "risk_categories": ",".join(_categories_for_sentence(relevant, record["protected"])),
                })

    evaluator_errors = []
    for row in risk_qa:
        if not row["judge_correct"] and row["deterministic_correct"]:
            evaluator_errors.append({**row, "audit_note": "Possible evaluator error: deterministic normalized match passed; official judge result preserved."})
    category_counts = Counter(category for record in risk_records for span in record["protected"] for category in span["categories"])
    # Simple descriptive association, not a predictive model.
    xs = [record["protected_token_fraction"] for record in risk_records]; ys = [record["reduction_percent"] for record in risk_records]
    xmean, ymean = mean(xs), mean(ys)
    denom = sum((x - xmean) ** 2 for x in xs) * sum((y - ymean) ** 2 for y in ys)
    correlation = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / denom ** .5 if denom else 0

    (OUTPUT / "records.json").write_text(json.dumps(risk_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(OUTPUT / "qa_results.csv", risk_qa)
    _write_csv(OUTPUT / "summary.csv", summaries)
    if transitions: _write_csv(OUTPUT / "failure_transitions.csv", transitions)
    if evaluator_errors: _write_csv(OUTPUT / "possible_evaluator_errors.csv", evaluator_errors)
    ablation_rows = [{k: record[k] for k in ("context_id", "category", "protected_span_count", "protected_tokens", "protected_token_fraction", "original_tokens", "risk_aware_tokens", "reduction_percent")} for record in risk_records]
    _write_csv(OUTPUT / "ablation.csv", ablation_rows)

    incurred = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    for event in usage_events:
        if not event["cached"]:
            bucket = incurred[event["scope"]]; bucket["calls"] += 1
            bucket["input_tokens"] += int(event["usage"].get("input_tokens", 0)); bucket["output_tokens"] += int(event["usage"].get("output_tokens", 0))
    total_in = sum(v["input_tokens"] for v in incurred.values()); total_out = sum(v["output_tokens"] for v in incurred.values())
    cost = total_in / 1_000_000 * .4 + total_out / 1_000_000 * 1.6
    report = {"benchmark_sha256": digest, "selected_context_ids": selected_ids, "recovered_questions": recovered, "new_failures": new_failures, "risk_category_span_counts": category_counts, "protected_fraction_reduction_correlation": correlation, "usage": incurred, "total_input_tokens": total_in, "total_output_tokens": total_out, "approximate_cost_usd": cost}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\nSUMMARY")
    for row in summaries: print(f"{row['strategy']:<20} tokens={row['mean_tokens']:.1f} reduction={row['mean_reduction']:.1%} median={row['median_reduction']:.1%} judge={row['judge_accuracy']:.1%} deterministic={row['deterministic_accuracy']:.1%} additional_failures={row['additional_failures_vs_original']}")
    print(f"RECOVERED={recovered} NEW_FAILURES={new_failures} protected_fraction/reduction_correlation={correlation:.3f}")
    print(f"RISK_CATEGORIES={dict(category_counts)}")
    print(f"USAGE in={total_in} out={total_out} cost≈${cost:.4f}")


if __name__ == "__main__":
    main()
