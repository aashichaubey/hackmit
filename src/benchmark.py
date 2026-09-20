"""Frozen benchmark_v1 runner: ORIGINAL, COMPACT_ENGLISH, TOKEN_OPTIMIZED."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from .benchmark_evaluate import ApiCache, CachedProvider, evaluate_blinded
from .benchmark_stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, write_outputs
from .representations import FACTS_SYSTEM, STRATEGY_INSTRUCTIONS, LLMProvider, TOKEN_SEARCH_SYSTEM, validate_semantics
from .tokenizer import TargetTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "benchmark_v1.json"
HASH_FILE = ROOT / "data" / "benchmark_v1.sha256"
RESULTS_ROOT = ROOT / "results" / "benchmark_v1"
STRATEGIES = ("ORIGINAL", "COMPACT_ENGLISH", "TOKEN_OPTIMIZED")
CANDIDATE_COUNT = 3


def dataset_hash(path: Path = DATASET) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_benchmark(items: list[dict]) -> dict:
    assert len(items) == 50, f"expected 50 contexts, got {len(items)}"
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids)), "context IDs must be unique"
    categories = Counter()
    word_counts = []
    for item in items:
        assert set(item) == {"id", "category", "context", "questions"}
        assert all(isinstance(item[key], str) and item[key].strip() for key in ("id", "category", "context"))
        assert len(item["questions"]) == 5, f"{item['id']} must have five questions"
        words = len(item["context"].split()); word_counts.append(words)
        assert 200 <= words <= 600, f"{item['id']} has {words} words"
        categories[item["category"]] += 1
        for question in item["questions"]:
            assert set(question) == {"question", "expected_answer"}
            assert all(isinstance(question[key], str) and question[key].strip() for key in question)
            assert question["expected_answer"].casefold() in item["context"].casefold(), f"answer not verbatim in context: {item['id']}"
    assert len(categories) == 10 and max(categories.values()) - min(categories.values()) <= 1
    return {"contexts": len(items), "qa_pairs": sum(len(item["questions"]) for item in items), "categories": dict(sorted(categories.items())), "min_words": min(word_counts), "max_words": max(word_counts), "mean_words": sum(word_counts) / len(word_counts)}


def load_and_freeze() -> tuple[list[dict], dict, str]:
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    status = validate_benchmark(items)
    digest = dataset_hash()
    if HASH_FILE.exists():
        frozen = HASH_FILE.read_text(encoding="utf-8").strip().split()[0]
        if frozen != digest:
            raise RuntimeError(f"frozen benchmark hash mismatch: expected {frozen}, got {digest}")
    else:
        HASH_FILE.write_text(f"{digest}  benchmark_v1.json\n", encoding="utf-8")
    return items, status, digest


def _record_usage(events: list, scope: str, result: dict) -> None:
    events.append({"scope": scope, "usage": result.get("usage", {}), "cached": result.get("cached", False)})


def compress_context(context: str, tokenizer: TargetTokenizer, provider: CachedProvider, candidate_count: int = CANDIDATE_COUNT) -> dict:
    """Question-independent compression boundary: accepts no questions or answers."""
    assert context.strip() and candidate_count == 3
    usage_events = []
    facts = provider.complete(FACTS_SYSTEM, f"SOURCE CONTEXT:\n{context}", purpose="generation", temperature=0)
    _record_usage(usage_events, "compression_generation", facts)
    compact = provider.complete(
        "Rewrite a supplied fact set without selecting among facts. Preserve all materially useful information for unknown future questions and output only the rewritten context.",
        f"Strategy: COMPACT_ENGLISH\nInstruction: {STRATEGY_INSTRUCTIONS['COMPACT_ENGLISH']} Preserve every materially useful proposition because future questions are unknown.\n\nFACTS:\n{facts['text']}",
        purpose="generation", temperature=0,
    )
    _record_usage(usage_events, "compression_generation", compact)
    original_tokens = tokenizer.count(context)
    candidates = [{"candidate": 0, "text": context, "token_count": original_tokens, "valid": True, "validation_reason": "Original baseline; valid by construction.", "source": "original"}]
    for attempt in range(1, candidate_count + 1):
        generated = provider.complete(
            TOKEN_SEARCH_SYSTEM,
            f"SEARCH ATTEMPT {attempt} OF {candidate_count}. Produce one self-contained encoding of all supplied propositions. Token count is measured externally; do not estimate or report it.\n\nSOURCE SEMANTIC CONTENT:\n{facts['text']}",
            purpose="generation", temperature=float(os.getenv("TOKEN_SEARCH_TEMPERATURE", "0.8")),
        )
        _record_usage(usage_events, "token_optimization_generation", generated)
        validation = validate_semantics(facts["text"], generated["text"], provider)
        # validate_semantics returns the underlying cached provider usage.
        usage_events.append({"scope": "semantic_validation", "usage": validation.get("usage", {}), "cached": validation.get("cached", False)})
        candidates.append({"candidate": attempt, "text": generated["text"], "token_count": tokenizer.count(generated["text"]), "valid": validation["valid"], "validation_reason": validation["reason"], "source": "generated"})
    valid = [candidate for candidate in candidates if candidate["valid"]]
    winner = min(valid, key=lambda candidate: (candidate["token_count"], candidate["candidate"]))
    valid_generated = [candidate for candidate in candidates[1:] if candidate["valid"]]
    return {
        "facts": facts["text"], "compact_english": compact["text"], "candidates": candidates, "winner": winner,
        "candidate_zero_selected": winner["candidate"] == 0,
        "fallback_prevented_larger": winner["candidate"] == 0 and bool(valid_generated) and min(c["token_count"] for c in valid_generated) > original_tokens,
        "usage_events": usage_events,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int)
    group.add_argument("--all", action="store_true")
    group.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    items, validation, digest = load_and_freeze()
    print(f"VALIDATION_OK contexts={validation['contexts']} qa_pairs={validation['qa_pairs']} categories={validation['categories']} words={validation['min_words']}..{validation['max_words']}")
    print(f"FROZEN_BENCHMARK_SHA256={digest}")
    if args.validate_only:
        return
    if not args.all and args.limit is None:
        parser.error("specify --limit N or --all")
    limit = 50 if args.all else args.limit
    if not 1 <= limit <= 50:
        parser.error("--limit must be between 1 and 50")
    if not args.all:
        by_category = defaultdict(list)
        for item in items: by_category[item["category"]].append(item)
        interleaved = []
        for position in range(5):
            for category in sorted(by_category): interleaved.append(by_category[category][position])
        items = interleaved[:limit]

    base_provider = LLMProvider()
    provider = CachedProvider(base_provider, ApiCache(RESULTS_ROOT / "cache" / "api_cache.json"))
    target_model = os.getenv("TARGET_MODEL", provider.answer_model)
    tokenizer = TargetTokenizer(target_model)
    calls_per_context = 2 + CANDIDATE_COUNT + CANDIDATE_COUNT + len(STRATEGIES) * 5 * 2
    print(f"MODELS generation={provider.generation_model} answer={provider.answer_model} judge={provider.judge_model}")
    print(f"TOKENIZER target={target_model} encoding={tokenizer.name}")
    print(f"PLANNED_API_CALLS contexts={limit} per_context={calls_per_context} total={limit*calls_per_context}")

    output = RESULTS_ROOT / ("full_50" if args.all else f"pilot_{limit}")
    output.mkdir(parents=True, exist_ok=True)
    rep_rows, qa_rows, records, usage_events = [], [], [], []
    for index, item in enumerate(items, 1):
        print(f"[{index}/{limit}] {item['id']}")
        compressed = compress_context(item["context"], tokenizer, provider)
        usage_events.extend(compressed["usage_events"])
        representations = {
            "ORIGINAL": item["context"], "COMPACT_ENGLISH": compressed["compact_english"],
            "TOKEN_OPTIMIZED": compressed["winner"]["text"],
        }
        original_tokens = tokenizer.count(item["context"])
        records.append({"context_id": item["id"], "category": item["category"], "facts": compressed["facts"], "representations": representations, "token_optimized_candidates": compressed["candidates"], "candidate_zero_selected": compressed["candidate_zero_selected"], "fallback_prevented_larger": compressed["fallback_prevented_larger"]})
        for strategy, text in representations.items():
            count = tokenizer.count(text)
            rep_rows.append({
                "context_id": item["id"], "category": item["category"], "strategy": strategy,
                "original_tokens": original_tokens, "compressed_tokens": count, "tokens_saved": original_tokens - count,
                "reduction_percent": 1 - count / original_tokens,
                "candidate_zero_selected": strategy == "TOKEN_OPTIMIZED" and compressed["candidate_zero_selected"],
                "fallback_prevented_larger": strategy == "TOKEN_OPTIMIZED" and compressed["fallback_prevented_larger"],
            })
            for question_index, question in enumerate(item["questions"], 1):
                evaluated = evaluate_blinded(text, question, provider)
                usage_events.extend((
                    {"scope": "downstream_qa", "usage": evaluated["answer_usage"], "cached": evaluated["answer_cached"]},
                    {"scope": "qa_judging", "usage": evaluated["judge_usage"], "cached": evaluated["judge_cached"]},
                ))
                qa_rows.append({
                    "context_id": item["id"], "category": item["category"], "strategy": strategy,
                    "question_index": question_index, "question": question["question"], "expected_answer": question["expected_answer"],
                    "model_answer": evaluated["model_answer"], "deterministic_correct": evaluated["deterministic_correct"],
                    "deterministic_method": evaluated["deterministic_method"], "judge_correct": evaluated["judge_correct"],
                    "judge_reason": evaluated["judge_reason"],
                })

    write_csv(output / "representations.csv", rep_rows)
    write_csv(output / "qa_results.csv", qa_rows)
    (output / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary, paired = write_outputs(output, rep_rows, qa_rows)
    incurred = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})
    for event in usage_events:
        if not event["cached"]:
            bucket = incurred[event["scope"]]; bucket["calls"] += 1
            bucket["input_tokens"] += int(event["usage"].get("input_tokens", 0)); bucket["output_tokens"] += int(event["usage"].get("output_tokens", 0))
    total_in = sum(v["input_tokens"] for v in incurred.values()); total_out = sum(v["output_tokens"] for v in incurred.values())
    approximate_cost = total_in / 1_000_000 * .40 + total_out / 1_000_000 * 1.60
    original_context_tokens = sum(tokenizer.count(item["context"]) for item in items)
    search_tokens = incurred["token_optimization_generation"]["input_tokens"] + incurred["token_optimization_generation"]["output_tokens"] + incurred["semantic_validation"]["input_tokens"] + incurred["semantic_validation"]["output_tokens"]
    usage_report = {"by_scope": incurred, "total_input_tokens": total_in, "total_output_tokens": total_out, "approximate_cost_usd": approximate_cost, "original_context_tokens": original_context_tokens, "compression_search_tokens": search_tokens, "compression_search_tokens_per_original_context_token": search_tokens / original_context_tokens, "pricing_assumption_per_million": {"input": .40, "output": 1.60}}
    (output / "usage.json").write_text(json.dumps(usage_report, indent=2) + "\n")
    manifest = {"benchmark_sha256": digest, "contexts": limit, "qa_pairs": limit * 5, "models": {"generation": provider.generation_model, "answer": provider.answer_model, "judge": provider.judge_model}, "target_model": target_model, "tokenizer": tokenizer.name, "candidate_count": CANDIDATE_COUNT, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nSUMMARY")
    for row in summary:
        print(f"{row['strategy']:<18} mean_tokens={row['mean_tokens']:.1f} median={row['median_tokens']:.1f} reduction={row['mean_reduction_percent']:.1%} judge={row['judge_accuracy']:.1%} CI_reduction=[{row['reduction_ci_low']:.1%},{row['reduction_ci_high']:.1%}] QA_diff={row['qa_diff_vs_original']:.1%} CI=[{row['qa_diff_ci_low']:.1%},{row['qa_diff_ci_high']:.1%}]")
    print("PAIRED")
    for strategy in ("COMPACT_ENGLISH", "TOKEN_OPTIMIZED"):
        group = [row for row in paired if row["strategy"] == strategy]
        print(f"{strategy}: saved_without_loss={sum(r['saved_without_qa_loss'] for r in group)} failures={sum(r['caused_failure'] for r in group)} candidate0={sum(r['candidate_zero_selected'] for r in group)}")
    print("USAGE")
    for scope, values in incurred.items(): print(f"{scope}: calls={values['calls']} in={values['input_tokens']} out={values['output_tokens']}")
    print(f"TOTAL cost≈${approximate_cost:.4f}; search/original_tokens={search_tokens/original_context_tokens:.2f}")


if __name__ == "__main__":
    main()
