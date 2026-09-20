"""Compiler-style semantic extraction, normalization, and deterministic serialization."""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

from dotenv import load_dotenv

from .dense_canonical import lcp_length
from .evaluate import evaluate_question
from .generate_dataset import DEFAULT_OUTPUT, validate_examples
from .representations import LLMProvider
from .tokenizer import TargetTokenizer

ROOT = Path(__file__).resolve().parents[1]

RELATIONS = (
    "is", "has", "lacks", "occurs", "located", "starts", "ends", "moves", "produces", "assembles",
    "requires", "allows", "prohibits", "stores", "arrives", "signs", "disables", "handles", "tests",
    "submits", "reviews", "disposes", "publishes", "assigns", "departs", "includes", "announces", "marks",
    "opens", "example", "causes", "covers", "changes",
)

IR_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string", "enum": list(RELATIONS)},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "time": {"type": "array", "items": {"type": "string"}},
                    "location": {"type": "array", "items": {"type": "string"}},
                    "condition": {"type": "array", "items": {"type": "string"}},
                    "exception": {"type": "array", "items": {"type": "string"}},
                    "qualifier": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["subject", "relation", "objects", "time", "location", "condition", "exception", "qualifier"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

EXTRACTION_SYSTEM = (
    "Convert the source context into a complete reusable semantic fact set for unknown future questions. Preserve all "
    "materially useful propositions: entities, relations, values, times, locations, conditions, exceptions, causes, and "
    "qualifiers. Use one atomic fact per record when practical. Use consistent concise entity names. Put unordered values "
    "in objects. Do not summarize for any presumed question. You will not receive downstream questions."
)


def extract_ir(context: str, provider: LLMProvider) -> dict:
    """Question-independent LLM extraction into strict schema."""
    assert isinstance(context, str) and context.strip()
    result = provider.complete_schema(EXTRACTION_SYSTEM, f"SOURCE CONTEXT:\n{context}", schema_name="semantic_ir", schema=IR_SCHEMA, temperature=0)
    validate_ir(result["data"])
    return result


def validate_ir(ir: dict) -> None:
    assert isinstance(ir, dict) and set(ir) == {"facts"} and isinstance(ir["facts"], list)
    required = {"subject", "relation", "objects", "time", "location", "condition", "exception", "qualifier"}
    for fact in ir["facts"]:
        assert set(fact) == required and fact["relation"] in RELATIONS
        assert isinstance(fact["subject"], str) and fact["subject"].strip()
        for field in required - {"subject", "relation"}:
            assert isinstance(fact[field], list) and all(isinstance(value, str) for value in fact[field])


def _space(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def _normalize_time(value: str) -> str:
    value = _space(value).casefold()
    def convert(match: re.Match) -> str:
        hour = int(match.group(1)); minute = int(match.group(2) or 0); meridiem = match.group(3).replace(".", "")
        if meridiem == "pm" and hour != 12: hour += 12
        if meridiem == "am" and hour == 12: hour = 0
        return f"{hour:02d}:{minute:02d}"
    value = re.sub(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)\b", convert, value)
    value = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: f"{int(m.group(1)):02d}:{m.group(2)}", value)
    return value


def _normalize_number(value: str) -> str:
    value = _space(value)
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    value = re.sub(r"\bzero\b", "0", value, flags=re.I)
    return value


def _entity(value: str) -> str:
    return _space(value).casefold().replace("&", "and")


def _values(values: list[str], *, time: bool = False) -> list[str]:
    normalizer = _normalize_time if time else _normalize_number
    normalized = {normalizer(value).casefold() for value in values if _space(value)}
    return sorted(normalized)


def normalize_ir(ir: dict) -> dict:
    """Pure deterministic normalization, ordering, and deduplication."""
    validate_ir(ir)
    facts = set()
    for raw in ir["facts"]:
        fact = (
            _entity(raw["subject"]), raw["relation"].casefold(), tuple(_values(raw["objects"])),
            tuple(_values(raw["time"], time=True)), tuple(_values(raw["location"])),
            tuple(_values(raw["condition"])), tuple(_values(raw["exception"])), tuple(_values(raw["qualifier"])),
        )
        facts.add(fact)
    ordered = sorted(facts)
    return {"facts": [
        {"subject": f[0], "relation": f[1], "objects": list(f[2]), "time": list(f[3]), "location": list(f[4]), "condition": list(f[5]), "exception": list(f[6]), "qualifier": list(f[7])}
        for f in ordered
    ]}


RELATION_SHORT = {
    "is": "is", "has": "has", "lacks": "lacks", "occurs": "on", "located": "at", "starts": "starts",
    "ends": "ends", "moves": "moves", "produces": "makes", "assembles": "builds", "requires": "needs",
    "allows": "allows", "prohibits": "no", "stores": "stores", "arrives": "arrives", "signs": "signs",
    "disables": "disables", "handles": "handles", "tests": "tests", "submits": "submits", "reviews": "reviews",
    "disposes": "discards", "publishes": "published", "assigns": "assigns", "departs": "departs",
    "includes": "includes", "announces": "announces", "marks": "marks", "opens": "opens", "example": "example",
    "causes": "causes", "covers": "covers", "changes": "changes",
}


@dataclass(frozen=True)
class Format:
    name: str
    field_sep: str
    fact_sep: str
    list_sep: str
    relation_mode: str


FORMATS = (
    Format("pipe_full", "|", ";", ",", "full"), Format("pipe_short", "|", ";", ",", "short"),
    Format("colon_full", ":", ";", ",", "full"), Format("colon_short", ":", ";", ",", "short"),
    Format("arrow_full", "→", ";", ",", "full"), Format("arrow_short", "→", ";", ",", "short"),
)


def _escape(value: str, fmt: Format) -> str:
    reserved = {"\\", fmt.field_sep, fmt.fact_sep, fmt.list_sep, "@", "#", "?", "!", "~"}
    output = value.replace("\\", "\\\\")
    for char in sorted(reserved - {"\\"}, key=len, reverse=True):
        if char.strip(): output = output.replace(char, "\\" + char)
    return output


def serialize_ir(ir: dict, fmt: Format) -> str:
    """Pure deterministic non-generative serializer."""
    validate_ir(ir)
    rows = []
    for fact in ir["facts"]:
        relation = RELATION_SHORT[fact["relation"]] if fmt.relation_mode == "short" else fact["relation"]
        fields = [_escape(fact["subject"], fmt), _escape(relation, fmt), fmt.list_sep.join(_escape(value, fmt) for value in fact["objects"])]
        extras = (("@", "time"), ("#", "location"), ("?", "condition"), ("!", "exception"), ("~", "qualifier"))
        for marker, key in extras:
            if fact[key]: fields.append(marker + fmt.list_sep.join(_escape(value, fmt) for value in fact[key]))
        rows.append(fmt.field_sep.join(fields))
    return fmt.fact_sep.join(rows)


def choose_format(irs: list[dict], tokenizer: TargetTokenizer) -> tuple[Format, list[dict]]:
    scores = []
    for fmt in FORMATS:
        counts = [tokenizer.count(serialize_ir(ir, fmt)) for ir in irs]
        scores.append({"format": fmt.name, "counts": counts, "total_tokens": sum(counts)})
    winner_name = min(scores, key=lambda row: (row["total_tokens"], row["format"]))["format"]
    return next(fmt for fmt in FORMATS if fmt.name == winner_name), scores


def _fact_set(ir: dict) -> set[str]:
    return {json.dumps(fact, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for fact in ir["facts"]}


def extraction_metrics(irs: list[dict], texts: list[str], tokenizer: TargetTokenizer) -> dict:
    sequences = [tokenizer.encode(text) for text in texts]
    pairs = list(combinations(range(10), 2))
    fact_sets = [_fact_set(ir) for ir in irs]
    jaccard = [len(fact_sets[i] & fact_sets[j]) / max(1, len(fact_sets[i] | fact_sets[j])) for i, j in pairs]
    fact_exact = [fact_sets[i] == fact_sets[j] for i, j in pairs]
    text_exact = [texts[i] == texts[j] for i, j in pairs]
    token_exact = [sequences[i] == sequences[j] for i, j in pairs]
    prefix = [lcp_length(sequences[i], sequences[j]) / max(1, min(len(sequences[i]), len(sequences[j]))) for i, j in pairs]
    counts = [len(sequence) for sequence in sequences]
    global_lcp = min(lcp_length(sequences[0], sequence) for sequence in sequences[1:])
    return {
        "structured_fact_exact_rate": mean(fact_exact), "structured_fact_jaccard": mean(jaccard),
        "exact_text_rate": mean(text_exact), "exact_token_rate": mean(token_exact),
        "average_pairwise_lcp_percentage": mean(prefix), "global_lcp_tokens": global_lcp,
        "global_lcp_percentage": global_lcp / max(1, min(counts)),
        "token_count_mean": mean(counts), "token_count_stddev": pstdev(counts), "token_counts": counts,
    }


def _load_comparison() -> dict[str, dict]:
    prior_rows = list(csv.DictReader((ROOT / "results" / "qa_results.csv").open(encoding="utf-8")))
    comparison = {}
    for strategy in ("ORIGINAL", "COMPACT_ENGLISH", "TOKEN_OPTIMIZED"):
        group = [row for row in prior_rows if row["strategy"] == strategy]
        assert len(group) == 15
        contexts = {row["example_id"]: row for row in group}
        comparison[strategy] = {
            "average_tokens": mean(int(row["token_count"]) for row in contexts.values()),
            "reduction": mean(float(row["compression_ratio"]) for row in contexts.values()),
            "judge_correct": sum(row["judge_correct"] == "True" for row in group),
            "judge_accuracy": sum(row["judge_correct"] == "True" for row in group) / 15,
            "exact_end_to_end_stability": 1.0 if strategy == "ORIGINAL" else None,
            "average_prefix_stability": 1.0 if strategy == "ORIGINAL" else None,
        }
    dense_rows = list(csv.DictReader((ROOT / "results" / "dense_canonical_summary.csv").open(encoding="utf-8")))
    dense = next(row for row in dense_rows if row["strategy"] == "DENSE_CANONICAL")
    comparison["DENSE_CANONICAL"] = {
        "average_tokens": float(dense["average_tokens"]), "reduction": float(dense["reduction"]),
        "judge_correct": int(dense["judge_correct"]), "judge_accuracy": float(dense["judge_accuracy"]),
        "exact_end_to_end_stability": float(dense["exact_stability_rate"]),
        "average_prefix_stability": float(dense["average_lcp_percentage"]),
    }
    return comparison


def main() -> None:
    load_dotenv(ROOT / ".env")
    examples = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    validate_examples(examples)
    assert len(examples) == 3 and all(len(example["questions"]) == 5 for example in examples)
    provider = LLMProvider()
    target_model = os.getenv("TARGET_MODEL", provider.answer_model)
    tokenizer = TargetTokenizer(target_model)
    print(f"Provider={provider.provider}; extractor={provider.generation_model}; answer={provider.answer_model}; judge={provider.judge_model}; tokenizer={tokenizer.name}")

    extraction_runs, usage_events = {}, []
    for example in examples:
        print(f"Context {example['id']}: extracting structured IR 10× (questions withheld)")
        runs = []
        for run_index in range(1, 11):
            extracted = extract_ir(example["context"], provider)
            normalized = normalize_ir(extracted["data"])
            runs.append({"run": run_index, "raw_ir": extracted["data"], "normalized_ir": normalized, "usage": extracted["usage"]})
            usage_events.append(("extraction", extracted["usage"]))
        extraction_runs[example["id"]] = runs

    # Format selection sees only normalized semantic IR from the first extraction of each frozen context.
    first_irs = [extraction_runs[example["id"]][0]["normalized_ir"] for example in examples]
    selected_format, format_scores = choose_format(first_irs, tokenizer)
    print(f"Selected deterministic format={selected_format.name}")

    records, qa_rows = [], []
    for example in examples:
        runs = extraction_runs[example["id"]]
        texts = [serialize_ir(run["normalized_ir"], selected_format) for run in runs]
        metrics = extraction_metrics([run["normalized_ir"] for run in runs], texts, tokenizer)

        # Serializer-only test: same exact normalized IR, 100 ordinary-Python calls.
        serializer_texts = [serialize_ir(runs[0]["normalized_ir"], selected_format) for _ in range(100)]
        serializer_sequences = [tokenizer.encode(text) for text in serializer_texts]
        serializer_stability = {
            "runs": 100,
            "exact_text_rate": sum(text == serializer_texts[0] for text in serializer_texts) / 100,
            "exact_token_rate": sum(sequence == serializer_sequences[0] for sequence in serializer_sequences) / 100,
        }
        assert serializer_stability["exact_text_rate"] == 1.0 and serializer_stability["exact_token_rate"] == 1.0

        first_text = texts[0]
        for question_index, item in enumerate(example["questions"], 1):
            evaluated = evaluate_question(first_text, item, provider)
            qa_rows.append({
                "example_id": example["id"], "question_index": question_index, "question": item["question"],
                "expected_answer": item["expected_answer"], "model_answer": evaluated["answer"],
                "deterministic_correct": evaluated["deterministic_correct"], "judge_correct": evaluated["judge_correct"],
                "judge_reason": evaluated["judge_reason"],
            })
            usage_events.extend((("answer", evaluated["answer_usage"]), ("judge", evaluated["judge_usage"])))
        records.append({
            "example_id": example["id"], "selected_format": selected_format.name,
            "runs": [{**run, "serialized": text, "token_count": tokenizer.count(text)} for run, text in zip(runs, texts)],
            "serializer_only_stability": serializer_stability, "end_to_end_metrics": metrics,
        })

    original_counts = {example["id"]: tokenizer.count(example["context"]) for example in examples}
    compiler_avg = mean(record["end_to_end_metrics"]["token_count_mean"] for record in records)
    compiler_reduction = mean(1 - record["end_to_end_metrics"]["token_count_mean"] / original_counts[record["example_id"]] for record in records)
    compiler_correct = sum(row["judge_correct"] for row in qa_rows)
    comparison = _load_comparison()
    comparison["SEMANTIC_COMPILER"] = {
        "average_tokens": compiler_avg, "reduction": compiler_reduction,
        "judge_correct": compiler_correct, "judge_accuracy": compiler_correct / 15,
        "exact_end_to_end_stability": mean(record["end_to_end_metrics"]["exact_token_rate"] for record in records),
        "average_prefix_stability": mean(record["end_to_end_metrics"]["average_pairwise_lcp_percentage"] for record in records),
    }

    output = ROOT / "results"
    (output / "semantic_compiler_runs.json").write_text(json.dumps({"format_scores": format_scores, "selected_format": selected_format.name, "contexts": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "semantic_compiler_qa.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qa_rows[0])); writer.writeheader(); writer.writerows(qa_rows)
    with (output / "semantic_compiler_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["strategy", "average_tokens", "reduction", "judge_correct", "judge_accuracy", "exact_end_to_end_stability", "average_prefix_stability"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({"strategy": strategy, **values} for strategy, values in comparison.items())

    print("\nFORMAT SEARCH (no QA information used)")
    for score in format_scores:
        marker = " <- selected" if score["format"] == selected_format.name else ""
        print(f"{score['format']:<12} counts={score['counts']} total={score['total_tokens']}{marker}")
    print("\nCOMPARISON")
    print("Strategy             Avg tokens   Reduction   Judge QA       Exact E2E   Avg prefix")
    for strategy, values in comparison.items():
        exact = "N/A" if values["exact_end_to_end_stability"] is None else f"{values['exact_end_to_end_stability']:.1%}"
        prefix = "N/A" if values["average_prefix_stability"] is None else f"{values['average_prefix_stability']:.1%}"
        print(f"{strategy:<20} {values['average_tokens']:>10.1f}   {values['reduction']:>8.1%}   {values['judge_correct']:>2}/15 ({values['judge_accuracy']:>5.1%})   {exact:>9}   {prefix:>10}")
    print("\nSTABILITY BY CONTEXT")
    for record in records:
        m = record["end_to_end_metrics"]
        print(f"Context {record['example_id']}: serializer=100/100 text+tokens; fact_exact={m['structured_fact_exact_rate']:.1%}; fact_jaccard={m['structured_fact_jaccard']:.1%}; final_exact={m['exact_token_rate']:.1%}; avg_lcp={m['average_pairwise_lcp_percentage']:.1%}; global_lcp={m['global_lcp_tokens']} ({m['global_lcp_percentage']:.1%}); counts={m['token_counts']}; mean={m['token_count_mean']:.2f}; sd={m['token_count_stddev']:.2f}")
        print(f"FIRST COMPILED REPRESENTATION:\n{record['runs'][0]['serialized']}\n")
    print("API USAGE")
    for scope in ("extraction", "answer", "judge"):
        uses = [usage for event, usage in usage_events if event == scope]
        print(f"{scope}={sum(u.get('input_tokens', 0) for u in uses)} in/{sum(u.get('output_tokens', 0) for u in uses)} out")


if __name__ == "__main__":
    main()
