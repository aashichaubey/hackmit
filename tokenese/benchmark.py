import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from openai import OpenAI
from dotenv import load_dotenv
from . import MODEL, PROMPT_VERSION
from .baseline import DeletionBaseline
from .bear import BearBaseline
from .cache import CachedClient
from .encode import profiles, render_english, render_tokenese, tokenese_legend
from .facts import Answer, MeetingFacts
from .judges import grade_exact, normalize
from .model import answer_question, extract_facts
from .prompts import answer_prompt
from .search import screen_profiles
from .tokens import count_tokens

ROOT = Path(__file__).resolve().parent.parent

def load_cases():
    return json.loads((ROOT / "data" / "meetings.json").read_text())["meetings"]

def choose_encoding(facts, question, model=MODEL, qualified_profiles=()):
    english = render_english(facts)
    best = (count_tokens(answer_prompt(english, question), model), english, "english")
    for name in qualified_profiles:
        context = render_tokenese(facts, name)
        size = count_tokens(answer_prompt(context, question, tokenese_legend(name)), model)
        if size < best[0]:
            best = (size, context, name)
    return best[1], best[2]

def method_input(facts, notes, question, method):
    if method == "raw":
        context, legend = notes, ""
    elif method == "english":
        context, legend = render_english(facts), ""
    else:
        context, legend = render_tokenese(facts, method), tokenese_legend(method)
    return context, answer_prompt(context, question, legend)

def matches_gold_fact(found, gold):
    if found["kind"] != gold["kind"] or found["deadline"] != gold["deadline"]:
        return False
    if gold["person"] and found["person"] != gold["person"]:
        return False
    left = normalize(found["text"])
    right = normalize(gold["text"])
    if left in right or right in left:
        return True
    left_quote = normalize(found.get("source_quote", ""))
    right_quote = normalize(gold.get("source_quote", ""))
    return bool(left_quote and right_quote and (left_quote in right_quote or right_quote in left_quote))

def run_case(client, case, model=MODEL, fact_source="gold", methods=("raw", "english", "symbols", "mixed"), deletion=None):
    result = {"id": case["id"], "split": case["split"], "fact_source": fact_source, "model": model, "prompt_version": PROMPT_VERSION, "notes": case["notes"], "questions": [], "extraction": None, "errors": []}
    facts = None
    try:
        if fact_source == "parsed":
            facts, usage = extract_facts(client, case["notes"], model)
            result["extraction"] = {"facts": facts.model_dump(), "usage": usage.model_dump()}
        else:
            facts = MeetingFacts(facts=case["facts"])
            result["extraction"] = {"facts": facts.model_dump(), "usage": {"input_tokens": 0, "output_tokens": 0}}
    except Exception as exc:
        result["errors"].append(f"extraction: {exc}")
    for question in case["questions"]:
        row = {"question": question["question"], "answers": question["answers"], "kind": question["kind"], "methods": {}}
        for method in methods:
            context = None
            prompt = None
            try:
                if method in ("deletion", "bear-2"):
                    if deletion is None or method not in deletion:
                        raise ValueError(f"{method} baseline unavailable")
                    target = count_tokens(answer_prompt(case["notes"], question["question"]), model)
                    if facts is not None:
                        target = count_tokens(answer_prompt(render_tokenese(facts, "symbols"), question["question"], tokenese_legend("symbols")), model)
                    detail = deletion[method].compress(case["notes"], question["question"], target, model)
                    context, prompt = detail["context"], answer_prompt(detail["context"], question["question"])
                else:
                    if method != "raw" and facts is None:
                        raise ValueError("Fact extraction failed")
                    context, prompt = method_input(facts, case["notes"], question["question"], method)
                answer, usage = answer_question(client, prompt, model)
                row["methods"][method] = {"context": context, "prompt": prompt, "context_tokens": count_tokens(context, model), "full_input_tokens": count_tokens(prompt, model), "answer": answer.model_dump(), "usage": usage.model_dump(), "correct": grade_exact(answer, question["answers"]), "error": None}
                if method in profiles():
                    row["methods"][method]["candidate_version"] = profiles()[method]["version"]
                if method in ("deletion", "bear-2"):
                    row["methods"][method].update({key: value for key, value in detail.items() if key not in ("context", "seconds")})
                    row["methods"][method]["compression_seconds"] = detail["seconds"]
            except Exception as exc:
                row["methods"][method] = {"error": str(exc), "correct": False, "context": context, "prompt": prompt}
                if context is not None:
                    row["methods"][method]["context_tokens"] = count_tokens(context, model)
                if prompt is not None:
                    row["methods"][method]["full_input_tokens"] = count_tokens(prompt, model)
        result["questions"].append(row)
    return result

def summarize(results, cases=None):
    methods = {}
    extraction_misses = 0
    extraction_additions = 0
    gold_cases = {case["id"]: case for case in (cases if cases is not None else load_cases())}
    for result in results:
        if result.get("fact_source") == "parsed":
            gold = gold_cases.get(result["id"])
            if gold:
                found = (result.get("extraction") or {}).get("facts", {}).get("facts", [])
                extraction_misses += sum(not any(matches_gold_fact(f, g) for f in found) for g in gold["facts"])
                extraction_additions += sum(not any(matches_gold_fact(f, g) for g in gold["facts"]) for f in found)
        for row in result["questions"]:
            for name, record in row["methods"].items():
                bucket = methods.setdefault(name, {"correct": 0, "questions": 0, "errors": 0, "owner_errors": 0, "deadline_errors": 0, "prompt_tokens": [], "api_input_tokens": 0, "api_output_tokens": 0, "compression_seconds": 0.0, "bear_input_tokens": 0, "bear_output_tokens": 0, "bear_trial_input_tokens": 0, "bear_trial_output_tokens": 0})
                bucket["questions"] += 1
                bucket["correct"] += bool(record.get("correct"))
                bucket["errors"] += bool(record.get("error"))
                if row["kind"] == "owner" and not record.get("correct"):
                    bucket["owner_errors"] += 1
                if row["kind"] == "deadline" and not record.get("correct"):
                    bucket["deadline_errors"] += 1
                if "full_input_tokens" in record:
                    bucket["prompt_tokens"].append(record["full_input_tokens"])
                usage = record.get("usage", {})
                bucket["api_input_tokens"] += usage.get("input_tokens", 0)
                bucket["api_output_tokens"] += usage.get("output_tokens", 0)
                bucket["compression_seconds"] += record.get("compression_seconds", 0)
                bucket["bear_input_tokens"] += record.get("bear_input_tokens", 0)
                bucket["bear_output_tokens"] += record.get("bear_output_tokens", 0)
                bucket["bear_trial_input_tokens"] += record.get("bear_trial_input_tokens", 0)
                bucket["bear_trial_output_tokens"] += record.get("bear_trial_output_tokens", 0)
    for bucket in methods.values():
        bucket["accuracy"] = bucket["correct"] / bucket["questions"] if bucket["questions"] else 0
        bucket["median_prompt_tokens"] = median(bucket["prompt_tokens"]) if bucket["prompt_tokens"] else None
        del bucket["prompt_tokens"]
    extraction_usage = {"input_tokens": sum((r.get("extraction") or {}).get("usage", {}).get("input_tokens", 0) for r in results), "output_tokens": sum((r.get("extraction") or {}).get("usage", {}).get("output_tokens", 0) for r in results)}
    overhead = extraction_usage["input_tokens"] + extraction_usage["output_tokens"]
    raw = methods.get("raw")
    for name, bucket in methods.items():
        answer_tokens = bucket["api_input_tokens"] + bucket["api_output_tokens"]
        bucket["workflow_total_api_tokens"] = answer_tokens + (overhead if name not in ("raw", "deletion", "bear-2") else 0)
        if raw and name not in ("raw", "deletion", "bear-2") and bucket["questions"]:
            raw_per_question = (raw["api_input_tokens"] + raw["api_output_tokens"]) / raw["questions"]
            this_per_question = answer_tokens / bucket["questions"]
            saved = raw_per_question - this_per_question
            bucket["breakeven_questions_per_meeting"] = None if saved <= 0 else math.ceil((overhead / max(len(results), 1)) / saved)
    return {"methods": methods, "extraction_usage": extraction_usage, "extraction_misses": extraction_misses, "extraction_additions": extraction_additions, "meetings": len(results)}

def qualify_profiles(validation_runs):
    accepted = []
    for name in profiles():
        candidate_correct = 0
        english_correct = 0
        candidate_tokens = 0
        english_tokens = 0
        critical_regressions = 0
        candidate_failures = 0
        for result in validation_runs:
            for row in result["questions"]:
                english = row["methods"].get("english", {})
                candidate = row["methods"].get(name, {})
                candidate_failures += bool(candidate.get("error")) or "full_input_tokens" not in candidate
                english_correct += bool(english.get("correct"))
                candidate_correct += bool(candidate.get("correct"))
                english_tokens += english.get("full_input_tokens", 0)
                candidate_tokens += candidate.get("full_input_tokens", 0)
                if row["kind"] in ("owner", "deadline", "decision", "agreement", "disagreement", "unknown") and english.get("correct") and not candidate.get("correct"):
                    critical_regressions += 1
        if candidate_failures == 0 and critical_regressions == 0 and candidate_correct >= english_correct - 1 and candidate_tokens < english_tokens:
            accepted.append((candidate_tokens, name))
    return [name for _, name in sorted(accepted)]

def local_token_metrics(cases):
    output = {}
    for split in ("dev", "validation", "test"):
        by_method = {name: {"context_tokens": [], "full_input_tokens": []} for name in ("raw", "english", "symbols", "mixed")}
        for case in cases:
            if case["split"] != split:
                continue
            facts = MeetingFacts(facts=case["facts"])
            for question in case["questions"]:
                for name, values in by_method.items():
                    context, prompt = method_input(facts, case["notes"], question["question"], name)
                    values["context_tokens"].append(count_tokens(context, MODEL))
                    values["full_input_tokens"].append(count_tokens(prompt, MODEL))
        output[split] = {name: {"questions": len(values["full_input_tokens"]), "mean_context_tokens": round(sum(values["context_tokens"]) / len(values["context_tokens"]), 2), "mean_full_input_tokens": round(sum(values["full_input_tokens"]) / len(values["full_input_tokens"]), 2), "total_full_input_tokens": sum(values["full_input_tokens"])} for name, values in by_method.items()}
    return output

def local_report():
    cases = load_cases()
    return {"status": "local screening only; no model answers measured", "model": MODEL, "dataset_version": "1", "case_counts": {split: sum(c["split"] == split for c in cases) for split in ("dev", "validation", "test")}, "profile_screen": screen_profiles([c for c in cases if c["split"] != "test"], MODEL), "local_token_metrics": local_token_metrics(cases), "runs": [], "summary": {}, "qualified_profiles": [], "structured_output_schemas": {"extraction": MeetingFacts.model_json_schema(), "answer": Answer.model_json_schema()}, "research_usage": {"grader_calls": 0, "embedding_calls": 0, "candidate_search_calls": 0}, "jev_pilot": {"status": "not run: no key" if not os.getenv("TYPESAFE_API_KEY") else "available; not run"}}

def write_search_log(report):
    path = ROOT / "reports" / "search.jsonl"
    rows = []
    for item in report["profile_screen"]:
        rows.append({"dataset_version": report["dataset_version"], "model": MODEL, "prompt_version": PROMPT_VERSION, "candidate": item["profile"], "candidate_version": item["version"], "split": "dev+validation", "mean_full_input_token_savings": item["mean_token_savings"], "hard_failures": item["hard_failures"], "cheaper_questions": item["cheaper_questions"], "total_questions": item["total_questions"], "failure_labels": ["full_prompt_cost"] if item["cheaper_questions"] == 0 else []})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

def success_gate(report):
    profile = next(iter(report["qualified_profiles"]), None)
    if profile is None:
        return {"passed": False, "reason": "No profile qualified on validation"}
    test = report["summary"]["parsed"]["test"]["methods"]
    raw = test["raw"]
    encoded = test[profile]
    rows = [row for run in report["runs"] if run["fact_source"] == "parsed" and run["split"] == "test" for row in run["questions"]]
    cheaper = sum(row["methods"][profile].get("full_input_tokens", float("inf")) < row["methods"]["english"].get("full_input_tokens", 0) for row in rows)
    critical = sum(row["kind"] in ("owner", "deadline", "decision", "agreement", "disagreement", "unknown") and row["methods"]["english"].get("correct") and not row["methods"][profile].get("correct") for row in rows)
    passed = encoded["accuracy"] >= 0.95 * raw["accuracy"] and critical == 0 and cheaper > len(rows) / 2
    return {"passed": passed, "profile": profile, "raw_accuracy": raw["accuracy"], "encoded_accuracy": encoded["accuracy"], "cheaper_questions": cheaper, "test_questions": len(rows), "critical_regressions": critical, "reason": "test quality and token gate"}

def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--include-deletion", action="store_true")
    parser.add_argument("--include-bear", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (args.include_deletion or args.include_bear or args.smoke) and not args.run_api:
        parser.error("Baseline and smoke options require --run-api")
    report = local_report()
    write_search_log(report)
    if args.run_api:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required for API benchmark")
        if args.include_bear and not os.getenv("TOKEN_COMPANY_API_KEY"):
            raise SystemExit("TOKEN_COMPANY_API_KEY is required for Bear-2 benchmark")
        client = CachedClient(OpenAI(), ROOT / "reports" / "cache.json")
        deletion = {}
        if args.include_deletion:
            deletion["deletion"] = DeletionBaseline()
        if args.include_bear:
            deletion["bear-2"] = BearBaseline()
        methods = ("raw", "english", "symbols", "mixed") + (("deletion",) if args.include_deletion else ()) + (("bear-2",) if args.include_bear else ())
        runs = []
        cases = load_cases()
        if args.smoke:
            cases = [{**case, "questions": case["questions"][:3]} for case in cases[:10]]
        for split in ("dev", "validation"):
            for source in ("gold", "parsed"):
                for case in cases:
                    if case["split"] == split:
                        runs.append(run_case(client, case, fact_source=source, methods=methods, deletion=deletion))
        report["qualified_profiles"] = qualify_profiles([run for run in runs if run["split"] == "validation" and run["fact_source"] == "gold"])
        for source in ("gold", "parsed"):
            for case in cases:
                if case["split"] == "test":
                    runs.append(run_case(client, case, fact_source=source, methods=methods, deletion=deletion))
        report["runs"] = runs
        report["summary"] = {source: {split: summarize([r for r in runs if r["fact_source"] == source and r["split"] == split]) for split in ("dev", "validation", "test")} for source in ("gold", "parsed")}
        if not args.smoke:
            report["success_gate"] = success_gate(report)
        report["status"] = "API smoke run complete" if args.smoke else "API benchmark complete"
    output_path = ROOT / "reports" / ("smoke.json" if args.smoke else "latest.json")
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))

if __name__ == "__main__":
    main()
