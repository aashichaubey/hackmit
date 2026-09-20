"""One-way validation selection, frozen test, baselines and full-workflow accounting."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .facts import MeetingFacts
from .evo_data import DATA, REPORTS, load_cases
from .evo_eval import evaluate, summarize, paired_regressions, classifier_metrics, context_for, experiment_identity, evaluate_batch
from .evo_grammar import GrammarSpec, compile_facts, integrity_errors, digest
from .evo_memory import ClassifiedMeeting, classifier_prompt, product_identity, code_hashes, verify_frozen, freeze_digest
from .evo_search import save
from .evo_usage import Runner, Ledger, dollars, BudgetExceeded
from .prompts import answer_prompt
from .tokens import count_tokens


def gate(candidate: list[dict], baselines: list[list[dict]], cases: list[dict], integrity=0, require_savings=True) -> dict:
    result = summarize(candidate)
    paired = paired_regressions(candidate, baselines)
    expected_ids = {q["id"] for c in cases for q in c["questions"]}
    complete = len(cases) == 12 and len(expected_ids) == 120 and len(candidate) == 120 and {r["question_id"] for r in candidate} == expected_ids
    reviewed = all(c["provenance"]["annotation_status"] == "independently_reviewed" for c in cases)
    usages = [summarize(b)["actual_input_tokens"] for b in baselines]
    savings = result["actual_input_tokens"] is not None and all(t is not None and result["actual_input_tokens"] < t for t in usages)
    passed = result["correct"] >= 119 and not paired["new_critical_errors"] and integrity == 0 and (savings or not require_savings)
    status = "incomplete_evaluation" if not complete or not reviewed or not result["usage_complete"] or any(t is None for t in usages) else "qualified" if passed else "no_qualified_candidate"
    return {"status": status, "metric_gate_pass": passed and complete, "independent_review_complete": reviewed,
            "full_corpus": complete, "integrity_failures": integrity, "lower_actual_input_than_baselines": savings,
            **paired, "summary": result}


def classify_cases(cases: list[dict], grammar: GrammarSpec, runner) -> dict:
    def one(case):
        result = runner.call(classifier_prompt(case["notes"]), ClassifiedMeeting, purpose="extraction", max_output_tokens=4096)
        facts = None
        reason = "extraction_failed"
        diagnostic = None
        if not result.get("error") and result.get("answer"):
            try:
                classified = ClassifiedMeeting.model_validate(result["answer"])
                candidate = MeetingFacts(facts=classified.facts)
                if classified.facts and all(f.source_quote in case["notes"] for f in classified.facts):
                    facts = candidate
                    compiled = compile_facts(facts, grammar)
                    errors = integrity_errors(facts, compiled, grammar)
                    reason = "integrity" if errors else "unsupported_or_ambiguous" if classified.unsupported_content or classified.ambiguity_notes else "not_cheaper" if count_tokens(answer_prompt(compiled.text, "")) >= count_tokens(answer_prompt(case["notes"], "")) else "encoded"
                else:
                    reason = "empty_or_unmatched_source"
                diagnostic = classifier_metrics(MeetingFacts(facts=case["facts"]), candidate)
            except ValueError:
                reason = "invalid_extraction"
        return case["id"], {"result": result, "facts": facts.model_dump() if facts is not None else None,
                             "route": reason, "metrics": diagnostic}
    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(one, cases))


def workflow_comparison(cases: list[dict], raw: list[dict], encoded: list[dict], extraction: dict) -> dict:
    outputs = {}
    for n in (1, 4, 10):
        raw_input = product_input = 0
        raw_usd = product_usd = 0.0
        complete = True
        quality = {"raw_correct": 0, "product_correct": 0, "questions": 0}
        for case in cases:
            ids = [q["id"] for q in case["questions"][:n]]
            left = {r["question_id"]: r for r in raw}
            right = {r["question_id"]: r for r in encoded}
            ext = extraction[case["id"]]["result"].get("usage")
            if ext is None or len(ids) < n:
                complete = False
                continue
            product_input += ext["input_tokens"]
            product_usd += dollars(ext)
            for qid in ids:
                a, b = left.get(qid), right.get(qid)
                if a is None or b is None or a.get("usage") is None or b.get("usage") is None:
                    complete = False
                    continue
                raw_input += a["usage"]["input_tokens"]
                product_input += b["usage"]["input_tokens"]
                raw_usd += dollars(a["usage"])
                product_usd += dollars(b["usage"])
                quality["raw_correct"] += bool(a["correct"])
                quality["product_correct"] += bool(b["correct"])
                quality["questions"] += 1
        outputs[str(n)] = {"accounting_status": "complete" if complete else "incomplete",
            "raw_input_tokens": raw_input if complete else None, "product_input_tokens": product_input if complete else None,
            "raw_usd": raw_usd if complete else None, "product_usd": product_usd if complete else None,
            "input_savings_fraction": 1-product_input/raw_input if complete and raw_input else None,
            "cost_savings_fraction": 1-product_usd/raw_usd if complete and raw_usd else None, **quality}
    return outputs


def benchmark(grammar: GrammarSpec, cases: list[dict], runner, seed: GrammarSpec | None = None, gold_rows=None) -> dict:
    methods = {"evolved": gold_rows if gold_rows is not None else evaluate(grammar, cases, runner, purpose="qualification")}
    for method in ("raw", "prose", "english"):
        methods[method] = evaluate(None, cases, runner, method=method, purpose="baseline")
    if seed is not None:
        methods["initial_seed"] = evaluate(seed, cases, runner, method="initial_seed", purpose="baseline")
    # Product usefulness is measured against original source questions, not only
    # field-oriented questions manufactured from the classifier's supported schema.
    product_cases = [{**c, 'questions':c.get('holistic_questions',c['questions'])} for c in cases]
    product_raw = evaluate(None, product_cases, runner, method='raw', purpose='product_qualification')
    extraction = classify_cases(cases, grammar, runner)
    parsed = {cid: MeetingFacts.model_validate(x["facts"]) for cid, x in extraction.items() if x["facts"] is not None}
    overrides = {c["id"]: c["notes"] for c in cases if extraction[c["id"]]["route"] != "encoded"}
    parsed_rows = {}
    for method in ("evolved", "prose", "english"):
        parsed_rows[method] = evaluate(grammar, product_cases, runner, method=method, purpose="product_qualification", parsed_facts=parsed,
                                        context_overrides=overrides if method == "evolved" else None)
    language = gate(methods["evolved"], [methods["english"], methods["prose"]], cases)
    product = gate(parsed_rows["evolved"], [product_raw, parsed_rows["english"]], product_cases, require_savings=False)
    workflows = workflow_comparison(product_cases, product_raw, parsed_rows["evolved"], extraction)
    parsed_rows['raw'] = product_raw
    saving_workloads = [n for n, w in workflows.items() if w["cost_savings_fraction"] is not None and w["cost_savings_fraction"] > 0 and w["product_correct"] >= w["raw_correct"]]
    if product["status"] == "qualified" and not saving_workloads:
        product["status"] = "no_qualified_candidate"
    return {"gold": methods, "parsed": parsed_rows, "extraction": extraction,
        "language_gate": language, "product_gate": product, "workflow": workflows, "product_saving_workloads": saving_workloads,
        "summaries": {"gold": {k: summarize(v) for k, v in methods.items()}, "parsed": {k: summarize(v) for k, v in parsed_rows.items()}}}


def selection(runner, directory: Path = REPORTS, data_dir: Path = DATA) -> dict:
    frozen_path = directory / "frozen.json"
    if frozen_path.exists():
        raise ValueError("A grammar is already frozen; validation cannot choose another")
    search = json.loads((directory / "search.json").read_text())
    if search["status"] != "development_complete":
        raise ValueError("Complete development evaluation first")
    options = [f for f in search["finalists"] if f["archive_pass"]]
    options.sort(key=lambda f: (-f["summary"]["correct"], f["summary"]["visible_prompt_tokens"], f["grammar_id"]))
    unresolved_archive = not bool(options)
    if not options:
        # Preserve an honest failure result and permit a diagnostic frozen test.
        # This is never a way around the development archive release gate.
        options = sorted(search['finalists'],key=lambda f:(-f['summary']['correct'],f['summary']['visible_prompt_tokens']))[:1]
        if not options:
            raise ValueError('No measured finalist available even for diagnostics')
    options = options[:2]
    contract = {"identity": experiment_identity(), "product_identity": product_identity(), "code_hashes": code_hashes(),
                "dataset_manifest": json.loads((data_dir / "manifest.json").read_text()),
                "candidate_ids": [o["grammar_id"] for o in options], "search_hash": digest(search)}
    lock = directory / "selection-lock.json"
    if lock.exists() and json.loads(lock.read_text()) != contract:
        raise ValueError("Selection contract changed")
    save(lock, contract)
    cases = load_cases(data_dir / "validation.json", "validation")
    baselines = {m: evaluate(None, cases, runner, method=m, purpose="baseline") for m in ("english", "prose")}
    evaluated = []
    for option in options:
        grammar = GrammarSpec(**option["grammar"])
        rows = evaluate(grammar, cases, runner, purpose="validation")
        evaluated.append({"grammar": grammar.to_dict(), "grammar_id": grammar.id, "rows": rows,
                          "gate": gate(rows, list(baselines.values()), cases)})
    passing = [e for e in evaluated if e["gate"]["metric_gate_pass"] and not unresolved_archive]
    # A non-passing grammar can be frozen only for a prominently diagnostic test.
    winner = min(passing, key=lambda e: e["gate"]["summary"]["actual_input_tokens"]) if passing else min(evaluated, key=lambda e: (-e["gate"]["summary"]["correct"], e["gate"]["summary"]["visible_prompt_tokens"]))
    grammar = GrammarSpec(**winner["grammar"])
    seed = GrammarSpec()
    validation = benchmark(grammar, cases, runner, seed, winner["rows"])
    save(directory / "validation.json", {"candidate_comparison": evaluated, **validation})
    frozen = {**contract, "grammar": grammar.to_dict(), "grammar_id": grammar.id,
              "initial_seed": seed.to_dict(), "language_status": "incomplete_evaluation", "product_status": "incomplete_evaluation",
              "validation_language_status": validation["language_gate"]["status"],
              "validation_product_status": validation["product_gate"]["status"],
              "diagnostic_only": not bool(passing), "accounting_status": runner.ledger.summary()["accounting_status"]}
    frozen["freeze_hash"] = freeze_digest(frozen)
    save(frozen_path, frozen)
    return frozen


def test_frozen(path: Path, runner, data_dir: Path = DATA) -> dict:
    directory = path.parent
    frozen = json.loads(path.read_text())
    grammar = verify_frozen(frozen, require_qualified=False)
    if frozen["dataset_manifest"] != json.loads((data_dir / "manifest.json").read_text()):
        raise ValueError("Frozen corpus hashes changed")
    if (directory / "test.json").exists():
        raise ValueError("Test already opened and completed; no adaptive reruns")
    lock = directory / "test-lock.json"
    if lock.exists() and json.loads(lock.read_text())["freeze_hash"] != frozen["freeze_hash"]:
        raise ValueError("Test freeze changed")
    save(lock, {"freeze_hash": frozen["freeze_hash"], "status": "opened"})
    cases = load_cases(data_dir / "test.json", "test")
    result = benchmark(grammar, cases, runner, GrammarSpec(**frozen["initial_seed"]))
    for scope in ("language", "product"):
        validation_status = frozen[f"validation_{scope}_status"]
        test_status = result[f"{scope}_gate"]["status"]
        frozen[f"{scope}_status"] = "qualified" if validation_status == test_status == "qualified" else "incomplete_evaluation" if "incomplete_evaluation" in (validation_status, test_status) else "no_qualified_candidate"
    frozen["accounting_status"] = runner.ledger.summary()["accounting_status"]
    if frozen["accounting_status"] != "complete":
        frozen["language_status"] = frozen["product_status"] = "incomplete_evaluation"
    result["diagnostic_only"] = frozen["diagnostic_only"] or frozen["validation_product_status"] != "qualified"
    result["ledger"] = runner.ledger.summary()
    result["freeze_hash"] = frozen["freeze_hash"]
    save(directory / "test.json", result)
    save(path, frozen)
    return result


def development_ablations(runner, directory=REPORTS, data_dir=DATA) -> dict:
    if (directory / "selection-lock.json").exists():
        raise ValueError("Ablations must finish before opening validation")
    search = json.loads((directory / "search.json").read_text())
    cases = load_cases(data_dir / "dev.json", "dev")
    eligible = [f for f in search["finalists"] if f["archive_pass"]]
    if not eligible:
        eligible = search["finalists"]
    best = min(eligible, key=lambda f: (-f["summary"]["correct"], f["summary"]["visible_prompt_tokens"]))
    grammar = GrammarSpec(**best["grammar"])
    sample = cases[:4]
    baseline = benchmark(grammar, sample, runner, GrammarSpec())
    batches = {}
    for method in ("raw", "english", "evolved"):
        batches[method] = [evaluate_batch(context_for(c, grammar, method), c["questions"][:4], runner) for c in sample]
    result = {"scope": "development only, first four public excerpts", "same_fact_and_extraction": baseline,
              "four_question_batches": batches, "repair_trace": search["repairs"],
              "matched_search_comparison": "Run search-no-repair with identical seed, generations, candidates and compare per-round shared probes; actual and logical call counts reported separately."}
    no_repair_path = directory / "search-no-repair.json"
    if no_repair_path.exists():
        other = json.loads(no_repair_path.read_text())
        matched_rounds = min(len(search['generations']),len(other['generations']))
        paired_rounds = []
        for index in range(matched_rounds):
            a,b = search['generations'][index],other['generations'][index]
            paired_rounds.append({'generation':index,'identical_probe':a['probe_ids']==b['probe_ids'],
                                 'repair_best_correct':max(r['summary']['correct'] for r in a['candidates']),
                                 'no_repair_best_correct':max(r['summary']['correct'] for r in b['candidates']),
                                 'probe_questions':len(a['probe_ids'])})
        result["matched_search_comparison"] = {"scope":'First two rounds only; later adaptive development is excluded from the matched-budget comparison',
            "repair_rounds": matched_rounds, "no_repair_rounds": matched_rounds, 'paired_rounds':paired_rounds,
            "repair_probe_evaluations": sum(len(r["rows"]) for g in search["generations"][:matched_rounds] for r in g["candidates"]),
            "no_repair_probe_evaluations": sum(len(r["rows"]) for g in other["generations"][:matched_rounds] for r in g["candidates"]),
            "repair_finalists": [{k:f[k] for k in ("grammar_id", "summary", "archive_pass")} for f in search["finalists"]],
            "no_repair_finalists": [{k:f[k] for k in ("grammar_id", "summary", "archive_pass")} for f in other["finalists"]]}
    save(directory / "ablations.json", result)
    return result


def reproduce_qualification(directory=REPORTS, data_dir=DATA) -> dict:
    frozen = json.loads((directory/'frozen.json').read_text())
    verify_frozen(frozen,require_qualified=False)
    if frozen['dataset_manifest'] != json.loads((data_dir/'manifest.json').read_text()):
        raise ValueError('Dataset manifest changed')
    result = {'freeze_hash':frozen['freeze_hash'], 'loaded_splits':['validation','test'], 'checks':{}}
    for split in ('validation','test'):
        report = json.loads((directory/f'{split}.json').read_text())
        cases = load_cases(data_dir/f'{split}.json',split)
        language = gate(report['gold']['evolved'], [report['gold']['english'],report['gold']['prose']],cases)
        product_cases = [{**c,'questions':c.get('holistic_questions',c['questions'])} for c in cases]
        product = gate(report['parsed']['evolved'],[report['parsed']['raw'],report['parsed']['english']],product_cases,require_savings=False)
        workflows = workflow_comparison(product_cases,report['parsed']['raw'],report['parsed']['evolved'],report['extraction'])
        result['checks'][split] = {'language_gate_matches':language == report['language_gate'],
                                  'product_quality_matches':product['summary'] == report['product_gate']['summary'],
                                  'workflow_arithmetic_matches':workflows == report['workflow']}
    result['passed'] = all(all(c.values()) for c in result['checks'].values())
    save(directory/'verification.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--select-validation", action="store_true")
    parser.add_argument("--frozen", type=Path, default=REPORTS / "frozen.json")
    parser.add_argument("--split", choices=("test",))
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument('--audit',action='store_true',help='Reproduce saved gate and workflow arithmetic without API calls')
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(reproduce_qualification(),indent=2))
        return
    if not args.run_api:
        parser.error('Paid evaluation requires --run-api; use --audit for saved-result verification')
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    runner = Runner(OpenAI(), Ledger(REPORTS / "ledger.sqlite"))
    try:
        result = selection(runner) if args.select_validation else development_ablations(runner) if args.ablations else test_frozen(args.frozen, runner) if args.split == "test" else None
    except BudgetExceeded as exc:
        save(REPORTS / "incomplete.json", {"status": "incomplete_evaluation", "reason": str(exc), "ledger": runner.ledger.summary()})
        raise
    print(json.dumps({"finished": result is not None, "ledger": runner.ledger.summary()}, indent=2))


if __name__ == "__main__":
    main()
