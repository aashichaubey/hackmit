from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .config import load_cases
from .policy import evaluate_policy
from .scoring import critical_checks, load_scorer
from .storage import digest, file_hash, read_jsonl, strict_json, utc_now, write_json, write_jsonl


def _known_sum(values):
    return sum(values) if values and all(v is not None for v in values) else None


def write_report(directory: Path) -> dict:
    manifest = strict_json((directory / "manifest.json").read_text())
    config = manifest["configuration"]
    if file_hash(Path(config["fixture_path"])) != manifest["versions"]["dataset"]["sha256"]:
        raise ValueError("fixture hash changed since this run; refusing to regrade different expected answers")
    if file_hash(Path(config["scorer"]["path"])) != manifest["versions"]["scorer"]["sha256"]:
        raise ValueError("supplied scorer hash changed since this run")
    cases = load_cases(Path(config["fixture_path"]))
    selected = cases[:config["max_cases"]] if config["max_cases"] else cases
    case_map = {c["case_id"]: c for c in selected}
    scorer = load_scorer(Path(config["scorer"]["path"]))
    predictions = read_jsonl(directory / "predictions.jsonl")
    seen = set()
    for pred in predictions:
        key = pred.get("case_id"), pred.get("run_id")
        if key in seen:
            raise ValueError("duplicate case/run in predictions")
        seen.add(key)
        if key[0] not in case_map or type(key[1]) is not int or not 0 <= key[1] < config["repeats"]:
            raise ValueError("unknown case/run in predictions")
        if pred.get("configuration_fingerprint") != manifest["configuration_fingerprint"]:
            raise ValueError("mixed configurations in predictions")
    complete = [p for p in predictions if p.get("baseline_output") is not None and p.get("compressed_output") is not None]
    rows = []
    # The supplied scorer enforces equal repeat sets. Interrupted work is scored
    # per repeat as explicit partial diagnostics; completed runs use its full
    # validation path. No null output is passed to the scorer.
    expected_pairs = len(selected) * config["repeats"]
    complete_run = len(complete) == expected_pairs and manifest["status"] == "completed"
    if complete_run:
        rows = scorer.validate_and_pair(selected, complete, False)
    else:
        for repeat in range(config["repeats"]):
            group = [p for p in complete if p["run_id"] == repeat]
            if group:
                rows.extend(scorer.validate_and_pair(selected, group, True))
    categories, slices = defaultdict(list), defaultdict(list)
    failures, critical = [], {"baseline": Counter(), "compressed": Counter()}
    critical_coverage = {"baseline": Counter(), "compressed": Counter()}
    exposures = defaultdict(lambda: {"occurrences": 0, "case_ids": set(), "paired_trials": set(), "cooccurring_regressions": 0})
    for row in rows:
        case = case_map[row["case_id"]]
        categories[row["category"]].append(row)
        chars = sum(len(m["content"]) for m in case["messages"])
        changes = (row.get("compression_metadata") or {}).get("substitutions", [])
        density = sum(max(0, c["end"] - c["start"]) for c in changes) / chars if chars else 0
        length = "short" if chars < 1000 else "medium" if chars < 10000 else "long"
        density_bucket = "none" if not changes else "low" if density < .1 else "high"
        labels = [f"length_chars:{length}", f"alias_density_chars:{density_bucket}",
                  "source_group:" + case.get("source_group", case["source"])]
        labels.extend("tag:" + tag for tag in case.get("tags", []))
        if "position" in case:
            labels.append("position:" + case["position"])
        for label in labels:
            slices[label].append(row)
        row["critical_checks"] = {}
        for arm in ("baseline", "compressed"):
            checks = critical_checks(case, row[arm + "_output"], row.get(arm + "_error"), scorer)
            row["critical_checks"][arm] = checks
            critical[arm].update(c["type"] for c in checks if not c["passed"])
            critical_coverage[arm].update(c["type"] for c in checks)
        regression = bool(row["baseline_pass"] and not row["compressed_pass"])
        row["paired_outcome"] = ("both_correct" if row["baseline_pass"] and row["compressed_pass"] else
                                 "original_only_correct" if regression else
                                 "compressed_only_correct" if row["compressed_pass"] else "both_wrong")
        if row["paired_outcome"] != "both_correct":
            failures.append({**row, "failure_explanation": "Observed grader/assertion failure; causal mechanism unclassified."})
        for entry_id in {c["entry_id"] for c in changes}:
            exposure = exposures[entry_id]
            exposure["occurrences"] += sum(c["entry_id"] == entry_id for c in changes)
            exposure["case_ids"].add(row["case_id"])
            exposure["paired_trials"].add((row["case_id"], row["run_id"]))
            exposure["cooccurring_regressions"] += int(regression)

    events = read_jsonl(directory / "attempts.jsonl")
    from .runner import _attempts_from_events, _unique, _trial_key, build_predictions
    attempts = _attempts_from_events(events, manifest["configuration_fingerprint"])
    trials = _unique(read_jsonl(directory / "trials.jsonl"), _trial_key, "trial")
    timing_path = directory / "pipeline_timings.json"
    timings = strict_json(timing_path.read_text()) if timing_path.exists() else {}
    rebuilt = build_predictions(config, selected, trials, attempts, manifest["configuration_fingerprint"], timings)
    if digest(predictions) != digest(rebuilt):
        raise ValueError("predictions disagree with durable attempts/trials; refusing unreconciled measurements")
    completions = [e for e in events if e["event"] == "completed" and e["target_invoked"]]
    models = sorted({e["result"]["model"] for e in completions if e["result"].get("model")})
    providers = sorted({e["result"]["provider"] for e in completions if e["result"].get("provider")})
    target_consistent = bool(completions and len(models) == len(providers) == 1 and
                             all(e["result"].get("model") and e["result"].get("provider") for e in completions))
    first = {
        "scope": "all configured pairs" if complete_run else "PARTIAL complete-pair diagnostics; unequal coverage may bias results",
        "overall": scorer.metrics(rows) if rows else None,
        "by_category": {k: scorer.metrics(v) for k, v in sorted(categories.items())},
        "slices": {k: scorer.metrics(v) for k, v in sorted(slices.items())},
        "uncertainty": scorer.cluster_bootstrap(rows, config["bootstrap"]["resamples"], config["bootstrap"]["seed"]) if rows else None,
    }
    # Never present an observed subset as a complete-cohort savings claim.
    if not complete_run and first["overall"]:
        first["complete_pair_subset_token_diagnostic"] = first["overall"]["tokens"]
        first["overall"]["tokens"] = None
        if first["uncertainty"]:
            first["uncertainty"]["net_token_savings_95pct_interval"] = None
    pipeline = {"mode": config["pipeline"]["mode"], "fallback_trigger": "terminal errors only; never gold/grade-based"}
    for arm in ("baseline", "compressed"):
        observed = [p for p in predictions if p["pipeline"][arm]["final_output"] is not None]
        outcomes = [p["pipeline"][arm] for p in observed]
        good = sum(not o["final_error"] and scorer.grade(case_map[p["case_id"]], o["final_output"]) for p, o in zip(observed, outcomes))
        latencies = [o["latency_ms"] for o in outcomes]
        pipeline[arm] = {
            "observed_pairs": len(outcomes), "expected_pairs": expected_pairs,
            "final_accuracy": good / expected_pairs if complete_run else None,
            "observed_subset_accuracy": good / len(outcomes) if outcomes else None,
            "fallback_count": sum(o["fallback_used"] for o in outcomes),
            "fallback_frequency": sum(o["fallback_used"] for o in outcomes) / expected_pairs if complete_run else None,
            "retry_count": sum(o["retry_count"] for o in outcomes),
            "target_attempts": sum(o["attempts"] for o in outcomes),
            "total_input_tokens": _known_sum([o["total_input_tokens"] for o in outcomes]) if complete_run else None,
            "total_output_tokens": _known_sum([o["total_output_tokens"] for o in outcomes]) if complete_run else None,
            "total_cost_usd": _known_sum([o["cost_usd"] for o in outcomes]) if complete_run else None,
            "observed_target_cost_credits": _known_sum([o.get("target_cost_credits") for o in outcomes]) if complete_run else None,
            "latency_ms": {"p50": scorer.quantile(latencies, .5), "p95": scorer.quantile(latencies, .95)}
            if complete_run and all(v is not None for v in latencies) else None,
            "context_overflow_pairs": [p["case_id"] for p in observed if p["pipeline"][arm]["context_overflow"]],
        }
    compressor_observed = [p["compression_metadata"] for p in predictions if p.get("compression_metadata")]
    pipeline["compressor"] = {key: _known_sum([c.get(key) for c in compressor_observed]) if complete_run else None
                              for key in ("input_tokens", "output_tokens", "cost_usd", "latency_ms")}
    pipeline["compressed_success_without_fallback_count"] = sum(
        p["pipeline"]["compressed"]["final_output"] is not None and not p["pipeline"]["compressed"]["final_error"]
        and not p["pipeline"]["compressed"]["fallback_used"]
        and scorer.grade(case_map[p["case_id"]], p["pipeline"]["compressed"]["final_output"]) for p in predictions)
    report = {
        "schema_version": manifest["schema_version"], "experiment_id": manifest["experiment_id"], "generated_at": utc_now(),
        "mode": config["mode"], "synthetic": config["mode"] == "mock", "experiment": config["experiment"],
        "dataset": config["dataset"], "versions": manifest["versions"], "configuration": config,
        "fixture_splits": sorted({c["split"] for c in selected}), "independent_clusters": len({r["cluster_id"] for r in rows}),
        "coverage": {"dataset_cases": len(cases), "selected_cases": len(selected), "expected_pairs": expected_pairs,
                     "complete_pairs": len(rows), "cases_completed": sum(all(any(r["case_id"] == c["case_id"] and r["run_id"] == run_id for r in rows) for run_id in range(config["repeats"])) for c in selected),
                     "complete": complete_run, "full_dataset": len(selected) == len(cases)},
        "first_attempt": first, "pipeline": pipeline,
        "critical_failures": {a: dict(counts) for a, counts in critical.items()},
        "critical_check_coverage": {a: dict(counts) for a, counts in critical_coverage.items()},
        "critical_check_definitions": "Case-declared checks plus exact-value/format/boundary assertions by category; YES/NO opposite-answer checks; UNKNOWN requirement. These describe observed outputs, not a causal diagnosis.",
        "resolved_models": models, "resolved_providers": providers, "resolved_target_consistent": target_consistent,
        "dictionary_exposure": {eid: {**info, "case_ids": sorted(info["case_ids"]), "paired_trials": len(info["paired_trials"])} for eid, info in exposures.items()},
        "substitution_pairs": sum(bool((p.get("compression_metadata") or {}).get("substitutions")) for p in predictions),
        "controls": {"original_vs_compressed": "executed" if config["experiment"] == "original_vs_compressed" else "not executed in this run",
                     "original_vs_original": "executed" if config["experiment"] == "original_vs_original" else "available, not executed in this run",
                     "concise_english": "unavailable", "same_mapping_ascii": "unavailable", "other_compressor": "unavailable"},
        "notes": [*manifest["limitations"], "Alias density uses source-character coverage; it is not a token-savings estimate.",
                  "Length slices use characters (<1000, <10000, >=10000), not a guessed target token count.",
                  "Dictionary exposure/regression co-occurrence is not evidence of causation.",
                  "Context overflows are flagged failures; capacity-expansion experiments require a different protocol."]}
    if not report["substitution_pairs"]:
        report["notes"].append("No dictionary substitutions occurred; this run does not establish the quality of compressed representations.")
    report["gate"] = evaluate_policy(report, manifest["policy"])
    write_json(directory / "report.json", report)
    write_jsonl(directory / "scored_cases.jsonl", rows)
    write_jsonl(directory / "failures.jsonl", failures)
    overall = first["overall"]
    lines = ["# Character compression evaluation", "", "**SYNTHETIC MOCK — no model-quality or token-efficiency evidence.**" if report["synthetic"] else "**LIVE — actual provider responses; benchmark scope below.**", "",
             f"Gate: **{report['gate']['state']}**. Mode: `{config['experiment']}` / `{config['pipeline']['mode']}`.",
             f"Coverage: {len(rows)}/{expected_pairs} paired trials; {len(selected)}/{len(cases)} dataset cases; {report['independent_clusters']} source clusters.",
             f"Splits: {', '.join(report['fixture_splits'])}. Substitutions occurred in {report['substitution_pairs']} trials.", ""]
    if overall:
        lines += ["| First-attempt metric | Value |", "|---|---:|",
                  f"| Original accuracy | {overall['baseline_accuracy']:.2%} |", f"| Compressed accuracy | {overall['compressed_accuracy']:.2%} |",
                  f"| Accuracy change | {overall['accuracy_delta_pp']:.2f} pp |",
                  f"| Original-correct → compressed-wrong | {overall['baseline_correct_compressed_wrong']} |",
                  f"| Compressed-only correct | {overall['baseline_wrong_compressed_correct']} |",
                  f"| Both correct / both wrong | {overall['both_correct']} / {overall['both_wrong']} |",
                  f"| Regression rate among original-correct | {overall['regression_rate_given_baseline_correct']} |",
                  f"| Trustworthy token-measurement coverage | {overall['token_measurement_coverage']:.2%} |",
                  f"| Full-request savings | {format(overall['tokens']['net_input_token_savings'], '.2%') if overall['tokens'] else 'UNKNOWN / withheld'} |", ""]
        lines += ["| Category | Pairs | Original accuracy | Compressed accuracy | Regressions |", "|---|---:|---:|---:|---:|"]
        for name, metrics in first["by_category"].items():
            lines.append(f"| {name} | {metrics['pairs']} | {metrics['baseline_accuracy']:.2%} | {metrics['compressed_accuracy']:.2%} | {metrics['baseline_correct_compressed_wrong']} |")
        if first["uncertainty"]:
            lines += ["", "Exploratory paired source-cluster bootstrap: accuracy delta interval " + str(first["uncertainty"]["accuracy_delta_pp_95pct_interval"]) + " pp; token savings interval " + str(first["uncertainty"]["net_token_savings_95pct_interval"]) + ".",
                      first["uncertainty"]["caution"]]
    lines += ["", "Pipeline fallback counts: " + str({a: pipeline[a]["fallback_count"] for a in ("baseline", "compressed")}) + ". Fallback answers are not successful compression.",
              "", "Critical assertion failures: " + str(report["critical_failures"]) + ". Unclassified failures remain unclassified.",
              "", "Gate reasons: " + "; ".join(report["gate"]["reasons"]), "",
              "See report.json for settings, separate version identities, slices, consumption, latency, and all gates. See failures.jsonl for exact paired requests, expected outputs, actual answers, and substitutions.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
