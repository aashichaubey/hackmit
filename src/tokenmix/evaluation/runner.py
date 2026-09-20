"""Sequential paired runner with an append-only request journal and resumption."""
from __future__ import annotations

import copy
import importlib.metadata
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from . import SCHEMA_VERSION
from .adapters import (CompressionResult, GenerationResult, Message, MockCompressor, MockTarget,
                       OpenRouterTarget, TokenmixCompressor, messages_from_json, redact_error)
from .config import load_cases
from .storage import append_jsonl, digest, file_hash, read_jsonl, strict_json, utc_now, write_json, write_jsonl


def source_hashes(config: dict) -> dict:
    package = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*.py"))}
    hashes["supplied_scorer"] = file_hash(Path(config["scorer"]["path"]))
    return hashes


def prepare_manifest(config: dict, cases: list[dict]) -> dict:
    dictionary = config["compression"].get("dictionary_path")
    dictionary_hash = file_hash(Path(dictionary)) if dictionary else None
    expected = config["compression"].get("expected_dictionary_sha256")
    if expected and expected != dictionary_hash:
        raise ValueError("stale dictionary: expected_dictionary_sha256 does not match")
    sources = source_hashes(config)
    versions = {"dataset": {"version": config["dataset"]["version"], "sha256": file_hash(Path(config["fixture_path"]))},
                "dictionary": {"version": config["compression"].get("dictionary_version"), "sha256": dictionary_hash},
                "compressor": {"version": config["compression"].get("version"), "source_sha256": sources.get("core.py")},
                "scorer": {"version": config["scorer"].get("version"), "sha256": sources["supplied_scorer"]},
                "model": {"requested": config["generation"]["model"], "resolved": None},
                "optimization_tokenizer": {"encoding": config["compression"].get("optimization_encoding"),
                                           "tiktoken_version": importlib.metadata.version("tiktoken")},
                "target_tokenizer": {"id": None, "accounting": "provider logical prompt usage; no local exact fallback"}}
    policy = strict_json(Path(config["policy_path"]).read_text()) if config["policy_path"] else None
    from .policy import validate_policy
    validate_policy(policy)
    fingerprint = digest({"config": config, "versions": versions, "sources": sources, "policy": policy})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"schema_version": SCHEMA_VERSION, "experiment_id": fingerprint[:24], "configuration_fingerprint": fingerprint,
            "configuration": config, "versions": versions, "source_hashes": sources,
            "implementation_revision": revision, "created_at": utc_now(), "updated_at": utc_now(),
            "mode": config["mode"], "status": "partial", "dataset_case_count": len(cases),
            "policy": policy, "limitations": [
                "Development fixtures and mock results cannot qualify for release.",
                "Tokenmix uses its configured optimization tokenizer; this is not DeepSeek token accounting.",
                "Only text conversations and deterministic exact/JSON graders are supported; tool execution and rubric judges are unsupported.",
                "A greedy dictionary replacement is not guaranteed to preserve meaning.",
                "No local exact DeepSeek chat-template counter: missing provider usage remains unknown.",
                "Client crashes after dispatch can leave outcome/usage unknown; such calls are not silently replayed.",
                "Provider seeds do not guarantee determinism; tokenizer and weight revisions may be unreported.",
            ]}


def _sum_known(values):
    return None if any(v is None for v in values) else sum(values)


def _trial_key(row):
    return row["case_id"], row["run_id"]


def _unique(rows, key, label):
    result = {}
    for row in rows:
        value = key(row)
        if value in result:
            raise ValueError(f"duplicate {label}: {value}")
        result[value] = row
    return result


def _attempts_from_events(events: list[dict], fingerprint: str) -> dict:
    starts, completed = {}, {}
    for event in events:
        if event.get("configuration_fingerprint") != fingerprint:
            raise ValueError("mixed configuration in attempt journal")
        if event.get("event") not in ("started", "completed"):
            raise ValueError("malformed attempt journal event")
        target = starts if event["event"] == "started" else completed
        aid = event["attempt_id"]
        if aid in target:
            raise ValueError(f"duplicate attempt event: {aid}")
        target[aid] = event
        if event["event"] == "completed":
            if aid not in starts or any(event.get(k) != starts[aid].get(k) for k in ("case_id", "run_id", "arm", "kind", "index")):
                raise ValueError("completed attempt lacks a matching start")
            result = event.get("result", {})
            if not isinstance(result.get("output"), str):
                raise ValueError("completed attempt must contain an actual output or empty-string terminal failure")
            if result.get("error") and result["output"] != "":
                raise ValueError("terminal failed calls require empty-string output")
            for key in ("input_tokens", "output_tokens", "cached_input_tokens", "latency_ms", "cost_usd", "cost_credits"):
                value = result.get(key)
                if value is not None and (type(value) not in (int, float) or value < 0):
                    raise ValueError(f"invalid attempt measurement: {key}")
                if key.endswith("tokens") and value is not None and type(value) is not int:
                    raise ValueError("token counts must be integers or unknown")
    return {aid: {"start": start, "completion": completed.get(aid)} for aid, start in starts.items()}


def run(config: dict, *, resume=False, compressor=None, target=None, progress=None) -> Path:
    cases = load_cases(Path(config["fixture_path"]))
    selected = cases[:config["max_cases"]] if config["max_cases"] else cases
    manifest = prepare_manifest(config, cases)
    output = Path(config["results_dir"]) / manifest["experiment_id"]
    fingerprint = manifest["configuration_fingerprint"]
    existing = output / "manifest.json"
    if existing.exists():
        if not resume:
            raise ValueError(f"run exists: {output}; use --resume, or choose a different results_dir")
        previous = strict_json(existing.read_text())
        if previous["configuration_fingerprint"] != fingerprint or previous["versions"] != manifest["versions"]:
            raise ValueError("resume configuration/fixture/version hashes do not match")
        manifest = previous
    elif resume:
        raise ValueError("no matching run to resume (configuration, fixtures, or implementation changed)")
    cache_path = output / "compression_cache.json"
    cache = strict_json(cache_path.read_text()) if cache_path.exists() else {}
    if compressor is None:
        compressor = (TokenmixCompressor(Path(config["compression"]["dictionary_path"]), cache)
                      if config["compression"]["adapter"] == "tokenmix" else MockCompressor())
    if target is None:
        target = (MockTarget() if config["mode"] == "mock" else
                  OpenRouterTarget(api_key_env=config["target"].get("api_key_env", "OPENROUTER_API_KEY"),
                                   timeout=config["timeout_seconds"]))
    output.mkdir(parents=True, exist_ok=True)
    # A single writer is mandatory; stale lock removal must be explicit after a crash.
    lock = output / ".runner.lock"
    try:
        lock_fd = lock.open("x")
    except FileExistsError as exc:
        raise ValueError(f"run is locked; if its process is gone, remove {lock} before resuming") from exc
    try:
        lock_fd.write(str(__import__("os").getpid()))
        lock_fd.close()
        write_json(existing, manifest)
        events_path = output / "attempts.jsonl"
        events = read_jsonl(events_path)
        attempts = _attempts_from_events(events, fingerprint)
        trials = _unique(read_jsonl(output / "trials.jsonl"), _trial_key, "trial")
        timings_path = output / "pipeline_timings.json"
        timings = strict_json(timings_path.read_text()) if timings_path.exists() else {}
        valid_keys = {(c["case_id"], r) for c in selected for r in range(config["repeats"])}
        for key, trial in trials.items():
            if key not in valid_keys or trial.get("configuration_fingerprint") != fingerprint:
                raise ValueError("unexpected case/repeat or configuration in trial journal")
        for data in attempts.values():
            start = data["start"]
            if _trial_key(start) not in valid_keys or start.get("arm") not in ("baseline", "compressed"):
                raise ValueError("unknown case/repeat/arm in attempt journal")
            if not data["completion"]:
                completion = {**start, "event": "completed", "completed_at": utc_now(),
                              "result": asdict(GenerationResult("", error="interrupted attempt: provider outcome and usage unknown"))}
                append_jsonl(events_path, completion)
                data["completion"] = completion
        calls = sum(a["start"]["target_invoked"] for a in attempts.values())

        def attempt(case_id, run_id, arm, index, kind, messages, error=None):
            nonlocal calls
            aid = digest([case_id, run_id, arm, index])[:24]
            if aid in attempts:
                return attempts[aid]["completion"]["result"]
            if error is None and config["mode"] == "live" and calls >= config["max_live_calls"]:
                return None
            started = {"event": "started", "attempt_id": aid, "configuration_fingerprint": fingerprint,
                       "case_id": case_id, "run_id": run_id, "arm": arm, "index": index, "kind": kind,
                       "started_at": utc_now(), "target_invoked": error is None,
                       "request": target.request(messages, config["generation"]) if error is None else None}
            append_jsonl(events_path, started)
            if error is None:
                calls += 1
                try:
                    result = target.generate(messages, copy.deepcopy(config["generation"]))
                except Exception as exc:
                    # Adapter failures are recorded, never changed into successes.
                    result = GenerationResult("", error=f"adapter exception: {redact_error(str(exc), getattr(target, 'secret', '') or '')}")
            else:
                result = GenerationResult("", error=error)
            completed = {**started, "event": "completed", "completed_at": utc_now(), "result": asdict(result)}
            append_jsonl(events_path, completed)
            attempts[aid] = {"start": started, "completion": completed}
            return completed["result"]

        def execute_arm(case_id, run_id, arm, trial):
            arm_started = time.perf_counter()
            timing_id = digest([case_id, run_id, arm])
            had_prior_attempts = any(a["start"]["case_id"] == case_id and a["start"]["run_id"] == run_id
                                     and a["start"]["arm"] == arm for a in attempts.values())
            base = messages_from_json(trial["baseline_messages"])
            compressed = messages_from_json(trial["compressed_messages"]) if trial["compressed_messages"] else None
            messages = base if arm == "baseline" else compressed
            error = trial["compression_error"] if arm == "compressed" else None
            result = attempt(case_id, run_id, arm, 0, "first", messages, error)
            if result is None:
                return False
            policy = config["pipeline"]
            for retry in range(1, policy["max_retries"] + 1):
                if not result["error"] or error:
                    break
                result = attempt(case_id, run_id, arm, retry, "retry", messages)
                if result is None:
                    return False
            if result["error"] and arm == "compressed" and policy["fallback_to_original"]:
                result = attempt(case_id, run_id, arm, policy["max_retries"] + 1, "fallback", base)
            if result is not None and timing_id not in timings:
                # Measure the active arm wall time, including retries/fallback
                # and runner overhead, plus its earlier compression time. The
                # randomized other-arm delay is deliberately excluded. Across
                # interruptions that latency cannot be recovered reliably.
                duration = None if had_prior_attempts else (time.perf_counter() - arm_started) * 1000
                if duration is not None and arm == "compressed":
                    duration += trial["compression"]["latency_ms"]
                timings[timing_id] = {"duration_ms": duration, "includes_compression": arm == "compressed"}
                write_json(timings_path, timings)
            return result is not None

        exhausted = False
        for run_id in range(config["repeats"]):
            for case in selected:
                key = case["case_id"], run_id
                if key not in trials:
                    messages = messages_from_json(case["messages"])
                    if config["boundary_instruction"]:
                        messages = (Message("system", config["boundary_instruction"]), *messages)
                    started = time.perf_counter()
                    error = None
                    try:
                        if config["experiment"] == "original_vs_original":
                            compressed = CompressionResult(messages, metadata={"control": "original_vs_original", "compression_bypassed": True})
                        else:
                            compressed = compressor.compress(messages, copy.deepcopy(config["compression"]))
                        compressed_messages = [asdict(m) for m in compressed.messages]
                        compression = asdict(compressed)
                        compression.pop("messages")
                    except Exception as exc:
                        error = f"compression error: {redact_error(str(exc), getattr(target, 'secret', '') or '')}"
                        compressed_messages = None
                        compression = {"substitutions": [], "latency_ms": (time.perf_counter() - started) * 1000,
                                       "input_tokens": None, "output_tokens": None, "cost_usd": None, "metadata": {}}
                    arm_order = ["baseline", "compressed"]
                    random.Random(digest([config["random_seed"], *key])).shuffle(arm_order)
                    trial = {"case_id": key[0], "run_id": run_id, "configuration_fingerprint": fingerprint,
                             "baseline_messages": [asdict(m) for m in messages], "compressed_messages": compressed_messages,
                             "compression": compression, "compression_error": error, "arm_order": arm_order}
                    write_json(cache_path, cache)
                    append_jsonl(output / "trials.jsonl", trial)
                    trials[key] = trial
                trial = trials[key]
                for arm in trial["arm_order"]:
                    if not execute_arm(*key, arm, trial):
                        exhausted = True
                        break
                if progress:
                    progress(f"{config['mode']}: {key[0]} repeat {run_id}; {calls} target calls")
                # Checkpoint after every pair, including bounded partial pairs.
                predictions = build_predictions(config, selected, trials, attempts, fingerprint, timings)
                write_jsonl(output / "predictions.jsonl", predictions)
                if exhausted:
                    break
            if exhausted:
                break
        predictions = build_predictions(config, selected, trials, attempts, fingerprint, timings)
        write_jsonl(output / "predictions.jsonl", predictions)
        manifest.update(status="partial" if exhausted else "completed", updated_at=utc_now(), target_calls=calls)
        write_json(existing, manifest)
        from .reporting import write_report
        write_report(output)
    finally:
        lock.unlink(missing_ok=True)
    return output


def build_predictions(config, selected, trials, attempts, fingerprint, timings=None):
    timings = timings or {}
    predictions = []
    for case in selected:
        for run_id in range(config["repeats"]):
            trial = trials.get((case["case_id"], run_id))
            row = {"case_id": case["case_id"], "run_id": run_id, "configuration_fingerprint": fingerprint,
                   "model_id": config["generation"]["model"], "tokenizer_id": None,
                   "dictionary_version": config["compression"].get("dictionary_version"),
                   "compression_version": config["compression"].get("version"),
                   "expected_output": case["expected_output"], "generation_config": config["generation"],
                   "original_messages": trial["baseline_messages"] if trial else case["messages"],
                   "compressed_messages": trial["compressed_messages"] if trial else None,
                   "compression_metadata": trial["compression"] if trial else None,
                   "token_count_source": {}, "pipeline": {}}
            for arm in ("baseline", "compressed"):
                selected_attempts = sorted([a for a in attempts.values() if a["start"]["case_id"] == case["case_id"]
                                           and a["start"]["run_id"] == run_id and a["start"]["arm"] == arm],
                                          key=lambda a: a["start"]["index"])
                first = selected_attempts[0]["completion"]["result"] if selected_attempts else None
                for name in ("output", "error", "input_tokens", "output_tokens"):
                    value = first.get(name) if first else None
                    if name == "input_tokens" and first and (not first["token_count_trustworthy"] or not value):
                        value = None
                    row[f"{arm}_{name}"] = value
                row["token_count_source"][arm] = first.get("token_count_source") if first else None
                latency = first.get("latency_ms") if first else None
                cost = first.get("cost_usd") if first else None
                compression = trial["compression"] if trial and arm == "compressed" else {"latency_ms": 0, "cost_usd": 0, "input_tokens": 0, "output_tokens": 0}
                row["baseline_latency_ms" if arm == "baseline" else "compressed_pipeline_latency_ms"] = _sum_known([latency, compression["latency_ms"]])
                row["baseline_cost_usd" if arm == "baseline" else "compressed_pipeline_cost_usd"] = _sum_known([cost, compression["cost_usd"]])
                results = [a["completion"]["result"] for a in selected_attempts]
                row["pipeline"][arm] = {
                    "final_output": results[-1]["output"] if results else None,
                    "final_error": results[-1]["error"] if results else None,
                    "fallback_used": any(a["start"]["kind"] == "fallback" for a in selected_attempts),
                    "attempts": len(results), "retry_count": sum(a["start"]["kind"] == "retry" for a in selected_attempts),
                    "total_input_tokens": _sum_known([r["input_tokens"] if r["token_count_trustworthy"] else None for r in results] + [compression["input_tokens"]]) if results else None,
                    "total_output_tokens": _sum_known([r["output_tokens"] for r in results] + [compression["output_tokens"]]) if results else None,
                    "cost_usd": _sum_known([r["cost_usd"] for r in results] + [compression["cost_usd"]]) if results else None,
                    "target_cost_credits": _sum_known([r.get("cost_credits") for r in results]) if results else None,
                    "latency_ms": timings.get(digest([case["case_id"], run_id, arm]), {}).get("duration_ms"),
                    "stage_latency_sum_ms": _sum_known([r["latency_ms"] for r in results] + [compression["latency_ms"]]) if results else None,
                    "context_overflow": any(r["context_overflow"] for r in results),
                    "resolved_models": sorted({r["model"] for r in results if r["model"]}),
                    "resolved_providers": sorted({r["provider"] for r in results if r["provider"]}),
                }
            predictions.append(row)
    return predictions
