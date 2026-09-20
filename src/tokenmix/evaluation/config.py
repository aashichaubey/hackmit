from __future__ import annotations

import copy
import os
import re
from pathlib import Path

from .adapters import messages_from_json
from .storage import read_jsonl, strict_json


def _keys(value, allowed, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if unknown := set(value) - set(allowed):
        raise ValueError(f"unknown {context} fields: {sorted(unknown)}")


def load_config(path: Path) -> dict:
    config = strict_json(path.read_text(encoding="utf-8"))
    return resolve_config(config, path.parent)


def resolve_config(value: dict, base: Path) -> dict:
    config = copy.deepcopy(value)
    _keys(config, {"schema_version", "mode", "fixture_path", "results_dir", "dataset", "target", "generation",
                  "compression", "scorer", "experiment", "repeats", "random_seed", "timeout_seconds",
                  "concurrency", "pipeline", "max_live_calls", "max_cases", "bootstrap", "policy_path",
                  "boundary_instruction"}, "config")
    if config.get("schema_version") != 1 or config.get("mode") not in ("mock", "live"):
        raise ValueError("schema_version=1 and mode=mock|live are required")
    for name in ("fixture_path", "results_dir"):
        config[name] = str((base / config[name]).resolve())
    config.setdefault("policy_path", None)
    if config["policy_path"]:
        config["policy_path"] = str((base / config["policy_path"]).resolve())
    _keys(config["dataset"], {"version", "held_out", "split_before_dictionary", "provenance"}, "dataset")
    for flag in ("held_out", "split_before_dictionary"):
        if type(config["dataset"].get(flag, False)) is not bool:
            raise ValueError(f"dataset.{flag} must be boolean")
    if not config["dataset"].get("version"):
        raise ValueError("dataset.version is required")
    _keys(config["target"], {"adapter", "api_key_env", "version"}, "target")
    _keys(config["scorer"], {"adapter", "path", "version"}, "scorer")
    if config["scorer"].get("adapter") != "starter":
        raise ValueError("the supplied starter scorer is required; no substitute grader is selected")
    config["scorer"]["path"] = str((base / config["scorer"]["path"]).resolve())
    if config["target"].get("adapter") not in ("mock", "openrouter"):
        raise ValueError("target.adapter must be mock or openrouter")
    if (config["mode"] == "mock") != (config["target"]["adapter"] == "mock"):
        raise ValueError("mode and target adapter disagree; mock/live substitution is forbidden")
    generation = config["generation"]
    _keys(generation, {"model", "temperature", "top_p", "max_tokens", "seed", "reasoning", "provider",
                       "response_format", "tools", "tool_choice", "stop"}, "generation")
    if not isinstance(generation.get("model"), str) or not generation["model"]:
        raise ValueError("generation.model is required")
    if type(generation.get("max_tokens")) is not int or generation["max_tokens"] <= 0:
        raise ValueError("positive generation.max_tokens is required")
    if config["mode"] == "live":
        provider = generation.get("provider", {})
        if provider.get("allow_fallbacks") is not False or provider.get("require_parameters") is not True:
            raise ValueError("live runs require provider.allow_fallbacks=false and require_parameters=true")
        if len(provider.get("only", [])) != 1:
            raise ValueError("pin exactly one OpenRouter provider in generation.provider.only for matched arms")
    comp = config["compression"]
    _keys(comp, {"adapter", "version", "dictionary_path", "dictionary_version", "expected_dictionary_sha256",
                 "optimization_encoding", "domain", "protect", "max_changes", "requested_level", "scope",
                 "delimiters", "decoding"}, "compression")
    if comp.get("adapter") not in ("tokenmix", "mock"):
        raise ValueError("compression.adapter must be tokenmix or mock")
    if config["mode"] == "live" and comp["adapter"] == "mock":
        raise ValueError("mock compressor is forbidden in live mode")
    if comp["adapter"] == "tokenmix":
        comp["dictionary_path"] = str((base / comp["dictionary_path"]).resolve())
        if not comp.get("optimization_encoding"):
            raise ValueError("explicit compression.optimization_encoding is required")
    comp.setdefault("domain", "general")
    comp.setdefault("protect", [])
    comp.setdefault("max_changes", 100)
    comp.setdefault("requested_level", None)
    comp.setdefault("scope", "user_messages")
    comp.setdefault("delimiters", ["<source>", "</source>"])
    if comp["scope"] not in ("user_messages", "delimited_passages"):
        raise ValueError("scope must be user_messages or delimited_passages")
    if (len(comp["delimiters"]) != 2 or any(not isinstance(x, str) or not x for x in comp["delimiters"])
            or comp["delimiters"][0] == comp["delimiters"][1]):
        raise ValueError("two distinct nonempty passage delimiters are required")
    if type(comp["max_changes"]) is not int or comp["max_changes"] < 0:
        raise ValueError("max_changes must be nonnegative")
    if not isinstance(comp["protect"], list) or any(not isinstance(x, str) or not x for x in comp["protect"]):
        raise ValueError("protect must contain nonempty strings")
    decoding = comp.setdefault("decoding", {"strategy": "natural_bilingual", "dictionary": "none", "instruction": "", "examples": []})
    _keys(decoding, {"strategy", "dictionary", "instruction", "examples", "version"}, "decoding")
    if decoding.get("strategy") not in ("natural_bilingual", "dictionary") or decoding.get("dictionary") not in ("none", "used_entries"):
        raise ValueError("unsupported decoding strategy or dictionary attachment")
    if not isinstance(decoding.get("instruction"), str) or not isinstance(decoding.get("examples"), list):
        raise ValueError("decoding instruction/examples must be string/list")
    config.setdefault("experiment", "original_vs_compressed")
    if config["experiment"] not in ("original_vs_compressed", "original_vs_original"):
        raise ValueError("unsupported control; unavailable variants must not be marked benchmarked")
    config.setdefault("repeats", 1)
    config.setdefault("random_seed", 17)
    config.setdefault("timeout_seconds", 60)
    config.setdefault("concurrency", 1)
    config.setdefault("max_live_calls", 0)
    config.setdefault("max_cases", None)
    config.setdefault("boundary_instruction", "")
    for name in ("repeats", "random_seed", "max_live_calls"):
        if type(config[name]) is not int or config[name] < (1 if name == "repeats" else 0):
            raise ValueError(f"invalid {name}")
    if config["mode"] == "live" and config["max_live_calls"] < 1:
        raise ValueError("live runs require a positive max_live_calls limit")
    if config["concurrency"] != 1:
        raise ValueError("this runner supports concurrency=1 for durable call accounting")
    if not isinstance(config["timeout_seconds"], (float, int)) or config["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be positive")
    if config["max_cases"] is not None and (type(config["max_cases"]) is not int or config["max_cases"] < 1):
        raise ValueError("max_cases must be null or positive")
    pipeline = config.setdefault("pipeline", {"mode": "first_attempt", "max_retries": 0, "fallback_to_original": False})
    _keys(pipeline, {"mode", "max_retries", "fallback_to_original"}, "pipeline")
    if pipeline.get("mode") not in ("first_attempt", "deployment"):
        raise ValueError("unsupported pipeline mode")
    if type(pipeline.get("max_retries")) is not int or pipeline["max_retries"] < 0 or type(pipeline.get("fallback_to_original")) is not bool:
        raise ValueError("invalid retry/fallback policy")
    if pipeline["mode"] == "first_attempt" and (pipeline["max_retries"] or pipeline["fallback_to_original"]):
        raise ValueError("first_attempt mode forbids retries and fallback")
    config.setdefault("bootstrap", {"resamples": 2000, "seed": 29})
    _keys(config["bootstrap"], {"resamples", "seed"}, "bootstrap")
    if any(type(config["bootstrap"].get(x)) is not int or config["bootstrap"][x] < (1 if x == "resamples" else 0) for x in ("resamples", "seed")):
        raise ValueError("invalid bootstrap settings")
    return config


def load_cases(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"fixture file is missing: {path}; supply the original starter kit, do not substitute fabricated fixtures")
    cases = read_jsonl(path)
    if not cases:
        raise ValueError("empty fixture dataset")
    seen = set()
    for case in cases:
        for name in ("case_id", "category", "cluster_id", "split", "source"):
            if not isinstance(case.get(name), str) or not case[name]:
                raise ValueError(f"case requires nonempty {name}")
        if case["case_id"] in seen:
            raise ValueError(f"duplicate case_id: {case['case_id']}")
        seen.add(case["case_id"])
        messages_from_json(case["messages"])
        grader = case.get("grader", {})
        if grader.get("type") not in ("exact", "json"):
            raise ValueError(f"unsupported grader: {grader.get('type')}; rubrics/LLM judges are not implemented")
        if "expected_output" not in case:
            raise ValueError("independent expected_output is required")
        if grader["type"] == "exact" and not isinstance(case["expected_output"], str):
            raise ValueError("exact grader requires a string expected_output")
        for check in case.get("critical_checks", []):
            if check.get("type") not in ("exact_value", "meaning_reversal", "output_contract", "unsupported_answer", "instruction_boundary"):
                raise ValueError("unsupported critical check type")
            if check.get("assertion") not in ("equals_expected", "not_in", "strict_json"):
                raise ValueError("unsupported critical assertion")
            if check["assertion"] == "not_in" and not isinstance(check.get("forbidden_outputs"), list):
                raise ValueError("not_in assertion requires forbidden_outputs")
    return cases


def load_dotenv(path: Path) -> None:
    """Read key=value only. Never source shell code or log credentials."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("invalid .env line; expected NAME=value")
        name, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z_][A-Z_0-9]*", name):
            raise ValueError("invalid .env variable name")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)
