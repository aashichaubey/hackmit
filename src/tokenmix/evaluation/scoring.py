"""Reuse the supplied scorer unchanged, with coverage and provenance checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_scorer(path: Path):
    if not path.is_file():
        raise ValueError(f"supplied scorer is missing: {path}")
    spec = importlib.util.spec_from_file_location("tokenmix_supplied_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("grade", "metrics", "cluster_bootstrap", "validate_and_pair"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"supplied scorer lacks {name}")
    return module


def critical_checks(case: dict, output: str, error: str | None, scorer) -> list[dict]:
    """Check explicit output assertions, never infer a compressor failure cause."""
    if error:
        return [{"type": "unclassified", "assertion": "terminal_call_error", "passed": False,
                 "evidence": error}]
    checks = list(case.get("critical_checks", []))
    if case["category"] in ("literal_fidelity", "format", "boundaries_unicode"):
        kind = {"literal_fidelity": "exact_value", "format": "output_contract",
                "boundaries_unicode": "instruction_boundary"}[case["category"]]
        checks.append({"type": kind, "assertion": "equals_expected"})
    if case["expected_output"] in ("YES", "NO"):
        checks.append({"type": "meaning_reversal", "assertion": "not_in",
                       "forbidden_outputs": ["NO" if case["expected_output"] == "YES" else "YES"]})
    if case["expected_output"] == "UNKNOWN":
        checks.append({"type": "unsupported_answer", "assertion": "equals_expected"})
    results = []
    normalized = output.strip() if case["grader"].get("strip_outer_whitespace", False) else output
    for check in checks:
        if check["assertion"] == "equals_expected":
            passed = scorer.grade(case, output)
        elif check["assertion"] == "not_in":
            passed = normalized not in check["forbidden_outputs"]
        else:
            try:
                scorer.strict_json(output)
                passed = True
            except (ValueError, RecursionError):
                passed = False
        results.append({**check, "passed": bool(passed), "evidence": None if passed else output})
    if not scorer.grade(case, output) and not any(not c["passed"] for c in results):
        results.append({"type": "unclassified", "assertion": "independent_grader", "passed": False,
                        "evidence": output})
    return results
