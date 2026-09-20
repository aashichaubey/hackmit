"""Release gates require explicitly configured held-out evidence."""
from __future__ import annotations

import math


def validate_policy(policy: dict | None) -> None:
    if policy is None:
        return
    fields = {"schema_version", "enabled", "version", "min_net_savings", "min_compressed_accuracy",
              "max_accuracy_decline_pp", "max_regression_rate", "min_cases", "min_clusters", "critical_limits", "required_slices"}
    if not isinstance(policy, dict) or set(policy) - fields or type(policy.get("enabled")) is not bool or policy.get("schema_version") != 1:
        raise ValueError("invalid release policy schema or unknown policy fields")
    for name in ("min_net_savings", "min_compressed_accuracy", "max_accuracy_decline_pp", "max_regression_rate", "min_cases", "min_clusters"):
        value = policy.get(name)
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid policy {name}")
        if name in ("min_cases", "min_clusters") and (type(value) is not int or value < 1):
            raise ValueError("sample requirements must be positive integers")
        if name in ("min_net_savings", "min_compressed_accuracy", "max_regression_rate") and value > 1:
            raise ValueError(f"{name} must be a fraction from 0 to 1")
    limits = policy.get("critical_limits")
    if limits is not None:
        kinds = {"exact_value", "meaning_reversal", "output_contract", "unsupported_answer", "instruction_boundary", "unclassified"}
        if not isinstance(limits, dict) or set(limits) - kinds:
            raise ValueError("unknown critical-check limit")
        if any(v is not None and (type(v) is not int or v < 0) for v in limits.values()):
            raise ValueError("critical limits must be nonnegative integers or null")
    slices = policy.get("required_slices")
    if slices is not None:
        if not isinstance(slices, list):
            raise ValueError("required_slices must be a list")
        for item in slices:
            if not isinstance(item, dict) or set(item) - {"name", "min_pairs", "min_accuracy"} or not isinstance(item.get("name"), str):
                raise ValueError("invalid required slice")
            if type(item.get("min_pairs")) is not int or item["min_pairs"] < 1:
                raise ValueError("required slice needs positive min_pairs")
            if item.get("min_accuracy") is not None and (type(item["min_accuracy"]) not in (float, int) or not 0 <= item["min_accuracy"] <= 1):
                raise ValueError("slice min_accuracy must be a fraction")


def evaluate_policy(report: dict, policy: dict | None) -> dict:
    validate_policy(policy)
    if not policy or not policy.get("enabled", False):
        return {"state": "NOT_CONFIGURED", "reasons": ["No enabled acceptance policy."], "checks": []}
    missing = []
    coverage = report["coverage"]
    if report["mode"] != "live" or report.get("synthetic"):
        missing.append("Mock runs are synthetic harness tests.")
    if not report["dataset"].get("held_out") or not report["dataset"].get("split_before_dictionary"):
        missing.append("Dataset is not attested held-out with source splits preceding dictionary learning.")
    if any("development" in s or "smoke" in s for s in report["fixture_splits"]):
        missing.append("Development/smoke fixtures are not release evidence.")
    if not coverage["complete"] or not coverage["full_dataset"]:
        missing.append("Run does not cover every case and configured repeat in the dataset.")
    if report["experiment"] != "original_vs_compressed":
        missing.append("Baseline-versus-baseline controls do not certify compression.")
    if not report["resolved_target_consistent"]:
        missing.append("Resolved model/provider is missing or inconsistent across attempts.")
    overall = report.get("first_attempt", {}).get("overall")
    if not overall or overall["token_measurement_coverage"] != 1:
        missing.append("Trustworthy complete-request token measurements are incomplete.")
    required = ("min_net_savings", "min_compressed_accuracy", "max_accuracy_decline_pp", "max_regression_rate",
                "min_cases", "min_clusters", "critical_limits", "required_slices")
    for name in required:
        if policy.get(name) is None:
            missing.append(f"Application policy {name} is unset.")
    if policy.get("min_cases") is not None and coverage["cases_completed"] < policy["min_cases"]:
        missing.append("Insufficient independent cases.")
    if policy.get("min_clusters") is not None and report["independent_clusters"] < policy["min_clusters"]:
        missing.append("Insufficient independent source clusters.")
    for spec in policy.get("required_slices") or []:
        name = spec["name"]
        section = report["first_attempt"]["slices"].get(name)
        if not section or section["pairs"] < spec.get("min_pairs", 1):
            missing.append(f"Required slice {name} is missing or too small.")
        elif section["token_measurement_coverage"] != 1:
            missing.append(f"Required slice {name} lacks trustworthy counts.")
    for kind in policy.get("critical_limits") or {}:
        if kind != "unclassified" and not report.get("critical_check_coverage", {}).get("compressed", {}).get(kind):
            missing.append(f"No observed critical checks for {kind}.")
    if not report.get("substitution_pairs"):
        missing.append("No dictionary substitutions were exercised.")
    interval = (report.get("first_attempt", {}).get("uncertainty") or {}).get("accuracy_delta_pp_95pct_interval")
    if not interval or interval[0] is None:
        missing.append("Non-inferiority confidence bound is unavailable.")
    if missing:
        return {"state": "INCONCLUSIVE", "reasons": missing, "checks": []}
    checks = []

    def check(name, actual, comparison, limit):
        checks.append({"name": name, "actual": actual, "limit": limit,
                       "passed": actual is not None and (actual >= limit if comparison == "min" else actual <= limit)})

    check("net_input_savings", overall["tokens"]["net_input_token_savings"], "min", policy["min_net_savings"])
    check("compressed_accuracy", overall["compressed_accuracy"], "min", policy["min_compressed_accuracy"])
    check("accuracy_noninferiority_lower_bound_pp", interval[0], "min", -policy["max_accuracy_decline_pp"])
    if overall["regression_rate_given_baseline_correct"] is None:
        return {"state": "INCONCLUSIVE", "reasons": ["Regression denominator is zero."], "checks": checks}
    check("regression_rate", overall["regression_rate_given_baseline_correct"], "max", policy["max_regression_rate"])
    for kind, limit in policy["critical_limits"].items():
        if limit is None:
            return {"state": "INCONCLUSIVE", "reasons": [f"Critical limit {kind} is unset."], "checks": checks}
        check("critical:" + kind, report["critical_failures"]["compressed"].get(kind, 0), "max", limit)
    for spec in policy["required_slices"]:
        section = report["first_attempt"]["slices"][spec["name"]]
        if spec.get("min_accuracy") is not None:
            check("slice_accuracy:" + spec["name"], section["compressed_accuracy"], "min", spec["min_accuracy"])
    return {"state": "PASS" if all(x["passed"] for x in checks) else "FAIL", "reasons": [], "checks": checks}
