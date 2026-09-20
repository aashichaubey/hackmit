"""Release-policy gate.

States:
  NOT_CONFIGURED - no policy supplied; nothing was evaluated.
  INCONCLUSIVE   - a policy is enabled but the evidence is inadequate
                   (mock run, partial coverage, approximate token counts,
                   too few cases/clusters, missing required slices).
  FAIL           - adequate evidence, and at least one threshold was missed.
  PASS           - adequate evidence, and every enabled threshold was met.

Deliberate design choices:

* Quality and savings are never collapsed into one weighted score. Each check
  is reported separately with its own observed value and threshold.
* A mock/development run can never reach PASS.
* Accuracy non-inferiority compares the *lower bound* of the bootstrap
  interval for the accuracy delta against the permitted decline, not the point
  estimate.
* Token gates require exact counts. `approximate_local_template` counts are
  diagnostic only and force INCONCLUSIVE rather than silently qualifying.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NOT_CONFIGURED = "NOT_CONFIGURED"
INCONCLUSIVE = "INCONCLUSIVE"
FAIL = "FAIL"
PASS = "PASS"

EXACT_TOKEN_SOURCES = {"provider_usage", "exact_local_tokenizer"}


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | INCONCLUSIVE | NOT_CONFIGURED
    observed: Any
    threshold: Any
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "status": self.status,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class PolicyResult:
    status: str
    checks: list[Check] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_status": self.status,
            "checks": [c.as_dict() for c in self.checks],
            "reasons": self.reasons,
            "evidence": self.evidence,
            "note": (
                "Quality and savings are reported as independent checks. "
                "A PASS requires a real, complete run with exact token measurements; "
                "development/mock runs are never release-qualified."
            ),
        }


def _token_source_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, int] = {}
    for p in predictions:
        src = p.get("token_count_source") or "unknown"
        sources[src] = sources.get(src, 0) + 1
    total = len(predictions) or 1
    exact = sum(n for s, n in sources.items() if s in EXACT_TOKEN_SOURCES)
    return {
        "sources": sources,
        "exact_fraction": exact / total,
        "all_exact": exact == total,
    }


def evaluate(
    report: dict[str, Any],
    manifest: dict[str, Any],
    predictions: list[dict[str, Any]],
    policy: dict[str, Any] | None,
) -> PolicyResult:
    """Apply `policy` to a scored run. Returns a PolicyResult, never raises."""
    if not policy or not policy.get("enabled", False):
        return PolicyResult(
            status=NOT_CONFIGURED,
            reasons=[
                "No release policy is enabled. The illustrative thresholds in "
                "release_policy.example.json are a template, not an approved default."
            ],
            evidence={"policy_supplied": bool(policy)},
        )

    overall = report.get("overall", {})
    tokens = overall.get("tokens")
    uncertainty = report.get("uncertainty", {})
    tok = _token_source_summary(predictions)

    checks: list[Check] = []
    blockers: list[str] = []

    # ---- evidence adequacy (gates the whole result) ----------------------
    if manifest.get("synthetic_results") or manifest.get("adapters", {}).get("compressor_is_mock") \
            or manifest.get("adapters", {}).get("target_is_mock"):
        blockers.append(
            "Run used mock adapters (synthetic_results=true); mock results are never release-qualified."
        )

    coverage = report.get("case_coverage", 0.0)
    min_coverage = policy.get("min_case_coverage", 1.0)
    if coverage < min_coverage:
        blockers.append(f"case coverage {coverage:.1%} < required {min_coverage:.1%}")

    pairs = overall.get("pairs", 0)
    if (min_cases := policy.get("min_cases")) is not None and pairs < min_cases:
        blockers.append(f"only {pairs} paired trials < required {min_cases}")

    clusters = uncertainty.get("clusters", 0)
    if (min_clusters := policy.get("min_source_clusters")) is not None and clusters < min_clusters:
        blockers.append(f"only {clusters} independent source clusters < required {min_clusters}")

    if policy.get("require_heldout_split", False):
        splits = set(report.get("fixture_splits", []))
        if splits & {"development_smoke"} or not splits:
            blockers.append(
                f"fixture splits {sorted(splits)} are development/smoke; a release gate "
                "requires a held-out split."
            )

    required_slices = policy.get("required_slices") or []
    present = set(report.get("by_category", {}))
    if missing := [s for s in required_slices if s not in present]:
        blockers.append(f"required slices missing from report: {missing}")

    token_cov = overall.get("token_measurement_coverage", 0.0)
    if token_cov < policy.get("min_token_measurement_coverage", 1.0):
        blockers.append(f"token measurement coverage {token_cov:.1%} below requirement")

    savings_gate_enabled = policy.get("min_net_input_savings") is not None
    if savings_gate_enabled and not tok["all_exact"]:
        blockers.append(
            "net-savings gate requires exact token counts, but "
            f"{100*(1-tok['exact_fraction']):.0f}% of records use approximate/unknown "
            f"sources {sorted(tok['sources'])}. Approximate counts are diagnostic only."
        )

    # ---- threshold checks (computed regardless, reported honestly) --------
    if (thr := policy.get("min_compressed_accuracy")) is not None:
        obs = overall.get("compressed_accuracy")
        checks.append(Check(
            "min_compressed_accuracy",
            PASS if obs is not None and obs >= thr else FAIL,
            obs, thr,
        ))

    if (thr := policy.get("max_accuracy_decline_pp")) is not None:
        obs = overall.get("accuracy_delta_pp")
        interval = uncertainty.get("accuracy_delta_pp_95pct_interval") or [None, None]
        lower = interval[0]
        if lower is None:
            checks.append(Check(
                "accuracy_non_inferiority", INCONCLUSIVE, obs, -abs(thr),
                "no bootstrap interval available for the accuracy delta",
            ))
            blockers.append("accuracy non-inferiority could not be evaluated")
        else:
            # Non-inferiority: the worst plausible decline must stay within thr.
            ok = lower >= -abs(thr)
            checks.append(Check(
                "accuracy_non_inferiority",
                PASS if ok else FAIL,
                {"delta_pp": obs, "lower_95": lower},
                -abs(thr),
                "compares the lower 95% bound of the accuracy delta, not the point estimate",
            ))

    if (thr := policy.get("max_regression_rate")) is not None:
        obs = overall.get("regression_rate_given_baseline_correct")
        if obs is None:
            checks.append(Check(
                "max_regression_rate", INCONCLUSIVE, None, thr,
                "undefined: no baseline-correct pairs",
            ))
            blockers.append("regression rate undefined (no baseline-correct pairs)")
        else:
            checks.append(Check("max_regression_rate", PASS if obs <= thr else FAIL, obs, thr))

    if savings_gate_enabled:
        thr = policy["min_net_input_savings"]
        obs = tokens.get("net_input_token_savings") if tokens else None
        if obs is None:
            checks.append(Check("min_net_input_savings", INCONCLUSIVE, None, thr,
                                "token totals incomplete"))
        else:
            checks.append(Check("min_net_input_savings", PASS if obs >= thr else FAIL, obs, thr))

    if (thr := policy.get("max_expansion_rate")) is not None:
        obs = tokens.get("expansion_rate") if tokens else None
        checks.append(Check(
            "max_expansion_rate",
            INCONCLUSIVE if obs is None else (PASS if obs <= thr else FAIL),
            obs, thr,
        ))

    for category, limit in (policy.get("max_category_regressions") or {}).items():
        cat = report.get("by_category", {}).get(category)
        obs = cat.get("baseline_correct_compressed_wrong") if cat else None
        checks.append(Check(
            f"critical_category_regressions[{category}]",
            INCONCLUSIVE if obs is None else (PASS if obs <= limit else FAIL),
            obs, limit,
            "" if cat else "category absent from this run",
        ))

    evidence = {
        "pairs": pairs,
        "case_coverage": coverage,
        "source_clusters": clusters,
        "token_measurement_coverage": token_cov,
        "token_count_sources": tok["sources"],
        "exact_token_fraction": tok["exact_fraction"],
        "synthetic_results": bool(manifest.get("synthetic_results")),
        "fixture_splits": report.get("fixture_splits", []),
        "experiment_mode": manifest.get("experiment_mode"),
    }

    if blockers:
        return PolicyResult(INCONCLUSIVE, checks, blockers, evidence)
    if any(c.status == FAIL for c in checks):
        return PolicyResult(
            FAIL, checks,
            [f"{c.name}: observed {c.observed} vs threshold {c.threshold}"
             for c in checks if c.status == FAIL],
            evidence,
        )
    if any(c.status == INCONCLUSIVE for c in checks):
        return PolicyResult(
            INCONCLUSIVE, checks,
            [f"{c.name} could not be evaluated" for c in checks if c.status == INCONCLUSIVE],
            evidence,
        )
    return PolicyResult(PASS, checks, [], evidence)


def load_policy(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))
