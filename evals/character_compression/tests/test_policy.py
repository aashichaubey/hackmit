"""Release-gate tests: all four states, and the rules that make PASS hard."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

from policy import FAIL, INCONCLUSIVE, NOT_CONFIGURED, PASS, evaluate  # noqa: E402


def make_report(**kw):
    base = {
        "fixture_count": 40,
        "case_coverage": 1.0,
        "fixture_splits": ["held_out_release"],
        "overall": {
            "pairs": 120,
            "baseline_accuracy": 0.90,
            "compressed_accuracy": 0.90,
            "accuracy_delta_pp": 0.0,
            "baseline_correct_compressed_wrong": 2,
            "regression_rate_given_baseline_correct": 0.02,
            "token_measurement_coverage": 1.0,
            "tokens": {"net_input_token_savings": 0.30, "expansion_rate": 0.0},
        },
        "by_category": {"negation": {"baseline_correct_compressed_wrong": 0, "pairs": 12}},
        "uncertainty": {
            "clusters": 31,
            "accuracy_delta_pp_95pct_interval": [-0.5, 0.5],
            "net_token_savings_95pct_interval": [0.25, 0.35],
        },
    }
    base["overall"].update(kw.pop("overall", {}))
    base.update(kw)
    return base


def make_manifest(synthetic=False, **kw):
    m = {
        "synthetic_results": synthetic,
        "adapters": {"compressor_is_mock": synthetic, "target_is_mock": synthetic},
        "experiment_mode": "first_attempt",
    }
    m.update(kw)
    return m


def preds(n=120, source="provider_usage"):
    return [{"case_id": f"c{i}", "token_count_source": source} for i in range(n)]


ENABLED = {
    "enabled": True,
    "min_net_input_savings": 0.20,
    "max_accuracy_decline_pp": 1.0,
    "min_case_coverage": 1.0,
    "min_token_measurement_coverage": 1.0,
    "require_heldout_split": True,
}


class GateStateTest(unittest.TestCase):
    def test_not_configured_without_policy(self):
        self.assertEqual(evaluate(make_report(), make_manifest(), preds(), None).status,
                         NOT_CONFIGURED)

    def test_not_configured_when_policy_disabled(self):
        policy = dict(ENABLED, enabled=False)
        self.assertEqual(evaluate(make_report(), make_manifest(), preds(), policy).status,
                         NOT_CONFIGURED)

    def test_example_policy_file_is_disabled(self):
        p = json.loads((KIT / "release_policy.example.json").read_text(encoding="utf-8"))
        self.assertFalse(p["enabled"], "illustrative policy must ship disabled")

    def test_pass_on_good_complete_real_run(self):
        r = evaluate(make_report(), make_manifest(), preds(), ENABLED)
        self.assertEqual(r.status, PASS, r.reasons)

    def test_fail_when_savings_below_threshold(self):
        rep = make_report(overall={"tokens": {"net_input_token_savings": 0.05,
                                              "expansion_rate": 0.1}})
        r = evaluate(rep, make_manifest(), preds(), ENABLED)
        self.assertEqual(r.status, FAIL)
        self.assertTrue(any("min_net_input_savings" in x for x in r.reasons))


class EvidenceAdequacyTest(unittest.TestCase):
    """A good-looking result must not PASS on inadequate evidence."""

    def test_mock_run_can_never_pass(self):
        r = evaluate(make_report(), make_manifest(synthetic=True), preds(), ENABLED)
        self.assertEqual(r.status, INCONCLUSIVE)
        self.assertTrue(any("mock" in x.lower() for x in r.reasons))

    def test_approximate_tokens_block_the_savings_gate(self):
        r = evaluate(make_report(), make_manifest(),
                     preds(source="approximate_local_template"), ENABLED)
        self.assertEqual(r.status, INCONCLUSIVE)
        self.assertTrue(any("exact token counts" in x for x in r.reasons))

    def test_development_split_blocks_release(self):
        rep = make_report(fixture_splits=["development_smoke"])
        r = evaluate(rep, make_manifest(), preds(), ENABLED)
        self.assertEqual(r.status, INCONCLUSIVE)
        self.assertTrue(any("development/smoke" in x for x in r.reasons))

    def test_partial_coverage_blocks_release(self):
        rep = make_report(case_coverage=0.5)
        r = evaluate(rep, make_manifest(), preds(), ENABLED)
        self.assertEqual(r.status, INCONCLUSIVE)

    def test_too_few_clusters_blocks_release(self):
        policy = dict(ENABLED, min_source_clusters=50)
        r = evaluate(make_report(), make_manifest(), preds(), policy)
        self.assertEqual(r.status, INCONCLUSIVE)
        self.assertTrue(any("cluster" in x for x in r.reasons))

    def test_missing_required_slice_blocks_release(self):
        policy = dict(ENABLED, required_slices=["negation", "long_context"])
        r = evaluate(make_report(), make_manifest(), preds(), policy)
        self.assertEqual(r.status, INCONCLUSIVE)
        self.assertTrue(any("slices missing" in x for x in r.reasons))


class NonInferiorityTest(unittest.TestCase):
    def test_uses_lower_bound_not_point_estimate(self):
        """Delta is 0pp (looks fine) but the interval admits a 5pp decline."""
        rep = make_report()
        rep["uncertainty"]["accuracy_delta_pp_95pct_interval"] = [-5.0, 5.0]
        r = evaluate(rep, make_manifest(), preds(), ENABLED)
        self.assertEqual(r.status, FAIL)
        check = next(c for c in r.checks if c.name == "accuracy_non_inferiority")
        self.assertEqual(check.status, FAIL)
        self.assertEqual(check.observed["delta_pp"], 0.0)

    def test_inconclusive_without_interval(self):
        rep = make_report()
        rep["uncertainty"]["accuracy_delta_pp_95pct_interval"] = None
        r = evaluate(rep, make_manifest(), preds(), ENABLED)
        self.assertEqual(r.status, INCONCLUSIVE)


class UndefinedMetricTest(unittest.TestCase):
    def test_undefined_regression_rate_is_inconclusive_not_pass(self):
        rep = make_report(overall={"regression_rate_given_baseline_correct": None})
        policy = dict(ENABLED, max_regression_rate=0.05)
        r = evaluate(rep, make_manifest(), preds(), policy)
        self.assertEqual(r.status, INCONCLUSIVE)


class CriticalCategoryTest(unittest.TestCase):
    def test_category_regression_limit_enforced(self):
        rep = make_report()
        rep["by_category"]["negation"]["baseline_correct_compressed_wrong"] = 3
        policy = dict(ENABLED, max_category_regressions={"negation": 0})
        r = evaluate(rep, make_manifest(), preds(), policy)
        self.assertEqual(r.status, FAIL)


class NoCompositeScoreTest(unittest.TestCase):
    def test_checks_are_reported_independently(self):
        r = evaluate(make_report(), make_manifest(), preds(), ENABLED)
        d = r.as_dict()
        self.assertIsInstance(d["checks"], list)
        self.assertGreater(len(d["checks"]), 1)
        # No single blended score is emitted anywhere.
        self.assertNotIn("score", d)
        for c in d["checks"]:
            self.assertIn("threshold", c)
            self.assertIn("observed", c)


if __name__ == "__main__":
    unittest.main()
