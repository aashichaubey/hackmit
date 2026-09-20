"""Scorer/accounting tests. These verify harness arithmetic, NOT model quality."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

from score_evals import cluster_bootstrap, load_jsonl, metrics, validate_and_pair  # noqa: E402

CASES = load_jsonl(KIT / "eval_cases.jsonl")


def _pred(case, run_id, baseline_ok, compressed_ok, **extra):
    """Build one prediction row with controlled pass/fail outcomes."""
    gold = case["expected_output"]
    good = gold if case["grader"]["type"] == "exact" else '{"Mira": 2, "Tovan": 1}'
    row = {
        "case_id": case["case_id"],
        "run_id": run_id,
        "baseline_output": good if baseline_ok else "DELIBERATE_TEST_WRONG",
        "compressed_output": good if compressed_ok else "DELIBERATE_TEST_WRONG",
    }
    row.update(extra)
    return row


class ArithmeticFixtureTest(unittest.TestCase):
    """The exact fixture required by the brief.

    original passes [1,1,0,0], compressed passes [1,0,1,0], 100 original /
    50 compressed input tokens per pair. This is a scorer unit test, not a
    model result.
    """

    def test_required_arithmetic_fixture(self):
        cases = CASES[:4]
        baseline = [1, 1, 0, 0]
        compressed = [1, 0, 1, 0]
        preds = [
            _pred(c, 0, bool(b), bool(k), baseline_input_tokens=100, compressed_input_tokens=50)
            for c, b, k in zip(cases, baseline, compressed)
        ]
        result = metrics(validate_and_pair(cases, preds, allow_partial=True))

        self.assertEqual(result["baseline_accuracy"], 0.5)
        self.assertEqual(result["compressed_accuracy"], 0.5)
        self.assertEqual(result["accuracy_delta_pp"], 0.0)
        self.assertEqual(result["baseline_correct_compressed_wrong"], 1)
        self.assertEqual(result["baseline_wrong_compressed_correct"], 1)
        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["both_wrong"], 1)
        self.assertEqual(result["regression_rate_given_baseline_correct"], 0.5)
        self.assertEqual(result["tokens"]["net_input_token_savings"], 0.5)


class PairedOutcomeTest(unittest.TestCase):
    def test_all_four_outcomes_counted(self):
        cases = CASES[:4]
        preds = [
            _pred(cases[0], 0, True, True),
            _pred(cases[1], 0, True, False),
            _pred(cases[2], 0, False, True),
            _pred(cases[3], 0, False, False),
        ]
        r = metrics(validate_and_pair(cases, preds, allow_partial=True))
        self.assertEqual(
            (r["both_correct"], r["baseline_correct_compressed_wrong"],
             r["baseline_wrong_compressed_correct"], r["both_wrong"]),
            (1, 1, 1, 1),
        )

    def test_regression_rate_undefined_when_no_baseline_correct(self):
        cases = CASES[:2]
        preds = [_pred(c, 0, False, False) for c in cases]
        r = metrics(validate_and_pair(cases, preds, allow_partial=True))
        self.assertIsNone(r["regression_rate_given_baseline_correct"])

    def test_terminal_error_is_failure_not_missing(self):
        cases = CASES[:2]
        preds = [_pred(c, 0, True, True) for c in cases]
        preds[0]["compressed_output"] = ""
        preds[0]["compressed_error"] = "synthetic terminal timeout"
        r = metrics(validate_and_pair(cases, preds, allow_partial=True))
        self.assertEqual(r["compressed_error_count"], 1)
        self.assertEqual(r["baseline_correct_compressed_wrong"], 1)


class TokenAccountingTest(unittest.TestCase):
    def test_negative_savings_reported_not_clamped(self):
        cases = CASES[:2]
        preds = [
            _pred(c, 0, True, True, baseline_input_tokens=100, compressed_input_tokens=180)
            for c in cases
        ]
        t = metrics(validate_and_pair(cases, preds, allow_partial=True))["tokens"]
        self.assertAlmostEqual(t["net_input_token_savings"], -0.8)
        self.assertEqual(t["expansion_rate"], 1.0)

    def test_dictionary_overhead_can_erase_savings(self):
        """A large codebook preamble can make the compressed request bigger."""
        cases = CASES[:2]
        # 60 tokens of payload saved, but a 70-token dictionary added.
        preds = [
            _pred(c, 0, True, True, baseline_input_tokens=100, compressed_input_tokens=40 + 70)
            for c in cases
        ]
        t = metrics(validate_and_pair(cases, preds, allow_partial=True))["tokens"]
        self.assertLess(t["net_input_token_savings"], 0)

    def test_null_measurements_are_unknown_not_zero(self):
        cases = CASES[:2]
        preds = [_pred(c, 0, True, True, baseline_input_tokens=100) for c in cases]
        r = metrics(validate_and_pair(cases, preds, allow_partial=True))
        self.assertIsNone(r["tokens"])  # withheld, not computed as zero
        self.assertEqual(r["token_measurement_coverage"], 0.0)

    def test_partial_token_coverage_withholds_totals(self):
        cases = CASES[:2]
        preds = [
            _pred(cases[0], 0, True, True, baseline_input_tokens=100, compressed_input_tokens=50),
            _pred(cases[1], 0, True, True),
        ]
        r = metrics(validate_and_pair(cases, preds, allow_partial=True))
        self.assertIsNone(r["tokens"])
        self.assertEqual(r["token_measurement_coverage"], 0.5)


class ValidationTest(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        cases = CASES[:2]
        preds = [_pred(c, 0, True, True) for c in cases]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_and_pair(cases, preds + [preds[0]], allow_partial=True)

    def test_unequal_repeats_rejected(self):
        cases = CASES[:2]
        preds = [_pred(c, 0, True, True) for c in cases]
        preds.append(_pred(cases[0], 1, True, True))
        with self.assertRaisesRegex(ValueError, "same run_id set"):
            validate_and_pair(cases, preds, allow_partial=True)

    def test_mixed_configuration_rejected(self):
        cases = CASES[:2]
        preds = [_pred(c, 0, True, True) for c in cases]
        preds[0]["dictionary_version"] = "v1"
        preds[1]["dictionary_version"] = "v2"
        with self.assertRaisesRegex(ValueError, "Mixed dictionary_version"):
            validate_and_pair(cases, preds, allow_partial=True)

    def test_null_output_refused(self):
        cases = CASES[:1]
        preds = [_pred(cases[0], 0, True, True)]
        preds[0]["compressed_output"] = None
        with self.assertRaisesRegex(ValueError, "actual string"):
            validate_and_pair(cases, preds, allow_partial=True)

    def test_malformed_log_rejected(self, ):
        bad = KIT / "tests" / "_malformed.jsonl"
        bad.write_text('{"case_id": "x", }\n', encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_jsonl(bad)
        finally:
            bad.unlink()


class BootstrapTest(unittest.TestCase):
    def test_cluster_resampling_is_reproducible_and_grouped(self):
        cases = CASES[:6]
        preds = [
            _pred(c, 0, True, True, baseline_input_tokens=100, compressed_input_tokens=60)
            for c in cases
        ]
        rows = validate_and_pair(cases, preds, allow_partial=True)
        a = cluster_bootstrap(rows, 200, 17)
        b = cluster_bootstrap(rows, 200, 17)
        self.assertEqual(a, b)
        self.assertEqual(a["clusters"], len({r["cluster_id"] for r in rows}))
        self.assertIsNotNone(a["net_token_savings_95pct_interval"])


if __name__ == "__main__":
    unittest.main()
