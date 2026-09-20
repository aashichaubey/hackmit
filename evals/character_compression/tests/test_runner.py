"""Runner behaviour: mock refusal in live mode, retry/fallback accounting,
independence of payloads, resume validation, and fixture validation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

from adapters import GenerationResult, MockCompressor, MockTarget  # noqa: E402
from run_evals import (  # noqa: E402
    MODE_CONTROL,
    MODE_PIPELINE,
    Config,
    Runner,
    validate_fixtures,
)
from score_evals import load_jsonl  # noqa: E402

CASES = load_jsonl(KIT / "eval_cases.jsonl")
OUT = KIT / "runs" / "_test"


class FailingTarget(MockTarget):
    """Fails the compressed arm N times, then succeeds."""

    def __init__(self, fail_times=1, **kw):
        super().__init__(**kw)
        self.remaining = fail_times
        self.calls = 0

    def generate(self, messages, config):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            return GenerationResult(output="", error="synthetic failure", model_id="mock-target-v1")
        return super().generate(messages, config)


class LiveModeGuardTest(unittest.TestCase):
    def test_live_mode_refuses_mock_compressor(self):
        cfg = Config(mode="live", compressor="mock", target="mock", token_counter=None)
        with self.assertRaisesRegex(RuntimeError, "live mode refused"):
            Runner(cfg, OUT)

    def test_mock_mode_allows_mocks(self):
        cfg = Config(mode="mock", compressor="mock", target="mock", token_counter=None)
        self.assertTrue(Runner(cfg, OUT).compressor.is_mock)

    def test_repo_compressor_reports_missing_integration(self):
        from adapters import RepoCompressor

        with self.assertRaises(NotImplementedError) as ctx:
            RepoCompressor(dictionary_path="nonexistent.jsonl")
        self.assertIn("does not exist yet", str(ctx.exception))


class LiveBudgetTest(unittest.TestCase):
    def test_live_call_budget_enforced(self):
        cfg = Config(mode="mock", compressor="mock", target="mock", token_counter=None)
        r = Runner(cfg, OUT, max_live_calls=3)
        r.cfg.mode = "live"  # bypass constructor guard to test the counter only
        with self.assertRaisesRegex(RuntimeError, "budget"):
            r.run(CASES[:5])
        self.assertLessEqual(r.live_calls, 3)


class RetryFallbackTest(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(mode="mock", compressor="mock", target="mock", token_counter=None)
        base.update(kw)
        return Config(**base)

    def test_first_attempt_mode_does_not_retry(self):
        cfg = self._cfg(experiment_mode="first_attempt")
        r = Runner(cfg, OUT)
        r.target = FailingTarget(fail_times=99)
        preds = r.run(CASES[:2])
        kinds = {a.kind for a in r.attempts}
        self.assertEqual(kinds, {"first_attempt"})
        self.assertTrue(all(p["compressed_error"] for p in preds))

    def test_pipeline_mode_records_retry_and_fallback_separately(self):
        cfg = self._cfg(
            experiment_mode=MODE_PIPELINE,
            retry={"max_retries": 1, "fallback_to_original": True},
        )
        r = Runner(cfg, OUT)
        r.target = FailingTarget(fail_times=99)
        preds = r.run(CASES[:1])
        kinds = [a.kind for a in r.attempts]
        self.assertIn("retry", kinds)
        self.assertIn("fallback", kinds)
        meta = preds[0]["compression_metadata"]
        self.assertTrue(meta["fell_back_to_original"])

    def test_fallback_is_not_credited_as_compression_success(self):
        """First-attempt quality must be preserved separately from the pipeline."""
        cfg = self._cfg(
            experiment_mode=MODE_PIPELINE,
            retry={"max_retries": 0, "fallback_to_original": True},
        )
        r = Runner(cfg, OUT)
        r.target = FailingTarget(fail_times=1)  # compressed first attempt fails
        preds = r.run(CASES[:1])
        meta = preds[0]["compression_metadata"]
        self.assertTrue(meta["fell_back_to_original"])
        self.assertEqual(meta["first_attempt_compressed_pass"], 0)


class ControlModeTest(unittest.TestCase):
    def test_control_mode_runs_original_in_both_arms(self):
        cfg = Config(mode="mock", compressor="mock", target="mock",
                     token_counter=None, experiment_mode=MODE_CONTROL)
        preds = Runner(cfg, OUT).run(CASES[:3])
        for p in preds:
            self.assertEqual(p["compressed_messages"], p["baseline_messages"])


class IndependenceTest(unittest.TestCase):
    def test_each_case_gets_fresh_independent_payload(self):
        cfg = Config(mode="mock", compressor="mock", target="mock", token_counter=None)
        preds = Runner(cfg, OUT).run(CASES[:6])
        ids = [p["case_id"] for p in preds]
        self.assertEqual(len(ids), len(set(ids)))
        # No message object is shared between records.
        seen_ids = set()
        for p in preds:
            for m in p["compressed_messages"]:
                self.assertNotIn(id(m), seen_ids)
                seen_ids.add(id(m))

    def test_arm_order_randomized_but_seed_reproducible(self):
        cfg = Config(mode="mock", compressor="mock", target="mock", token_counter=None, seed=5)
        a = [p["compression_metadata"]["arm_order"] for p in Runner(cfg, OUT).run(CASES[:10])]
        b = [p["compression_metadata"]["arm_order"] for p in Runner(cfg, OUT).run(CASES[:10])]
        self.assertEqual(a, b)                      # reproducible under the seed
        self.assertGreater(len({tuple(x) for x in a}), 1)  # genuinely shuffled

    def test_repeats_produce_same_run_id_set_for_every_case(self):
        cfg = Config(mode="mock", compressor="mock", target="mock",
                     token_counter=None, repeats=3)
        preds = Runner(cfg, OUT).run(CASES[:4])
        by_case = {}
        for p in preds:
            by_case.setdefault(p["case_id"], set()).add(p["run_id"])
        self.assertTrue(all(v == {0, 1, 2} for v in by_case.values()))


class FixtureValidationTest(unittest.TestCase):
    def test_supplied_fixtures_are_valid(self):
        self.assertEqual(validate_fixtures(CASES), [])

    def test_detects_duplicates_and_bad_roles(self):
        bad = [
            {"case_id": "a", "category": "x", "messages": [{"role": "user", "content": "q"}],
             "expected_output": "1", "grader": {"type": "exact"}},
            {"case_id": "a", "category": "x", "messages": [{"role": "bogus", "content": "q"}],
             "expected_output": "1", "grader": {"type": "exact"}},
        ]
        problems = validate_fixtures(bad)
        self.assertTrue(any("duplicate" in p for p in problems))
        self.assertTrue(any("bad role" in p for p in problems))

    def test_detects_unsupported_grader(self):
        bad = [{"case_id": "a", "category": "x",
                "messages": [{"role": "user", "content": "q"}],
                "expected_output": "1", "grader": {"type": "fuzzy_similarity"}}]
        self.assertTrue(any("unsupported grader" in p for p in validate_fixtures(bad)))


class FingerprintTest(unittest.TestCase):
    def test_fingerprint_changes_with_meaningful_config(self):
        a = Config(compressor="mock").fingerprint()
        b = Config(compressor="repo").fingerprint()
        self.assertNotEqual(a, b)

    def test_fingerprint_stable_for_cosmetic_change(self):
        a = Config(notes="one").fingerprint()
        b = Config(notes="two").fingerprint()
        self.assertEqual(a, b)


class CompressorOverheadTest(unittest.TestCase):
    def test_compressor_owns_dictionary_insertion(self):
        c = MockCompressor()
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "Who? Return only the name."}]
        out = c.compress(msgs, {})
        joined = " ".join(m["content"] for m in out.messages)
        self.assertIn("codebook", joined)
        self.assertEqual(out.metadata["dictionary_included_by"], "compressor")

    def test_scope_user_only_leaves_system_text_untouched(self):
        c = MockCompressor(scope="user_only")
        msgs = [{"role": "system", "content": "source data must stay"},
                {"role": "user", "content": "the following"}]
        out = c.compress(msgs, {})
        self.assertIn("source data must stay", out.messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
