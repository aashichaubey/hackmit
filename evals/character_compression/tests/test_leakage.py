"""Gold-leakage tests.

The single most dangerous failure mode in this harness is an expected answer
reaching the compressor or the target LLM: the run would look excellent and
mean nothing. These tests assert the guard works and that the real code paths
actually route through it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

from adapters import (  # noqa: E402
    GOLD_FIELDS,
    GoldLeakageError,
    MockCompressor,
    MockTarget,
    assert_no_gold_fields,
    sanitize_messages,
)
from run_evals import Config, Runner  # noqa: E402
from score_evals import load_jsonl  # noqa: E402

CASES = load_jsonl(KIT / "eval_cases.jsonl")


class RecordingCompressor(MockCompressor):
    """Captures exactly what it was handed."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = []

    def compress(self, messages, config):
        self.seen.append(messages)
        return super().compress(messages, config)


class RecordingTarget(MockTarget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = []

    def generate(self, messages, config):
        self.seen.append(messages)
        return super().generate(messages, config)


class GuardTest(unittest.TestCase):
    def test_guard_detects_each_gold_field(self):
        for fld in GOLD_FIELDS:
            with self.assertRaises(GoldLeakageError):
                assert_no_gold_fields({"role": "user", "content": "x", fld: "LEAK"}, "t")

    def test_guard_detects_nested_leak(self):
        payload = [{"role": "user", "content": "x"}, {"meta": {"deep": {"grader": {}}}}]
        with self.assertRaises(GoldLeakageError):
            assert_no_gold_fields(payload, "t")

    def test_sanitize_strips_everything_but_role_and_content(self):
        msgs = [{"role": "user", "content": "hi", "expected_output": "LEAK"}]
        self.assertEqual(sanitize_messages(msgs), [{"role": "user", "content": "hi"}])

    def test_clean_payload_passes(self):
        assert_no_gold_fields([{"role": "user", "content": "fine"}], "t")


class RunnerLeakageTest(unittest.TestCase):
    def _runner(self, cases):
        cfg = Config(mode="mock", compressor="mock", target="mock", token_counter=None)
        r = Runner(cfg, KIT / "runs" / "_test")
        r.compressor = RecordingCompressor()
        r.target = RecordingTarget()
        return r, cfg

    def test_compressor_never_receives_gold(self):
        cases = CASES[:5]
        r, _ = self._runner(cases)
        r.run(cases)
        self.assertTrue(r.compressor.seen)
        for payload in r.compressor.seen:
            assert_no_gold_fields(payload, "compressor")
            for m in payload:
                self.assertEqual(set(m), {"role", "content"})

    def test_target_never_receives_gold(self):
        cases = CASES[:5]
        r, _ = self._runner(cases)
        r.run(cases)
        self.assertTrue(r.target.seen)
        for payload in r.target.seen:
            assert_no_gold_fields(payload, "target")

    def test_gold_absent_from_saved_inference_fields(self):
        cases = CASES[:5]
        r, _ = self._runner(cases)
        for pred in r.run(cases):
            assert_no_gold_fields(pred["compressed_messages"], "saved compressed")
            assert_no_gold_fields(pred["baseline_messages"], "saved baseline")

    def test_expected_output_string_never_appears_in_payload(self):
        """Belt-and-braces: the gold *value* must not appear either."""
        cases = [c for c in CASES[:12] if len(str(c["expected_output"])) > 3]
        r, _ = self._runner(cases)
        r.run(cases)
        golds = {str(c["expected_output"]) for c in cases}
        for payload in r.target.seen:
            blob = " ".join(m["content"] for m in payload)
            for gold in golds:
                # The gold may legitimately occur inside the passage (e.g. a
                # name); assert only that it was not appended as an answer.
                self.assertFalse(
                    blob.rstrip().endswith("Answer: " + gold),
                    f"gold {gold!r} leaked as an answer",
                )


if __name__ == "__main__":
    unittest.main()
