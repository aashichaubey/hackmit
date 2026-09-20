"""Offline tests for OpenRouter response parsing.

Regression cover for a real integration finding: deepseek-v4-flash is a
reasoning model that can return `content: null` with the whole output budget
spent on reasoning tokens. Grading that as a wrong answer would silently
understate accuracy in both arms, so it must surface as an error.

No network: urlopen is stubbed with recorded-shape payloads.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

import adapters  # noqa: E402
from adapters import OpenRouterTarget  # noqa: E402

MSGS = [{"role": "user", "content": "Who hid the silver key? Return only the name."}]


def _response(payload: dict) -> io.BytesIO:
    buf = io.BytesIO(json.dumps(payload).encode("utf-8"))
    buf.__enter__ = lambda: buf  # type: ignore[attr-defined]
    buf.__exit__ = lambda *a: None  # type: ignore[attr-defined]
    return buf


def _target() -> OpenRouterTarget:
    with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-v1-TESTONLY"}):
        return OpenRouterTarget(model="deepseek/deepseek-v4-flash")


class ReasoningModelTest(unittest.TestCase):
    def test_truncated_reasoning_becomes_error_not_wrong_answer(self):
        payload = {
            "id": "gen-test",
            "model": "deepseek/deepseek-v4-flash",
            "choices": [{
                "finish_reason": "length",
                "message": {"role": "assistant", "content": None, "reasoning": "thinking..."},
            }],
            "usage": {
                "prompt_tokens": 83, "completion_tokens": 32, "cost": 5.4e-06,
                "completion_tokens_details": {"reasoning_tokens": 32},
            },
        }
        with mock.patch.object(urllib.request, "urlopen", return_value=_response(payload)):
            r = _target().generate(MSGS, {"max_tokens": 32})
        self.assertEqual(r.output, "")
        self.assertIsNotNone(r.error)
        self.assertIn("truncated_before_answer", r.error)
        self.assertEqual(r.finish_reason, "length")
        self.assertEqual(r.reasoning_tokens, 32)
        self.assertTrue(r.failed)

    def test_normal_answer_parsed_with_provider_usage_and_cost(self):
        payload = {
            "id": "gen-test2",
            "model": "deepseek/deepseek-v4-flash",
            "choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Mira"}}],
            "usage": {
                "prompt_tokens": 83, "completion_tokens": 2, "cost": 1.58e-06,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
        with mock.patch.object(urllib.request, "urlopen", return_value=_response(payload)):
            r = _target().generate(MSGS, {"max_tokens": 2048})
        self.assertEqual(r.output, "Mira")
        self.assertIsNone(r.error)
        self.assertEqual(r.input_tokens, 83)
        self.assertEqual(r.token_count_source, "provider_usage")
        self.assertAlmostEqual(r.cost_usd, 1.58e-06)

    def test_empty_content_with_stop_is_a_real_empty_answer(self):
        """finish_reason=stop and empty content is a genuine (wrong) answer."""
        payload = {
            "id": "g", "model": "m",
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }
        with mock.patch.object(urllib.request, "urlopen", return_value=_response(payload)):
            r = _target().generate(MSGS, {})
        self.assertEqual(r.output, "")
        self.assertIsNone(r.error)  # graded as wrong, not an infrastructure failure


class UsageAccountingTest(unittest.TestCase):
    def test_missing_usage_leaves_tokens_unknown_not_zero(self):
        payload = {"id": "g", "model": "m",
                   "choices": [{"finish_reason": "stop", "message": {"content": "X"}}]}
        with mock.patch.object(urllib.request, "urlopen", return_value=_response(payload)):
            r = _target().generate(MSGS, {})
        self.assertIsNone(r.input_tokens)
        self.assertIsNone(r.token_count_source)
        self.assertIsNone(r.cost_usd)

    def test_provider_cost_preferred_over_supplied_pricing(self):
        payload = {"id": "g", "model": "m",
                   "choices": [{"finish_reason": "stop", "message": {"content": "X"}}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.5}}
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-v1-TESTONLY"}):
            t = OpenRouterTarget(pricing={"input_per_token": 1.0, "output_per_token": 1.0})
        with mock.patch.object(urllib.request, "urlopen", return_value=_response(payload)):
            r = t.generate(MSGS, {})
        self.assertEqual(r.cost_usd, 0.5)  # observed billing, not 110.0

    def test_malformed_response_is_an_error(self):
        with mock.patch.object(urllib.request, "urlopen", return_value=_response({"nope": 1})):
            r = _target().generate(MSGS, {})
        self.assertIn("malformed_response", r.error)
        self.assertEqual(r.output, "")


class CredentialTest(unittest.TestCase):
    def test_missing_key_raises_before_any_call(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                OpenRouterTarget()

    def test_errors_are_redacted(self):
        leaked = "Authorization: Bearer sk-or-v1-abcdef123456 failed"
        self.assertNotIn("abcdef123456", adapters._redact(leaked))
        self.assertIn("REDACTED", adapters._redact(leaked))


if __name__ == "__main__":
    unittest.main()
