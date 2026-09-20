"""Benchmark schema, freeze, and question-leakage regression tests."""

import inspect
import json
import csv
import unittest

from src.benchmark import CANDIDATE_COUNT, DATASET, compress_context, dataset_hash, validate_benchmark
from src.benchmark_stats import summarize


class FakeTokenizer:
    def count(self, text):
        return len(text)


class RecordingProvider:
    def __init__(self):
        self.calls = []

    def complete(self, system, user, *, purpose, temperature=0):
        self.calls.append((system, user, purpose, temperature))
        if purpose == "semantic_judge":
            text = '{"valid": true, "reason": "complete"}'
        elif user.startswith("SOURCE CONTEXT:"):
            text = "fact one; fact two"
        else:
            text = "compact facts"
        return {"text": text, "usage": {}, "cached": False, "model": "test"}


class BenchmarkMethodologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = json.loads(DATASET.read_text(encoding="utf-8"))

    def test_dataset_validates_and_is_balanced(self):
        status = validate_benchmark(self.items)
        self.assertEqual(status["contexts"], 50)
        self.assertEqual(status["qa_pairs"], 250)
        self.assertEqual(set(status["categories"].values()), {5})

    def test_compressor_signature_excludes_questions_and_answers(self):
        self.assertEqual(list(inspect.signature(compress_context).parameters), ["context", "tokenizer", "provider", "candidate_count"])

    def test_questions_and_answers_never_enter_compression_prompts(self):
        provider = RecordingProvider()
        compress_context("safe source context", FakeTokenizer(), provider)
        transcript = "\n".join(system + "\n" + user for system, user, _, _ in provider.calls)
        self.assertNotIn("SENTINEL_QUESTION", transcript)
        self.assertNotIn("SENTINEL_ANSWER", transcript)
        self.assertFalse(any(purpose in {"answer", "judge"} for _, _, purpose, _ in provider.calls))

    def test_candidate_zero_is_original_and_caps_length(self):
        provider = RecordingProvider()
        result = compress_context("x", FakeTokenizer(), provider)
        self.assertEqual(result["candidates"][0]["candidate"], 0)
        self.assertEqual(result["candidates"][0]["text"], "x")
        self.assertEqual(result["winner"]["candidate"], 0)

    def test_candidate_budget_is_frozen(self):
        self.assertEqual(CANDIDATE_COUNT, 3)

    def test_hash_is_sha256(self):
        self.assertEqual(len(dataset_hash()), 64)

    def test_question_level_paired_transitions_match_raw_results(self):
        root = DATASET.parents[1] / "results" / "benchmark_v1" / "full_50"
        with (root / "representations.csv").open(encoding="utf-8") as handle:
            representations = list(csv.DictReader(handle))
        with (root / "qa_results.csv").open(encoding="utf-8") as handle:
            questions = list(csv.DictReader(handle))
        summary, paired = summarize(representations, questions)
        by_strategy = {row["strategy"]: row for row in summary}
        self.assertEqual(by_strategy["COMPACT_ENGLISH"]["additional_question_failures"], 25)
        self.assertEqual(by_strategy["COMPACT_ENGLISH"]["question_gains"], 6)
        self.assertEqual(by_strategy["COMPACT_ENGLISH"]["contexts_with_additional_failure"], 20)
        self.assertEqual(by_strategy["COMPACT_ENGLISH"]["contexts_saved_without_additional_qa_loss"], 30)
        self.assertEqual(by_strategy["TOKEN_OPTIMIZED"]["additional_question_failures"], 28)
        self.assertEqual(by_strategy["TOKEN_OPTIMIZED"]["question_gains"], 5)
        self.assertEqual(by_strategy["TOKEN_OPTIMIZED"]["contexts_with_additional_failure"], 21)
        self.assertEqual(by_strategy["TOKEN_OPTIMIZED"]["contexts_saved_without_additional_qa_loss"], 26)
        compact = [row for row in paired if row["strategy"] == "COMPACT_ENGLISH"]
        self.assertEqual(sum(row["caused_failure"] for row in compact), 20)


if __name__ == "__main__":
    unittest.main()
