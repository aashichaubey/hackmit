"""Regression tests for the question-independent methodology boundary."""

import inspect
import unittest

from src.dense_canonical import generate_dense_canonical, stability_metrics
from src.semantic_compiler import FORMATS, extract_ir, normalize_ir, serialize_ir
from src.representations import (
    GENERATED_STRATEGIES, encode_facts, extract_facts, generate_representations,
    generate_token_candidate, search_token_optimized, validate_semantics,
)


class RecordingProvider:
    def __init__(self):
        self.calls = []

    def complete(self, system, user, *, purpose, temperature=0):
        self.calls.append({"system": system, "user": user, "purpose": purpose, "temperature": temperature})
        text = "FACT_ALPHA\nFACT_BETA" if user.startswith("SOURCE CONTEXT:") else "ENCODED"
        return {"text": text, "usage": {}, "model": "test"}


class SearchProvider:
    def complete(self, system, user, *, purpose, temperature=0):
        if purpose == "semantic_judge":
            return {"text": '{"valid": true, "reason": "complete"}', "usage": {}, "model": "test"}
        attempt = int(user.split("SEARCH ATTEMPT ", 1)[1].split(" ", 1)[0])
        return {"text": "x" * (11 - attempt), "usage": {}, "model": "test"}


class CharacterTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def count(self, text):
        return len(text)


class MethodologyTests(unittest.TestCase):
    def test_generation_api_cannot_accept_questions_or_answers(self):
        self.assertEqual(list(inspect.signature(generate_representations).parameters), ["context", "provider"])
        self.assertEqual(list(inspect.signature(extract_facts).parameters), ["context", "provider"])
        self.assertEqual(list(inspect.signature(encode_facts).parameters), ["facts", "strategy", "provider"])
        self.assertEqual(list(inspect.signature(generate_token_candidate).parameters), ["facts", "attempt", "provider"])
        self.assertEqual(list(inspect.signature(validate_semantics).parameters), ["facts", "candidate", "provider"])
        self.assertEqual(list(inspect.signature(search_token_optimized).parameters), ["original", "facts", "tokenizer", "provider", "candidate_count"])
        self.assertEqual(list(inspect.signature(generate_dense_canonical).parameters), ["facts", "provider"])
        self.assertEqual(list(inspect.signature(extract_ir).parameters), ["context", "provider"])

    def test_questions_never_enter_generation_prompts(self):
        provider = RecordingProvider()
        sentinel_question = "SENTINEL_SECRET_QUESTION"
        sentinel_answer = "SENTINEL_SECRET_ANSWER"
        context = "Public context contains neither sentinel."
        generate_representations(context, provider)
        transcript = "\n".join(call["system"] + call["user"] for call in provider.calls)
        self.assertNotIn(sentinel_question, transcript)
        self.assertNotIn(sentinel_answer, transcript)
        self.assertTrue(all(call["purpose"] == "generation" for call in provider.calls))

    def test_every_encoder_receives_identical_shared_facts(self):
        provider = RecordingProvider()
        generate_representations("A context.", provider)
        encoder_calls = provider.calls[1:]
        self.assertEqual(len(encoder_calls), len(GENERATED_STRATEGIES))
        self.assertTrue(all("FACTS:\nFACT_ALPHA\nFACT_BETA" in call["user"] for call in encoder_calls))

    def test_canonical_temperature_is_zero(self):
        provider = RecordingProvider()
        encode_facts("FACT", "CANONICAL", provider)
        self.assertEqual(provider.calls[0]["temperature"], 0)

    def test_search_counts_ten_and_selects_shortest_valid_candidate(self):
        result = search_token_optimized("ORIGINAL-LONG", "FACT", CharacterTokenizer(), SearchProvider())
        self.assertEqual(len(result["candidates"]), 11)
        self.assertEqual(result["winner"]["candidate"], 10)
        self.assertEqual(result["winner"]["token_count"], 1)

    def test_original_candidate_zero_prevents_longer_selection(self):
        result = search_token_optimized("x", "FACT", CharacterTokenizer(), SearchProvider())
        self.assertEqual(result["winner"]["candidate"], 0)
        self.assertEqual(result["candidates"][0]["source"], "original_baseline")

    def test_stability_metrics_detect_exact_and_variable_runs(self):
        exact = stability_metrics(["abc"] * 10, CharacterTokenizer())
        self.assertTrue(exact["all_text_identical"])
        self.assertEqual(exact["pairwise_exact_token_rate"], 1)
        varied = stability_metrics(["abc"] * 9 + ["abd"], CharacterTokenizer())
        self.assertFalse(varied["all_token_sequences_identical"])
        self.assertLess(varied["average_pairwise_token_similarity"], 1)

    def test_normalization_and_serialization_are_deterministic(self):
        ir = {"facts": [
            {"subject": "  Lab ", "relation": "has", "objects": ["B", "a"], "time": [], "location": [], "condition": [], "exception": [], "qualifier": []},
            {"subject": "lab", "relation": "has", "objects": ["a", "B"], "time": [], "location": [], "condition": [], "exception": [], "qualifier": []},
        ]}
        normalized = normalize_ir(ir)
        self.assertEqual(len(normalized["facts"]), 1)
        outputs = [serialize_ir(normalized, FORMATS[0]) for _ in range(100)]
        self.assertEqual(len(set(outputs)), 1)


if __name__ == "__main__":
    unittest.main()
