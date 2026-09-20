"""Leakage and integrity tests for RISK_AWARE_COMPACT."""

import csv
import inspect
import json
import unittest

from src.benchmark import DATASET, HASH_FILE, dataset_hash
from src.risk_aware import PRIOR, RISK_CATEGORIES, compress_risk_aware, detect_risks


class FakeProvider:
    def __init__(self): self.calls = []
    def complete_schema(self, system, user, *, schema_name, schema, temperature=0):
        self.calls.append((system, user, "risk_detection"))
        return {"data": {"protected": [{"text": "must", "categories": ["REQUIREMENT"]}]}, "usage": {}, "cached": False}
    def complete(self, system, user, *, purpose, temperature=0):
        self.calls.append((system, user, purpose))
        return {"text": "compressed", "usage": {}, "cached": False}


class ParaphraseProvider(FakeProvider):
    def complete_schema(self, system, user, *, schema_name, schema, temperature=0):
        return {"data": {"protected": [{"text": "reduce margin below floor", "categories": ["COMPARISON_THRESHOLD"]}]}, "usage": {}, "cached": False}


class RiskAwareMethodologyTests(unittest.TestCase):
    def test_frozen_hash_matches(self):
        self.assertEqual(dataset_hash(), HASH_FILE.read_text().split()[0])

    def test_signatures_exclude_questions_answers_and_failures(self):
        self.assertEqual(list(inspect.signature(detect_risks).parameters), ["context", "provider"])
        self.assertEqual(list(inspect.signature(compress_risk_aware).parameters), ["context", "protected", "provider"])

    def test_prompts_only_receive_context_and_protected_spans(self):
        provider = FakeProvider(); context = "Operators MUST verify labels."
        detected = detect_risks(context, provider)
        self.assertEqual(detected["protected"][0]["text"], "MUST")
        compress_risk_aware(context, detected["protected"], provider)
        transcript = "\n".join(system + user for system, user, _ in provider.calls)
        for forbidden in ("SENTINEL_QUESTION", "SENTINEL_ANSWER", "previous QA failure", "judge verdict"):
            self.assertNotIn(forbidden, transcript)

    def test_completed_experiment_preserves_sixteen_context_subset(self):
        report = json.loads((PRIOR.parents[1] / "risk_aware_v1" / "report.json").read_text())
        self.assertEqual(len(report["selected_context_ids"]), 16)

    def test_corrected_paired_report_finds_twenty_question_level_failure_contexts(self):
        with (PRIOR / "paired_analysis.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        selected = {row["context_id"] for row in rows if row["strategy"] == "COMPACT_ENGLISH" and row["caused_failure"] == "True"}
        self.assertEqual(len(selected), 20)

    def test_risk_taxonomy_is_frozen(self):
        self.assertEqual(len(RISK_CATEGORIES), 11)

    def test_nonverbatim_detector_phrase_maps_to_exact_source_sentence(self):
        context = "The discount was declined because it would have reduced margin below the floor. Another fact follows."
        detected = detect_risks(context, ParaphraseProvider())
        self.assertEqual(detected["protected"][0]["text"], "The discount was declined because it would have reduced margin below the floor.")


if __name__ == "__main__": unittest.main()
