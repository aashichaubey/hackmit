"""Scorer unit tests, NOT real compressor/LLM measurements."""
import copy
import json
from pathlib import Path
import unittest

from score_evals import (cluster_bootstrap, grade, json_equal, load_jsonl,
                         metrics, strict_json, validate_and_pair)

ROOT = Path(__file__).parent


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_jsonl(ROOT / 'eval_cases.jsonl')

    def fake_predictions(self, selected=None):
        records = []
        for case in selected or self.cases:
            gold = case['expected_output']
            text = json.dumps(gold, ensure_ascii=False) if case['grader']['type'] == 'json' else gold
            records.append({'case_id': case['case_id'], 'run_id': 0,
                            'baseline_output': text, 'compressed_output': text})
        return records

    def test_all_40_gold_fixtures(self):
        self.assertEqual(len(self.cases), 40)
        for case, pred in zip(self.cases, self.fake_predictions()):
            self.assertTrue(grade(case, pred['baseline_output']), case['case_id'])
            self.assertFalse(grade(case, 'DELIBERATE_TEST_ERROR'), case['case_id'])

    def test_strict_format(self):
        case = next(c for c in self.cases if c['case_id'] == 'format_two_lines')
        self.assertFalse(grade(case, 'red\nblue\n'))
        self.assertFalse(grade(case, '- red\n- blue'))

    def test_json_key_order(self):
        case = next(c for c in self.cases if c['case_id'] == 'format_json')
        self.assertTrue(grade(case, '{"Tovan": 1, "Mira": 2}'))
        self.assertFalse(grade(case, '{"Tovan": true, "Mira": 2}'))
        self.assertFalse(grade(case, '{"Tovan": 1.0, "Mira": 2}'))
        self.assertFalse(grade(case, '```json\n{"Tovan": 1, "Mira": 2}\n```'))

    def test_json_rejects_duplicates_and_nan(self):
        with self.assertRaises(ValueError):
            strict_json('{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            strict_json('{"x":NaN}')
        self.assertFalse(json_equal(True, 1))

    def test_unmeasured_outputs_are_not_scored(self):
        preds = load_jsonl(ROOT / 'predictions.template.jsonl')
        with self.assertRaisesRegex(ValueError, 'actual string'):
            validate_and_pair(self.cases, preds, False)

    def test_missing_counts_are_unknown(self):
        rows = validate_and_pair(self.cases, self.fake_predictions(), False)
        result = metrics(rows)
        self.assertIsNone(result['tokens'])
        self.assertIsNone(result['cost'])
        self.assertEqual(result['token_measurement_coverage'], 0)

    def test_regression_and_recovery_not_cancelled(self):
        preds = self.fake_predictions()
        preds[0]['compressed_output'] = 'WRONG'
        preds[1]['baseline_output'] = 'WRONG'
        rows = validate_and_pair(self.cases, preds, False)
        result = metrics(rows)
        self.assertEqual(result['accuracy_delta_pp'], 0)
        self.assertEqual(result['baseline_correct_compressed_wrong'], 1)
        self.assertEqual(result['baseline_wrong_compressed_correct'], 1)
        self.assertAlmostEqual(result['regression_rate_given_baseline_correct'], 1/39)

    def test_errors_count_as_failures(self):
        preds = self.fake_predictions()
        preds[0]['compressed_error'] = 'synthetic unit-test timeout'
        result = metrics(validate_and_pair(self.cases, preds, False))
        self.assertEqual(result['compressed_error_count'], 1)
        self.assertEqual(result['baseline_correct_compressed_wrong'], 1)

    def test_token_savings_includes_expansion(self):
        preds = self.fake_predictions(self.cases[:2])
        for pred, b, c in zip(preds, [100, 100], [40, 120]):
            pred.update(baseline_input_tokens=b, compressed_input_tokens=c)
        rows = validate_and_pair(self.cases[:2], preds, False)
        result = metrics(rows)['tokens']
        self.assertAlmostEqual(result['net_input_token_savings'], .2)
        self.assertAlmostEqual(result['full_request_compression_ratio'], 1.25)
        self.assertEqual(result['expansion_rate'], .5)

    def test_missing_duplicate_and_unequal_repeats(self):
        preds = self.fake_predictions()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            validate_and_pair(self.cases, preds[:-1], False)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_and_pair(self.cases, preds + [preds[0]], False)
        extra = copy.deepcopy(preds[0])
        extra['run_id'] = 1
        with self.assertRaisesRegex(ValueError, 'same run_id set'):
            validate_and_pair(self.cases, preds + [extra], False)

    def test_mixed_models_rejected(self):
        preds = self.fake_predictions()
        preds[0]['model_id'] = 'test_model_a'
        preds[1]['model_id'] = 'test_model_b'
        with self.assertRaisesRegex(ValueError, 'Mixed model_id'):
            validate_and_pair(self.cases, preds, False)

    def test_bootstrap_reproducible(self):
        preds = self.fake_predictions()
        preds[0]['compressed_output'] = 'WRONG'
        rows = validate_and_pair(self.cases, preds, False)
        a = cluster_bootstrap(rows, 100, 17)
        b = cluster_bootstrap(rows, 100, 17)
        self.assertEqual(a, b)
        self.assertIsNone(a['net_token_savings_95pct_interval'])


if __name__ == '__main__':
    unittest.main()
