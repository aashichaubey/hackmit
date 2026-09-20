#!/usr/bin/env python3
"""Score paired original/compressed LLM outputs. Python 3.10+, standard library only.

This program does NOT invoke a compressor or an LLM. Supply actual measured
outputs and full-request token counts in the predictions JSONL file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any


def reject_constant(value: str) -> None:
    raise ValueError(f'Non-JSON numeric constant: {value}')


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'Duplicate JSON key: {key}')
        out[key] = value
    return out


def strict_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = strict_json(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
            if not isinstance(obj, dict):
                raise ValueError(f'{path}:{line_number}: expected a JSON object')
            records.append(obj)
    if not records:
        raise ValueError(f'{path}: no records')
    return records


def json_equal(actual: Any, expected: Any) -> bool:
    """Type-sensitive; key order ignored, array order preserved. No True == 1."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            json_equal(actual[k], expected[k]) for k in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            json_equal(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def grade(case: dict[str, Any], output: str) -> bool:
    spec = case['grader']
    expected = case['expected_output']
    if spec['type'] == 'exact':
        if not isinstance(expected, str):
            raise ValueError(f'{case["case_id"]}: exact grader requires a string gold')
        if spec.get('strip_outer_whitespace', False):
            return output.strip() == expected.strip()
        return output == expected
    if spec['type'] == 'json':
        try:
            return json_equal(strict_json(output), expected)
        except (ValueError, json.JSONDecodeError, RecursionError):
            return False
    raise ValueError(f'Unsupported grader: {spec["type"]}')


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def complete_values(rows: list[dict[str, Any]], key: str) -> list[float] | None:
    vals = [r.get(key) for r in rows]
    return None if any(v is None for v in vals) else vals


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    baseline_correct = sum(r['baseline_pass'] for r in rows)
    compressed_correct = sum(r['compressed_pass'] for r in rows)
    regressions = sum(r['baseline_pass'] and not r['compressed_pass'] for r in rows)
    recoveries = sum(not r['baseline_pass'] and r['compressed_pass'] for r in rows)
    result: dict[str, Any] = {
        'pairs': n,
        'cases': len({r['case_id'] for r in rows}),
        'baseline_accuracy': baseline_correct / n,
        'compressed_accuracy': compressed_correct / n,
        'accuracy_delta_pp': 100 * (compressed_correct - baseline_correct) / n,
        'both_correct': baseline_correct - regressions,
        'baseline_correct_compressed_wrong': regressions,
        'baseline_wrong_compressed_correct': recoveries,
        'both_wrong': n - baseline_correct - recoveries,
        'regression_rate_given_baseline_correct': safe_div(regressions, baseline_correct),
        'literal_output_agreement': sum(
            r['baseline_output'].strip() == r['compressed_output'].strip() for r in rows
        ) / n,
        'baseline_error_count': sum(bool(r.get('baseline_error')) for r in rows),
        'compressed_error_count': sum(bool(r.get('compressed_error')) for r in rows),
    }
    b = complete_values(rows, 'baseline_input_tokens')
    c = complete_values(rows, 'compressed_input_tokens')
    token_coverage = sum(
        r.get('baseline_input_tokens') is not None and r.get('compressed_input_tokens') is not None
        for r in rows
    ) / n
    result['token_measurement_coverage'] = token_coverage
    result['tokens'] = None
    if b is not None and c is not None:
        savings = [1 - ct / bt for bt, ct in zip(b, c)]
        result['tokens'] = {
            'baseline_total_input': sum(b),
            'compressed_total_input': sum(c),
            'net_input_token_savings': 1 - sum(c) / sum(b),
            'full_request_compression_ratio': sum(b) / sum(c),
            'mean_per_request_savings': statistics.mean(savings),
            'p50_per_request_savings': quantile(savings, 0.5),
            'p05_per_request_savings': quantile(savings, 0.05),
            'expansion_rate': sum(ct > bt for bt, ct in zip(b, c)) / n,
            'baseline_correct_per_1000_input_tokens': 1000 * baseline_correct / sum(b),
            'compressed_correct_per_1000_input_tokens': 1000 * compressed_correct / sum(c),
            'baseline_input_tokens_per_correct': safe_div(sum(b), baseline_correct),
            'compressed_input_tokens_per_correct': safe_div(sum(c), compressed_correct),
        }
    for name, field in [('baseline', 'baseline_latency_ms'),
                        ('compressed_pipeline', 'compressed_pipeline_latency_ms')]:
        vals = complete_values(rows, field)
        result[name + '_latency_ms'] = None if vals is None else {
            'p50': quantile(vals, 0.5), 'p95': quantile(vals, 0.95), 'mean': statistics.mean(vals)
        }
    bc = complete_values(rows, 'baseline_cost_usd')
    cc = complete_values(rows, 'compressed_pipeline_cost_usd')
    result['cost'] = None
    if bc is not None and cc is not None:
        result['cost'] = {
            'baseline_total_usd': sum(bc),
            'compressed_pipeline_total_usd': sum(cc),
            'net_cost_savings': (1 - sum(cc) / sum(bc)) if sum(bc) else None,
            'baseline_usd_per_correct': safe_div(sum(bc), baseline_correct),
            'compressed_pipeline_usd_per_correct': safe_div(sum(cc), compressed_correct),
        }
    return result


def cluster_bootstrap(rows: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    """Resample source clusters; keep paired outcomes and repeated trials together."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row['cluster_id']].append(row)
    tokens_complete = all(
        r.get('baseline_input_tokens') is not None and r.get('compressed_input_tokens') is not None
        for r in rows
    )
    stats = []
    for group in groups.values():
        stats.append((len(group),
                      sum(r['compressed_pass'] - r['baseline_pass'] for r in group),
                      sum(r['baseline_input_tokens'] for r in group) if tokens_complete else 0,
                      sum(r['compressed_input_tokens'] for r in group) if tokens_complete else 0))
    rng = random.Random(seed)
    deltas, savings = [], []
    for _ in range(iterations):
        sample = rng.choices(stats, k=len(stats))
        n = sum(s[0] for s in sample)
        deltas.append(100 * sum(s[1] for s in sample) / n)
        if tokens_complete:
            savings.append(1 - sum(s[3] for s in sample) / sum(s[2] for s in sample))
    return {
        'method': 'paired source-cluster percentile bootstrap',
        'clusters': len(groups),
        'resamples': iterations,
        'seed': seed,
        'accuracy_delta_pp_95pct_interval': [quantile(deltas, .025), quantile(deltas, .975)],
        'net_token_savings_95pct_interval': [quantile(savings, .025), quantile(savings, .975)]
            if tokens_complete else None,
        'caution': ('Exploratory intervals, not a release certification. Small or unrepresentative '
                    'source samples and all-equal outcomes can yield misleadingly narrow intervals. '
                    'Zero observed errors does not establish zero future error risk.')
    }


def validate_and_pair(cases: list[dict[str, Any]], predictions: list[dict[str, Any]],
                      allow_partial: bool) -> list[dict[str, Any]]:
    case_map = {c['case_id']: c for c in cases}
    if len(case_map) != len(cases):
        raise ValueError('Duplicate case_id in fixtures')
    seen: set[tuple[str, str]] = set()
    runs_by_case: dict[str, set[str]] = defaultdict(set)
    rows = []
    for pred in predictions:
        cid = pred.get('case_id')
        if cid not in case_map:
            raise ValueError(f'Unknown case_id: {cid}')
        run = json.dumps(pred.get('run_id', 0), sort_keys=True)
        if (cid, run) in seen:
            raise ValueError(f'Duplicate case/run: {cid}/{run}')
        seen.add((cid, run))
        runs_by_case[cid].add(run)
        for arm in ['baseline', 'compressed']:
            if not isinstance(pred.get(arm + '_output'), str):
                raise ValueError(f'{cid}/{run}: {arm}_output must be an actual string, not null. '
                                 'For a failed request use an empty string and set its error field.')
        for field in ['baseline_input_tokens', 'compressed_input_tokens',
                      'baseline_output_tokens', 'compressed_output_tokens']:
            value = pred.get(field)
            if value is not None and (type(value) is not int or value < (1 if 'input' in field else 0)):
                raise ValueError(f'{cid}/{run}: {field} must be a positive input-token integer, '
                                 'a nonnegative output-token integer, or null')
        for field in ['baseline_latency_ms', 'compressed_pipeline_latency_ms',
                      'baseline_cost_usd', 'compressed_pipeline_cost_usd']:
            value = pred.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f'{cid}/{run}: invalid {field}')
        case = case_map[cid]
        row = dict(pred)
        row.update(category=case['category'], cluster_id=case.get('cluster_id', cid),
                   expected_output=case['expected_output'], grader=case['grader'])
        row['baseline_pass'] = int(not pred.get('baseline_error') and grade(case, pred['baseline_output']))
        row['compressed_pass'] = int(not pred.get('compressed_error') and grade(case, pred['compressed_output']))
        rows.append(row)
    missing = set(case_map) - set(runs_by_case)
    if missing and not allow_partial:
        raise ValueError(f'Missing {len(missing)} cases. Run all fixtures or explicitly use --allow-partial.')
    run_sets = list(runs_by_case.values())
    if run_sets and any(s != run_sets[0] for s in run_sets[1:]):
        raise ValueError('Every included case must have the same run_id set; unequal repeats bias aggregates.')
    for field in ['model_id', 'tokenizer_id', 'compression_version', 'dictionary_version']:
        values = {str(r[field]) for r in rows if r.get(field) is not None}
        if len(values) > 1:
            raise ValueError(f'Mixed {field} values. Score each experiment configuration separately.')
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('eval_cases.jsonl'))
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('report.json'))
    parser.add_argument('--details', type=Path, default=Path('scored_cases.jsonl'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('--bootstrap must be at least 100')
    try:
        cases = load_jsonl(args.cases)
        rows = validate_and_pair(cases, load_jsonl(args.predictions), args.allow_partial)
        categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            categories[row['category']].append(row)
        report = {
            'fixture_count': len(cases),
            'case_coverage': len({r['case_id'] for r in rows}) / len(cases),
            'fixture_splits': sorted({c.get('split', 'unspecified') for c in cases}),
            'overall': metrics(rows),
            'by_category': {k: metrics(v) for k, v in sorted(categories.items())},
            'uncertainty': cluster_bootstrap(rows, args.bootstrap, args.seed),
            'notes': [
                'Expected outputs come from fixtures, never from the baseline LLM.',
                'Input-token counts must include the whole request, including any codebook.',
                'Null token, cost, and timing fields are unknown; they are not treated as zero.',
                'Cost/latency fields for the compressed arm must include compression, retries and fallback.',
                'Observed correct-to-wrong transitions also contain generation noise; use repeated paired trials.',
                'This deterministic smoke suite is not a substitute for held-out book and production evaluations.'
            ]
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        with args.details.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(report['overall'], indent=2, allow_nan=False))
        print(f'Report: {args.out}\nScored examples: {args.details}')
        return 0
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
