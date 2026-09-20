"""Development-only measurement of complete-source structural grammar."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .evo_benchmark import workflow_comparison
from .evo_classifier_next import paired_diagnostics
from .evo_data import DATA, REPORTS, load_cases
from .evo_eval import evaluate, summarize, experiment_identity
from .evo_grammar import digest
from .evo_search import save
from .evo_source import GRAMMAR_ID, RULES, compile_source
from .evo_usage import Ledger, Runner


def run_study(runner: Runner | None = None) -> dict:
    cases = [{**c,'questions':c['holistic_questions']} for c in load_cases(DATA/'dev.json','dev')]
    encodings = {c['id']:compile_source(c['notes']) for c in cases}
    report = {'status':'local_only' if runner is None else 'development_only_unqualified',
              'loaded_splits':['dev'],'grammar_id':GRAMMAR_ID,'rules':[asdict(r) for r in RULES],
              'identity':{k:v for k,v in experiment_identity().items() if not k.startswith('extraction_')},
              'compiler_hash':digest(Path(__file__).with_name('evo_source.py').read_text()),
              'dataset_hash':digest(cases),'encodings':{k:asdict(v) for k,v in encodings.items()},
              'source_context_tokens':sum(v.source_tokens for v in encodings.values()),
              'encoded_context_tokens':sum(v.encoded_tokens for v in encodings.values()),
              'caveat':'Exact source reversibility does not prove model comprehension. Adaptive development with provisional labels; not independent qualification.'}
    if runner:
        raw = evaluate(None,cases,runner,method='raw',purpose='baseline')
        encoded = evaluate(None,cases,runner,method='raw',purpose='probe',
                           context_overrides={k:v.text for k,v in encodings.items()})
        for row in raw:
            row.update(fact_source='source',grammar_id=None)
        for row in encoded:
            row.update(method='source_grammar',fact_source='source',grammar_id=GRAMMAR_ID,
                       route=encodings[row['case_id']].route)
        # This compiler makes no extraction call: zero is known, not missing usage.
        extraction = {c['id']:{'result':{'usage':{'input_tokens':0,'output_tokens':0,'cached_input_tokens':0}}} for c in cases}
        report.update(raw_rows=raw,encoded_rows=encoded,raw_summary=summarize(raw),
                      encoded_summary=summarize(encoded),paired=paired_diagnostics(raw,encoded),
                      workflows=workflow_comparison(cases,raw,encoded,extraction),ledger=runner.ledger.summary())
        report['pilot_pass'] = report['paired']['no_new_errors'] and all(
            v['accounting_status'] == 'complete' and v['input_savings_fraction'] > 0
            for v in report['workflows'].values())
    save(REPORTS/('source-grammar-dev.json' if runner else 'source-grammar-local.json'),report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-api',action='store_true')
    args = parser.parse_args()
    runner = None
    if args.run_api:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        runner = Runner(OpenAI(),Ledger(REPORTS/'ledger.sqlite'))
    report = run_study(runner)
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw_rows','encoded_rows','encodings','raw_summary','encoded_summary')},indent=2))
