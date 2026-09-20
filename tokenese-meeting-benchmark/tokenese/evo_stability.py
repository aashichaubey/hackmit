"""Budgeted repeated calls for selected development failure attribution only."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .evo_data import REPORTS
from .evo_eval import score_answer, experiment_identity
from .evo_search import save
from .evo_usage import Ledger, Runner
from .facts import Answer


def run_stability(runner: Runner) -> dict:
    source = json.loads((REPORTS/'source-grammar-dev.json').read_text())
    if source['loaded_splits'] != ['dev'] or any(r['split'] != 'dev' for r in source['raw_rows']+source['encoded_rows']):
        raise ValueError('Stability diagnosis may read development results only')
    question_ids = source['paired']['new_errors_vs_raw']
    if len(question_ids) > 4:
        raise ValueError('This diagnostic is bounded to four development failures')
    rows = []
    # Alternate which representation runs first. This is not an efficacy test.
    for replicate in range(1,6):
        methods = ['raw','encoded'] if replicate % 2 else ['encoded','raw']
        for qid in question_ids:
            for method in methods:
                original = next(r for r in source[method+'_rows'] if r['question_id'] == qid)
                result = runner.call(original['request']['input'],Answer,purpose='probe',replicate=replicate)
                rows.append({'question_id':qid,'method':method,'replicate':replicate,
                             'correct':not result.get('error') and score_answer(result.get('answer'),original),**result})
    summary = []
    for qid in question_ids:
        for method in ('raw','encoded'):
            selected = [r for r in rows if r['question_id']==qid and r['method']==method]
            answers = Counter(json.dumps(r['answer'],sort_keys=True,ensure_ascii=False) for r in selected)
            summary.append({'question_id':qid,'method':method,'correct':sum(bool(r['correct']) for r in selected),
                            'replicates':len(selected),'distinct_outputs':len(answers),'output_counts':dict(answers)})
    report = {'status':'development_failure_diagnostic','loaded_splits':['dev'],'identity':experiment_identity(),
              'question_ids':question_ids,'rows':rows,'summary':summary,'ledger':runner.ledger.summary(),
              'caveat':'Selected failures, five repeats each. Not a representative benchmark or independent significance test. Original pilot is unchanged; best-of-five is not used.'}
    save(REPORTS/'source-grammar-stability.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-api',action='store_true',required=True)
    parser.parse_args()
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    report = run_stability(Runner(OpenAI(),Ledger(REPORTS/'ledger.sqlite')))
    print(json.dumps(report['summary'],indent=2,ensure_ascii=False))
