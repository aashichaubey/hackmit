"""Explicitly budgeted block-grammar successor; quality loss remains a metric."""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from .evo_data import ROOT
from .evo_grammar import digest
from .evo_public_data import REPORTS,load_training,sample_training
from .evo_public_eval import SpanAnswer,prompt,score,aggregate
from .evo_public_report import paired_interval
from .evo_transcript_blocks import GRAMMAR,compile_blocks,decode_blocks,decode_blocks_quote
from .evo_usage import Ledger,Runner

ALLOCATION_REASON='User-directed public QA exploration; allocate existing global capacity, retain paired final-evaluation reserve.'


def restore_answer(source, encoding, result):
    restored=dict(result)
    if not result.get('answer') or result.get('error'):
        return restored
    spans=[]
    try:
        for span in result['answer']['spans']:
            # A correct source quotation is already grounded even if the model
            # expanded a compact label itself. No reference answer is consulted.
            if span in source:
                spans.append(span)
            else:
                spans.append(decode_blocks_quote(encoding.text,encoding.profile,span))
        restored['answer']={**result['answer'],'spans':spans}
    except ValueError as exc:
        restored['error']='Quote restoration: '+str(exc)
    return restored


def run():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT/'.env')
    datasets,_,sources=load_training()
    cases=[c for name in ('meetingqa','MeeQA') for c in sample_training(datasets[name])]
    config={'grammar':GRAMMAR,'sources':sources,'case_ids':[c['id'] for c in cases],
            'prompt_hash':digest(prompt('CONTEXT','QUESTION')),'output_limit':2048,
            'quality_policy':'Report quality loss and savings; no zero-loss qualification claim.'}
    path=REPORTS/'blocks-config.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Recorded configuration changed')
    path.write_text(json.dumps(config,indent=2)+'\n')
    ledger=Ledger(ROOT/'reports/evolution/ledger.sqlite')
    purpose=ledger.allocate_successor_discovery('publicqa-blocks',48,128,ALLOCATION_REASON)
    runner=Runner(OpenAI(),ledger)
    def evaluate(case,method):
        encoding=compile_blocks(case['context'])
        assert decode_blocks(encoding.text,encoding.profile)==case['context']
        context=encoding.text if method=='encoded' else case['context']
        result=runner.call(prompt(context,case['question']),SpanAnswer,purpose=purpose if method=='encoded' else 'baseline',max_output_tokens=2048,leave_calls=128)
        restored=restore_answer(case['context'],encoding,result) if method=='encoded' else result
        row={'id':case['id'],'dataset':case['dataset'],'meeting_id':case['meeting_id'],'family_id':case['family_id'],
             'method':method,'profile':encoding.profile if method=='encoded' else 'raw',
             'score':score(case,restored),'result':result,'restoration_error':restored.get('error')}
        print(json.dumps({'id':case['id'],'method':method,'f1':row['score']['f1']}),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(lambda c:evaluate(c,'raw'),cases))
        rows+=list(pool.map(lambda c:evaluate(c,'encoded'),cases))
    methods={m:aggregate([r for r in rows if r['method']==m]) for m in ('raw','encoded')}
    raw,encoded=methods['raw'],methods['encoded']
    report={**config,'study':'block_grammar_training_tradeoff','qualified':False,'methods':methods,
            'input_savings_fraction':1-encoded['actual_input_tokens']/raw['actual_input_tokens'] if encoded['actual_input_tokens'] and raw['actual_input_tokens'] else None,
            'f1_loss_percentage_points':100*(raw['macro_f1']-encoded['macro_f1']),
            'answerability_loss_percentage_points':100*(raw['balanced_answerability']-encoded['balanced_answerability']),
            'uncertainty':paired_interval(rows),'ledger':ledger.summary()}
    (REPORTS/'blocks-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (REPORTS/'blocks.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    import sys
    if '--run-api' not in sys.argv:
        raise SystemExit('Pass --run-api to run this explicitly allocated successor pilot.')
    print(json.dumps(run(),indent=2))
