"""Fixed four-question development diagnostic for source-preserving inheritance."""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from .evo_data import ROOT
from .evo_grammar import digest
from .evo_public_data import REPORTS,load_training,sample_training
from .evo_public_eval import SpanAnswer,prompt
from .evo_transcript_search import GRAMMAR,compile_inheritance,decode_inheritance
from .evo_usage import Ledger,Runner


def run():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT/'.env')
    datasets,_,sources=load_training()
    cases=sample_training(datasets['MeeQA'])[:4]
    config={'grammar':GRAMMAR,'case_ids':[c['id'] for c in cases],'sources':sources,
            'compiler_sha256':digest((ROOT/'tokenese/evo_transcript_search.py').read_text()),
            'prompt_hash':digest(prompt('CONTEXT','QUESTION')),'output_limit':2048,
            'purpose':'four-question development diagnostic, not qualification'}
    path=REPORTS/'inheritance-config.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Diagnostic config changed; do not overwrite the recorded experiment')
    path.write_text(json.dumps(config,indent=2)+'\n')
    runner=Runner(OpenAI(),Ledger(ROOT/'reports/evolution/ledger.sqlite'))
    def evaluate(case):
        encoded=compile_inheritance(case['context'])
        assert decode_inheritance(encoded.text,encoded.profile)==case['context']
        rows=[]
        for method,context in [('raw',case['context']),('encoded',encoded.text)]:
            result=runner.call(prompt(context,case['question']),SpanAnswer,purpose='baseline' if method=='raw' else 'probe',max_output_tokens=2048)
            rows.append({'id':case['id'],'dataset':case['dataset'],'meeting_id':case['meeting_id'],'family_id':case['family_id'],
                         'method':method,'profile':encoded.profile if method=='encoded' else 'raw','result':result})
        return rows
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[r for pair in pool.map(evaluate,cases) for r in pair]
    (REPORTS/'inheritance-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    return {'rows':len(rows),'ledger':runner.ledger.summary()}


def audit_diagnostic():
    from .evo_public_eval import aggregate,score
    from .evo_transcript_search import decode_quote
    from .evo_public_report import paired_interval
    datasets,_,_=load_training()
    cases={c['id']:c for c in datasets['MeeQA']}
    rows=json.loads((REPORTS/'inheritance-rows.json').read_text())
    inputs_reproduced = True
    for row in rows:
        case=cases[row['id']]
        current_encoding=compile_inheritance(case['context'])
        expected_context=current_encoding.text if row['method']=='encoded' else case['context']
        inputs_reproduced &= row['result']['request']['input']==prompt(expected_context,case['question'])
        evaluated=dict(row['result'])
        if row['method']=='encoded' and evaluated.get('answer'):
            encoded=compile_inheritance(case['context'])
            try:
                evaluated['answer']={**evaluated['answer'],'spans':[decode_quote(encoded.text,encoded.profile,s) for s in evaluated['answer']['spans']]}
            except ValueError as exc:
                evaluated['error']='Quote reconstruction: '+str(exc)
                row['quote_reconstruction_error']=str(exc)
        row['score']=score(case,evaluated,'raw')
    methods={m:aggregate([r for r in rows if r['method']==m]) for m in ('raw','encoded')}
    report={'study':'inheritance_four_question_development_diagnostic','qualified':False,'advance':False,
            'methods':methods,'recorded_inputs_reproduced':inputs_reproduced,'uncertainty':paired_interval(rows),
            'limitation':'Four development questions cannot satisfy the 32-question pilot advancement or release requirements.',
            'quote_reconstruction_errors':sum('quote_reconstruction_error' in r for r in rows)}
    (REPORTS/'inheritance-scored-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (REPORTS/'inheritance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    import sys
    if '--audit' in sys.argv:
        print(json.dumps(audit_diagnostic(),indent=2))
    elif '--run-api' in sys.argv:
        print(json.dumps(run(),indent=2))
        print(json.dumps(audit_diagnostic(),indent=2))
    else:
        raise SystemExit('Use --audit for saved rows, or --run-api for the fixed diagnostic under the shared ledger.')
