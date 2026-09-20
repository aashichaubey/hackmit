"""One frozen, family-disjoint MeeQA test measurement; never used for tuning."""
from __future__ import annotations
import csv
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from . import MODEL
from .evo_data import ROOT
from .evo_grammar import digest
from .evo_public_data import REPORTS,SOURCES,REPOSITORIES,adapt_extractive,load_training,require_disjoint
from .evo_public_eval import SpanAnswer,prompt,score,aggregate
from .evo_public_report import paired_interval
from .evo_transcript import compile_minimal_transcript,decode_transcript
from .evo_usage import Ledger,Runner

FILES=('tokenese/evo_transcript.py','tokenese/evo_public_eval.py','tokenese/evo_public_data.py','tokenese/evo_public_holdout.py')


def freeze():
    datasets,_,sources=load_training()
    manifest={'method':'minimal_separator','model':MODEL,'max_output_tokens':2048,'temperature':0,
              'family_exclusions':sorted({c['family_id'] for data in datasets.values() for c in data}),
              'training_sources':sources,'test_repository':REPOSITORIES['MeeQA'],
              'selection_reason':'Measured modest F1 loss, fewer quotation failures and lower total cost than blocks on development; user permits slight quality loss.',
              'sampling':{'questions':64,'answerable':32,'unanswerable':32,'max_per_meeting':4,'ordering':'hash(id), label round robin'},
              'interpretation':'Exploratory independent tradeoff measurement, not a claim of equal accuracy.',
              'review_thresholds':{'f1_loss_percentage_points':5,'balanced_answerability_loss_percentage_points':5},
              'files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES},
              'prompt_hash':digest(prompt('CONTEXT','QUESTION')),'schema':SpanAnswer.model_json_schema()}
    path=REPORTS/'heldout-frozen.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:
            raise ValueError('Frozen successor changed; do not adapt to held-out results')
    else:
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def load_heldout(manifest):
    repository,commit=REPOSITORIES['MeeQA']
    paths={}
    for relative in ('Data/original/test_data.json.zip','Data/test_meetings.csv'):
        target=SOURCES/'MeeQA'/relative
        if not target.exists():
            request=urllib.request.Request(f'https://raw.githubusercontent.com/{repository}/{commit}/{relative}',headers={'User-Agent':'Tokenese-research'})
            with urllib.request.urlopen(request,timeout=90) as response:
                body=response.read()
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(body)
        paths[relative]=target
    with zipfile.ZipFile(paths['Data/original/test_data.json.zip']) as archive:
        rows=json.loads(archive.read('test_data.json'))['data']
    with paths['Data/test_meetings.csv'].open() as stream:
        names=[r['meeting_name'] for r in csv.DictReader(stream)]
    cases,excluded=adapt_extractive(rows,'MeeQA','test',names)
    retained=[c for c in cases if c['family_id'] not in manifest['family_exclusions']]
    selected=[];meetings=Counter()
    pools={label:iter(sorted((c for c in retained if c['expected_found']==label),key=lambda c:digest(c['id']))) for label in (True,False)}
    for _ in range(32):
        for label in (True,False):
            for case in pools[label]:
                if meetings[case['meeting_id']]<4:
                    selected.append(case);meetings[case['meeting_id']]+=1;break
            else:
                raise ValueError('Insufficient independent test cases for the frozen sample; do not silently weaken sampling')
    require_disjoint([{'family_id':f} for f in manifest['family_exclusions']],selected)
    provenance={'source_sha256':{r:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in paths.items()},
                'adapted_questions':len(cases),'excluded_annotation_records':len(excluded),
                'family_disjoint_questions':len(retained),'family_overlap_questions_removed':len(cases)-len(retained),
                'selected_ids':[c['id'] for c in selected],'selected_families':sorted({c['family_id'] for c in selected})}
    return selected,provenance


def run():
    from dotenv import load_dotenv
    from openai import OpenAI
    manifest=freeze()  # Written/checked before opening any held-out labels.
    cases,provenance=load_heldout(manifest)
    (REPORTS/'heldout-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    load_dotenv(ROOT/'.env')
    runner=Runner(OpenAI(),Ledger(ROOT/'reports/evolution/ledger.sqlite'))
    def evaluate(case,method):
        encoding=compile_minimal_transcript(case['context'])
        assert decode_transcript(encoding.text,encoding.profile)==case['context']
        context=encoding.text if method=='encoded' else case['context']
        result=runner.call(prompt(context,case['question']),SpanAnswer,purpose='qualification',max_output_tokens=2048)
        row={'id':case['id'],'dataset':case['dataset'],'meeting_id':case['meeting_id'],'family_id':case['family_id'],
             'method':method,'profile':encoding.profile if method=='encoded' else 'raw',
             'score':score(case,result,encoding.profile if method=='encoded' else 'raw'),'result':result}
        print(json.dumps({'id':case['id'],'method':method,'f1':row['score']['f1'],'error':result['error']}),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(lambda c:evaluate(c,'raw'),cases))
        rows+=list(pool.map(lambda c:evaluate(c,'encoded'),cases))
    methods={m:aggregate([r for r in rows if r['method']==m]) for m in ('raw','encoded')}
    raw,encoded=methods['raw'],methods['encoded']
    report={'study':'frozen_family_disjoint_MeeQA_test','manifest_hash':digest(manifest),'provenance':provenance,
            'methods':methods,'uncertainty':paired_interval(rows),
            'input_savings_fraction':1-encoded['actual_input_tokens']/raw['actual_input_tokens'] if encoded['actual_input_tokens'] and raw['actual_input_tokens'] else None,
            'cost_savings_fraction':1-encoded['actual_usd']/raw['actual_usd'] if encoded['actual_usd'] and raw['actual_usd'] else None,
            'f1_loss_percentage_points':100*(raw['macro_f1']-encoded['macro_f1']),
            'answerability_loss_percentage_points':100*(raw['balanced_answerability']-encoded['balanced_answerability']),
            'equal_quality_claim':False,'ledger':runner.ledger.summary()}
    report['within_exploratory_tolerance']=report['f1_loss_percentage_points']<=5 and report['answerability_loss_percentage_points']<=5
    (REPORTS/'heldout-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (REPORTS/'heldout.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    import sys
    if '--run-api' not in sys.argv:
        raise SystemExit('Pass --run-api to freeze and run the one-shot independent evaluation.')
    print(json.dumps(run(),indent=2))
