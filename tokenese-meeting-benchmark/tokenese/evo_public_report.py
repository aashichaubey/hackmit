"""Offline paired uncertainty and integrity audit for native public QA studies."""
from __future__ import annotations
import json
import random
from collections import defaultdict
from statistics import mean
from .evo_public_data import REPORTS, load_training
from .evo_public_eval import SpanAnswer, aggregate, prompt, score
from .evo_transcript import compile_transcript, compile_readable_transcript, compile_minimal_transcript, decode_transcript


def paired_interval(rows: list[dict], repetitions=5000, seed=20260920) -> dict:
    pairs=defaultdict(dict)
    families=defaultdict(list)
    for row in rows:
        pairs[row['id']][row['method']]=row
    for pair in pairs.values():
        if set(pair)!={'raw','encoded'}:
            raise ValueError('Incomplete paired evaluation')
        families[pair['raw']['family_id']].append(pair['encoded']['score']['f1']-pair['raw']['score']['f1'])
    clusters=list(families.values());rng=random.Random(seed)
    samples=sorted(mean(value for cluster in rng.choices(clusters,k=len(clusters)) for value in cluster) for _ in range(repetitions))
    deltas=[d for cluster in clusters for d in cluster]
    return {'unit':'meeting family, paired cluster bootstrap','families':len(clusters),'repetitions':repetitions,'seed':seed,
            'mean_question_f1_delta':mean(deltas),'percentile_95_interval':[samples[int(.025*repetitions)],samples[int(.975*repetitions)]],
            'improved':sum(d>0 for d in deltas),'regressed':sum(d<0 for d in deltas),'tied':sum(d==0 for d in deltas),
            'interpretation':'Exploratory training uncertainty, not proof of equivalence or a qualification gate.'}


def run() -> dict:
    datasets,excluded,sources=load_training()
    cases={c['id']:c for dataset in datasets.values() for c in dataset}
    studies={}
    for name,compiler in [('pilot',compile_transcript),('pilot-readable',compile_readable_transcript),('pilot-minimal',compile_minimal_transcript)]:
        path=REPORTS/(name+'-rows.json')
        if not path.exists():
            continue
        rows=json.loads(path.read_text());report=json.loads((REPORTS/(name+'.json')).read_text())
        if report['sources']!=sources:
            raise ValueError('Published input file hashes changed')
        checks=[]
        for row in rows:
            case=cases[row['id']]
            checks.append(score(case,row['result'],row['profile'])==row['score'])
            encoding=compiler(case['context'])
            checks.append(decode_transcript(encoding.text,encoding.profile)==case['context'])
            expected_profile=encoding.profile if row['method']=='encoded' else 'raw'
            expected_context=encoding.text if row['method']=='encoded' else case['context']
            request=row['result']['request']
            checks.append(row['profile']==expected_profile)
            checks.append(request['input']==prompt(expected_context,case['question']))
            checks.append(request['schema']==SpanAnswer.model_json_schema())
            checks.append(request['max_output_tokens']==report['max_output_tokens'])
        methods={m:aggregate([r for r in rows if r['method']==m]) for m in ('raw','encoded')}
        checks.append(methods==report['methods'])
        raw,encoded=methods['raw'],methods['encoded']
        studies[name]={'reproduced':all(checks),'uncertainty':paired_interval(rows),
                       'actual_input_reduction':1-encoded['actual_input_tokens']/raw['actual_input_tokens'] if raw['actual_input_tokens'] is not None and encoded['actual_input_tokens'] is not None else None,
                       'actual_cost_reduction':1-encoded['actual_usd']/raw['actual_usd'] if raw['actual_usd'] is not None and encoded['actual_usd'] is not None else None,
                       'advance':report['advance']}
    result={'studies':studies,'all_reproduced':all(s['reproduced'] for s in studies.values()),
            'heldout_opened':False,'sources':sources,'exclusions':len(excluded)}
    (REPORTS/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
