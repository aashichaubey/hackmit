"""Offline frozen-result audit and separately registered workspace-policy scoring."""
from __future__ import annotations
import hashlib
import json
from .evo_data import ROOT
from .evo_grammar import digest
from .evo_public_data import REPORTS
from .evo_public_holdout import load_heldout
from .evo_public_eval import SpanAnswer,prompt,score,aggregate
from .evo_public_report import paired_interval
from .evo_transcript import compile_minimal_transcript,decode_transcript


def run():
    from .evo_transcript_response import restore_response
    manifest=json.loads((REPORTS/'heldout-frozen.json').read_text())
    report=json.loads((REPORTS/'heldout.json').read_text())
    rows=json.loads((REPORTS/'heldout-rows.json').read_text())
    cases,provenance=load_heldout(manifest)
    indexed={c['id']:c for c in cases}
    checks={'frozen_files':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['files'].items()),
            'manifest':digest(manifest)==report['manifest_hash'],'source_provenance':provenance==report['provenance'],
            'pairs_complete':len(rows)==128 and len({(r['id'],r['method']) for r in rows})==128 and {r['id'] for r in rows}==set(indexed)}
    reconstructed=True;requests_match=True;scores_match=True;corrected=[];changes=[]
    for row in rows:
        case=indexed[row['id']];encoding=compile_minimal_transcript(case['context'])
        profile=encoding.profile if row['method']=='encoded' else 'raw'
        context=encoding.text if row['method']=='encoded' else case['context']
        request=row['result']['request']
        reconstructed &= decode_transcript(encoding.text,encoding.profile)==case['context']
        requests_match &= (row['profile']==profile and request['input']==prompt(context,case['question']) and
                           request['schema']==SpanAnswer.model_json_schema() and request['model']==manifest['model'] and
                           request['max_output_tokens']==manifest['max_output_tokens'] and request['temperature']==manifest['temperature'])
        scores_match &= score(case,row['result'],profile)==row['score']
        restored=restore_response(case['context'],encoding,row['result'],'minimal' if row['method']=='encoded' else 'raw')
        corrected_score=score(case,restored,'raw')
        corrected.append({**row,'score':corrected_score,'product_error':restored.get('error')})
        if corrected_score!=row['score']:
            changes.append({'id':row['id'],'method':row['method'],'original_f1':row['score']['f1'],'product_f1':corrected_score['f1'],'product_error':restored.get('error')})
    original={m:aggregate([r for r in rows if r['method']==m]) for m in ('raw','encoded')}
    checks.update(reconstruction=reconstructed,requests=requests_match,native_scores=scores_match,aggregates=original==report['methods'])
    methods={m:aggregate([r for r in corrected if r['method']==m]) for m in ('raw','encoded')}
    raw,encoded=methods['raw'],methods['encoded']
    uncertainty=paired_interval(corrected)
    uncertainty['interpretation']='Paired meeting-family test uncertainty; not proof of equivalence or a general-purpose accuracy guarantee.'
    product={'study':'same_frozen_test_workspace_grounding_policy','methods':methods,
             'amendment':json.loads((REPORTS/'heldout-scoring-amendment.json').read_text()),
             'policy_sha256':hashlib.sha256((ROOT/'tokenese/evo_transcript_response.py').read_bytes()).hexdigest(),
             'score_changes':changes,'uncertainty':uncertainty,'no_new_model_calls':True,
             'input_savings_fraction':report['input_savings_fraction'],'cost_savings_fraction':report['cost_savings_fraction'],
             'f1_loss_percentage_points':100*(raw['macro_f1']-encoded['macro_f1']),
             'answerability_loss_percentage_points':100*(raw['balanced_answerability']-encoded['balanced_answerability']),
             'total_token_savings_fraction':1-(encoded['actual_input_tokens']+encoded['actual_output_tokens'])/(raw['actual_input_tokens']+raw['actual_output_tokens'])}
    product['within_exploratory_tolerance']=product['f1_loss_percentage_points']<=5 and product['answerability_loss_percentage_points']<=5
    audit={'checks':checks,'all_reproduced':all(checks.values()),'families':len(provenance['selected_families']),
           'questions':len(cases),'product_policy_changed_rows':len(changes),'no_model_calls':True}
    (REPORTS/'heldout-product-policy.json').write_text(json.dumps(product,indent=2)+'\n')
    (REPORTS/'heldout-product-policy-rows.json').write_text(json.dumps(corrected,indent=2)+'\n')
    (REPORTS/'heldout-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return {'audit':audit,'product':product}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
