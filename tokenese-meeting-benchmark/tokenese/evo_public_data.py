"""Pinned public QA adapters. Training inspection never opens test labels."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from .evo_data import ROOT
from .evo_grammar import digest

DATA = ROOT/'data'/'publicqa'
SOURCES = DATA/'sources'
REPORTS = ROOT/'reports'/'publicqa'
REPOSITORIES = {
    'meetingqa':('adobe-research/meetingqa','5e3d1fbf4fefb60790d2445ce6721085b274024b'),
    'MeeQA':('reutapel/MeeQA','7b9a6c43eef4e0830336862ffc19b48ff52615a6'),
    'MISeD':('google-research-datasets/MISeD','870d1aaf1063193d508e0224b86e011f921fd262'),
}


def meeting_family(name: str) -> str:
    lower = name.strip().lower()
    if re.fullmatch(r'(?:es|is|ts|en)\d{4}[a-z]',lower):
        return 'ami:'+lower[:-1]
    if re.fullmatch(r'(?:bmr|bed|bro|buw)\d+',lower):
        return 'icsi:'+lower
    return 'meeting:'+lower


def validate_spans(row: dict) -> list[dict]:
    if type(row['is_impossible']) is not bool:
        raise ValueError('invalid_answerability_type')
    texts,starts = row['answers']['text'],row['answers']['answer_start']
    if len(texts) != len(starts):
        raise ValueError('unequal_span_arrays')
    if row['is_impossible']:
        if any(texts):
            raise ValueError('unanswerable_with_nonempty_answer')
        return []
    spans = []
    for text,start in zip(texts,starts):
        if not text or type(start) is not int or start < 0 or row['context'][start:start+len(text)] != text:
            raise ValueError('invalid_source_offset')
        spans.append({'start':start,'end':start+len(text),'text':text})
    if not spans:
        raise ValueError('answerable_without_spans')
    return sorted(spans,key=lambda s:s['start'])


def adapt_extractive(rows: list[dict], dataset: str, split: str, meeting_names=()) -> tuple[list[dict],list[dict]]:
    names = sorted(meeting_names,key=len,reverse=True)
    grouped,excluded = {},[]
    for row in rows:
        meeting = row.get('title') if dataset == 'meetingqa' else next((n for n in names if row['id'].startswith(n)),None)
        if not meeting:
            excluded.append({'id':row['id'],'reason':'unmapped_meeting'})
            continue
        try:
            spans = validate_spans(row)
        except ValueError as exc:
            excluded.append({'id':row['id'],'reason':str(exc)})
            continue
        key = digest([meeting,row['context'],row['question']])
        item = grouped.setdefault(key,{'id':dataset+':'+key[:20],'dataset':dataset,'split':split,
            'meeting_id':meeting,'family_id':meeting_family(meeting),'context':row['context'],'question':row['question'],
            'expected_found':not row['is_impossible'],'references':[],'upstream_ids':[],'question_flags':[],
            'annotation_status':'published_human_extractive','disagreement':False})
        item['disagreement'] |= item['expected_found'] != (not row['is_impossible'])
        if spans not in item['references']:
            item['references'].append(spans)
        item['upstream_ids'].append(row['id'])
        item['question_flags'].append({k:v for k,v in row.items() if k.startswith('isQuestion')})
    cases = []
    for case in grouped.values():
        if case.pop('disagreement'):
            excluded.append({'id':case['id'],'reason':'annotator_answerability_disagreement','upstream_ids':case['upstream_ids']})
        else:
            cases.append(case)
    return cases,excluded


def load_training() -> tuple[dict,list[dict],dict]:
    paths = {'meetingqa':SOURCES/'meetingqa/AllData/Dataset/final-AMI-train.json',
             'MeeQA':SOURCES/'MeeQA/Data/original/train_data.json.zip',
             'MISeD':SOURCES/'MISeD/mised/train.jsonl'}
    adobe = json.loads(paths['meetingqa'].read_text())['data']
    with zipfile.ZipFile(paths['MeeQA']) as archive:
        meeqa = json.loads(archive.read('train_data.json'))['data']
    names = [r['meeting_name'] for r in csv.DictReader((SOURCES/'MeeQA/Data/train_meetings.csv').open())]
    datasets, exclusions = {},[]
    for dataset,rows in [('meetingqa',adobe),('MeeQA',meeqa)]:
        datasets[dataset], rejected = adapt_extractive(rows,dataset,'train',names)
        exclusions += [{'dataset':dataset,**r} for r in rejected]
    dialogues = [json.loads(line) for line in paths['MISeD'].read_text().splitlines() if line.strip()]
    datasets['MISeD'] = []
    for dialog in dialogues:
        meeting = dialog['meeting']
        segments = meeting['transcriptSegments']
        history = []
        for index,turn in enumerate(dialog['dialog']['dialogTurns']):
            ranges = turn.get('responseAttribution',{}).get('indexRanges',[])
            if any(type(r.get('startIndex')) is not int or type(r.get('endIndex')) is not int or
                   not 0 <= r['startIndex'] <= r['endIndex'] < len(segments) for r in ranges):
                exclusions.append({'dataset':'MISeD','id':f"{dialog['dialogId']}:{index}",'reason':'invalid_attribution_range'})
            else:
                datasets['MISeD'].append({'id':f"MISeD:{dialog['dialogId']}:{index}",'dataset':'MISeD','split':'train',
                    'meeting_id':meeting['meetingId'],'family_id':meeting_family(meeting['meetingId']),
                    'segments':segments,'question':turn['query'],'reference_response':turn['response'],
                    'history':list(history),'attribution_ranges':ranges,'query_metadata':turn.get('queryMetadata',{}),
                    'annotation_status':'published_human_verified_semiautomatic'})
            history.append({'query':turn['query'],'response':turn['response']})
    metadata = {dataset:{'repository':REPOSITORIES[dataset][0],'commit':REPOSITORIES[dataset][1],
                          'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'path':str(path.relative_to(ROOT))}
                for dataset,path in paths.items()}
    return datasets,exclusions,metadata


def sample_training(cases: list[dict], per_class=8) -> list[dict]:
    selected,meetings = [],Counter()
    # Round-robin labels avoids allowing the first stratum to exhaust all meetings.
    pools = {label:iter(sorted([c for c in cases if c['expected_found'] == label],key=lambda c:digest(c['id']))) for label in (True,False)}
    for _ in range(per_class):
        for label in (True,False):
            for case in pools[label]:
                if meetings[case['meeting_id']] < 2:
                    selected.append(case);meetings[case['meeting_id']] += 1
                    break
            else:
                raise ValueError('Insufficient independently grouped cases for the declared sample')
    return selected


def require_disjoint(development: list[dict], heldout: list[dict]) -> None:
    if {c['family_id'] for c in development} & {c['family_id'] for c in heldout}:
        raise ValueError('Meeting-family leakage across datasets/splits')


def write_catalog() -> dict:
    datasets,excluded,metadata = load_training()
    families = {k:{c['family_id'] for c in v} for k,v in datasets.items()}
    result = {'loaded_splits':['train'],'sources':metadata,
              'counts':{k:{'questions':len(v),'meeting_families':len(families[k])} for k,v in datasets.items()},
              'cross_dataset_family_overlap':{a+' / '+b:len(families[a]&families[b]) for a in families for b in families if a < b},
              'exclusion_counts':dict(Counter(r['dataset']+':'+r['reason'] for r in excluded)),
              'pilot_ids':{k:[c['id'] for c in sample_training(v)] for k,v in datasets.items() if k != 'MISeD'}}
    REPORTS.mkdir(parents=True,exist_ok=True)
    (REPORTS/'catalog.json').write_text(json.dumps(result,indent=2)+'\n')
    (REPORTS/'excluded-rows.json').write_text(json.dumps(excluded,indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(write_catalog(),indent=2))
