"""Offline annotation review candidates, without grading or changing labels."""
from __future__ import annotations

import json
import re
from collections import Counter

from .evo_data import DATA, REPORTS, load_cases
from .evo_grammar import digest
from .evo_search import save


def audit_case(case: dict) -> list[dict]:
    issues = []
    def add(key, category, original, reason, suggestion=None):
        issues.append({'id':case['id']+':'+key,'case_id':case['id'],'category':category,
                       'original':original,'reason':reason,'suggestion_for_review':suggestion,
                       'review':{'status':'pending','reviewer':None,'decision':None,'rationale':None}})
    notes = case['notes']
    for index, fact in enumerate(case['facts']):
        deadline = fact.get('deadline','')
        if deadline and deadline not in fact['source_quote'] and re.search(r'^#+\s*'+re.escape(deadline)+r'\s*$',notes,re.M):
            add(f'fact-{index}:deadline','heading_as_deadline',fact,
                'The deadline matches a section heading but is absent from the quoted fact. Check whether it is a meeting date rather than a task deadline.')
    for track in ('questions','holistic_questions'):
        seen = set()
        for q in case.get(track,[]):
            key = f'{track}:{q["id"]}'
            if q['id'] in seen:
                add(key+':duplicate','duplicate_question',q,'Question ID repeats within this track.')
            seen.add(q['id'])
            valid_spans = all(isinstance(s.get('start'),int) and isinstance(s.get('end'),int)
                              and 0 <= s['start'] < s['end'] <= len(notes)
                              and notes[s['start']:s['end']] == s.get('quote') for s in q.get('evidence',[]))
            if not q.get('evidence') or not valid_spans:
                add(key+':evidence','invalid_evidence',q,'Evidence is missing or does not match original character offsets.')
            found = q.get('expected_found')
            if type(found) is not bool or not q.get('answers') or ((not found) != (q['answers'] == ['not found'])):
                add(key+':found','inconsistent_found',q,'Expected found and accepted answers are inconsistent.')
            if found and re.match(r'^(?:Is|Are|Was|Were|Did|Does|Do|Has|Have|Had|Will|Would|Can|Could|Should)\b',q['question']) and not re.search(r'\bor\b',q['question'],re.I):
                forms = {a.strip().casefold().rstrip('.') for a in q['answers']}
                if not forms & {'yes','no'}:
                    add(key+':yesno','yesno_answer_missing',q,'The fixed instruction requires yes/no for explicit yes/no questions, but no such answer is accepted. Check question intent before editing.')
                elif forms - {'yes','no'}:
                    add(key+':yesno-extra','yesno_extra_variant',q,'A yes/no answer is accepted, but additional variants permit output outside the fixed format. This does not prevent a compliant answer from passing.')
            if found and re.match(r'^Who\b',q['question']):
                candidates = sorted({a.split(',',1)[0].strip() for a in q['answers'] if ',' in a
                                     and a.split(',',1)[0].strip() in notes} - set(q['answers']))
                if candidates:
                    add(key+':alias','possible_missing_name_variant',q,
                        'A shorter name occurs in the source but is absent from accepted answers. Check that it uniquely identifies the same person.',candidates)
    return issues


def build_packet(cases: list[dict]) -> dict:
    if any(c.get('split') != 'dev' for c in cases):
        raise ValueError('This review packet is development-only')
    issues = [issue for case in cases for issue in audit_case(case)]
    # A concise first review: one example of each category, then at most six total.
    selected, categories = [], set()
    for issue in issues:
        if issue['category'] == 'yesno_extra_variant':
            continue  # Keep lower-priority format leniency in the complete queue.
        if issue['category'] not in categories:
            selected.append(issue['id'])
            categories.add(issue['category'])
    selected = selected[:6]
    return {'status':'pending_independent_review','loaded_splits':['dev'],'dataset_hash':digest(cases),
            'counts':dict(Counter(i['category'] for i in issues)),'issues':issues,'selected_issue_ids':selected,
            'sources':{c['id']:{'notes':c['notes'],'provenance':c['provenance'],'notes_hash':digest(c['notes'])} for c in cases},
            'caveat':'These are mechanical review candidates, not confirmed annotation errors. No label changes, regrading, semantic certification or qualification occurs.'}


def markdown_packet(packet: dict) -> str:
    lines = ['# Annotation review packet — development only','',
             'Please assess the draft labels against the source, without referring to model answers. This is a selected diagnostic sample, not an independent qualification corpus. Full excerpts are included because absence/unknown labels require checking the whole source.','',
             'For each item, record: reviewer identity, accept/correct/uncertain, corrected label if needed, and source-based rationale. Changes belong in a new dataset version; original frozen results must remain intact.','']
    source_ids = set()
    for issue in packet['issues']:
        if issue['id'] not in packet['selected_issue_ids']:
            continue
        source_ids.add(issue['case_id'])
        lines += ['## '+issue['id'],'',issue['reason'],'','Draft annotation:','```json',
                  json.dumps(issue['original'],indent=2,ensure_ascii=False),'```','']
        if issue['suggestion_for_review']:
            lines += ['Candidate variant(s), **not accepted automatically**: '+', '.join(issue['suggestion_for_review']), '']
        lines += ['Reviewer: ___  Decision: ___  Corrected label: ___','', 'Rationale: ___','']
    for case_id in sorted(source_ids):
        source = packet['sources'][case_id]
        lines += ['## Full source: '+case_id,'',f"[Public source]({source['provenance']['url']})",'',source['notes'],'']
    return '\n'.join(lines)


if __name__ == '__main__':
    packet = build_packet(load_cases(DATA/'dev.json','dev'))
    save(REPORTS/'label-review.json',packet)
    (REPORTS/'label-review.md').write_text(markdown_packet(packet))
    print(json.dumps({'counts':packet['counts'],'selected':packet['selected_issue_ids']},indent=2))
