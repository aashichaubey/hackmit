"""Separate bounded-language comprehension from holistic source usefulness."""
from __future__ import annotations

import json
from collections import Counter

from .evo_data import DATA, REPORTS, write_manifest
from .evo_grammar import digest
from .evo_search import save


def language_questions(case: dict) -> list[dict]:
    """Questions derived from annotated fields, never answer-model outputs."""
    candidates = []
    facts = case['facts']
    for f in facts:
        matching = [o for o in facts if o['kind'] == f['kind'] and o['text'] == f['text']]
        owners = {o['person'] for o in matching}
        deadlines = {o['deadline'] for o in matching}
        subject = f"the {f['kind']} about “{f['text']}”"
        owner = next(iter(owners)) if len(owners) == 1 else ''
        due = next(iter(deadlines)) if len(deadlines) == 1 else ''
        # Named field queries do not pretend that this tests free-form summarization.
        candidates.append((f"Who is named for {subject}?", [owner] if owner else ['not found'], 'owner', f))
        candidates.append((f"What deadline is recorded for {subject}?", [due] if due else ['not found'], 'deadline', f))
        verbs = {'action':'an action', 'proposal':'a proposal', 'decision':'a decision', 'agreement':'an agreement', 'disagreement':'a disagreement'}
        candidates.append((f"Is {verbs[f['kind']]} about “{f['text']}” explicitly recorded?", ['yes'], f['kind'], f))
        if f['kind'] == 'proposal' and not any(o['kind'] == 'decision' and o['text'] == f['text'] for o in facts):
            candidates.append((f"Was “{f['text']}” decided?", ['not found'], 'proposal', f))
    unique = {}
    for question, answers, category, fact in candidates:
        unique.setdefault(question, (question, answers, category, fact))
    positive = [q for q in unique.values() if q[1] != ['not found']]
    negative = [q for q in unique.values() if q[1] == ['not found']]
    # Six supported answers + four absences when available; never invent fields.
    selected = positive[:6] + negative[:4]
    for candidate in list(unique.values()):
        if len(selected) >= 10:
            break
        if candidate not in selected:
            selected.append(candidate)
    if len(selected) != 10:
        raise ValueError(f"Insufficient annotated facts for ten honest field questions: {case['id']}")
    questions = []
    for i, (question, answers, category, fact) in enumerate(selected):
        quote = fact['source_quote']
        start = case['notes'].find(quote)
        if start < 0:
            raise ValueError('Source quote missing')
        questions.append({'id':f"{case['id']}-language-{i+1:02d}", 'question':question,'answers':answers,
            'expected_found':answers != ['not found'], 'category':category,
            'evidence':[{'start':start,'end':start+len(quote),'quote':quote}],
            'annotation_basis':'deterministic field query over provisional annotated facts'})
    return questions


def audit_cases(cases: list[dict]) -> dict:
    questions = [q for c in cases for q in c['questions']]
    return {'source_documents':len(cases), 'questions':len(questions),
            'categories':dict(Counter(q['category'] for q in questions)),
            'answerable':sum(q['expected_found'] for q in questions),
            'unknown':sum(not q['expected_found'] for q in questions),
            'independently_reviewed':sum(c['provenance']['annotation_status']=='independently_reviewed' for c in cases),
            'unique_source_series':len({c['provenance']['series_id'] for c in cases}),
            'warnings':['Cross-series splits test domain shift but confound split with organization.',
                        'Provisional model-authored facts can contain semantic annotation errors.',
                        'Field-preservation QA is narrower than arbitrary meeting questions.',
                        'Macro meeting accuracy and category metrics supplement exact micro accuracy.']}


def update_suite():
    if (REPORTS/'selection-lock.json').exists():
        raise ValueError('Cannot update labels after opening validation')
    audit = {}
    for split in ('dev','validation','test'):
        path = DATA/f'{split}.json'
        payload = json.loads(path.read_text())
        for case in payload['cases']:
            if 'holistic_questions' not in case:
                case['holistic_questions'] = case['questions']
            case['questions'] = language_questions(case)
        payload['evaluation_tracks'] = {'questions':'bounded-language field preservation',
            'holistic_questions':'provisional broader source questions; semantic labels require review'}
        save(path,payload)
        audit[split] = audit_cases(payload['cases'])
    write_manifest()
    save(REPORTS/'suite-audit.json',audit)


if __name__ == '__main__':
    update_suite()
