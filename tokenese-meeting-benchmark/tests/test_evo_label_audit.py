from copy import deepcopy
import pytest

from tokenese.evo_label_audit import audit_case,build_packet


def example():
    notes = '## 2026-09-20\nAna, the lead, will review. The team discussed hiring.'
    return {'id':'dev-example','split':'dev','notes':notes,'provenance':{'url':'https://example.org'},
            'facts':[{'kind':'action','person':'Ana','text':'review','deadline':'2026-09-20','source_quote':'Ana, the lead, will review.'}],
            'questions':[], 'holistic_questions':[{'id':'q1','question':'Who will review?','answers':['Ana, the lead'],
            'expected_found':True,'evidence':[{'start':13,'end':39,'quote':notes[13:39]}]}]}


def test_audit_flags_review_candidates_without_mutating_or_accepting_labels():
    case = example()
    original = deepcopy(case)
    packet = build_packet([case])
    assert case == original
    assert packet['counts']['heading_as_deadline'] == 1
    assert packet['counts']['possible_missing_name_variant'] == 1
    assert all(i['review']['status'] == 'pending' for i in packet['issues'])
    assert packet['status'] == 'pending_independent_review'
    case['split'] = 'test'
    with pytest.raises(ValueError):
        build_packet([case])


def test_audit_validates_broader_track_evidence_and_yesno_format():
    case = example()
    question = case['holistic_questions'][0]
    question.update(question='Did Ana agree?',answers=['review'],evidence=[{'start':-1,'end':2,'quote':'x'}])
    categories = {i['category'] for i in audit_case(case)}
    assert {'yesno_answer_missing','invalid_evidence'} <= categories


def test_unknown_is_not_automatically_changed_to_no():
    case = example()
    case['holistic_questions'][0].update(question='Did the team approve hiring?',answers=['not found'],expected_found=False)
    issues = audit_case(case)
    assert not any(i['category'] in ('yesno_answer_missing','yesno_extra_variant','inconsistent_found') for i in issues)


def test_choice_questions_and_optional_synonyms_are_not_missing_boolean_labels():
    case = example()
    case['holistic_questions'][0].update(question='Was the proposal accepted or rejected?',answers=['accepted'])
    assert not any(i['category'].startswith('yesno') for i in audit_case(case))
    case['holistic_questions'][0].update(question='Was the proposal accepted?',answers=['no','rejected'])
    categories = {i['category'] for i in audit_case(case)}
    assert 'yesno_extra_variant' in categories and 'yesno_answer_missing' not in categories
