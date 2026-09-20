from tokenese.evo_classifier_next import source_spans, materialize, IndexedClassification
from tokenese.evo_classifier_next import materialize_selection, SelectedClassification
from tokenese.evo_classifier_next import paired_diagnostics
import pytest


def test_spans_are_exact_and_preserve_dates_and_negation():
    notes = 'Morgan: Ana will not ship by next Friday.\nBo proposed a review.'
    spans = source_spans(notes)
    assert all(notes[s.start:s.end] == s.text for s in spans)
    parsed = IndexedClassification(facts=[{'kind':'action','person':'Ana','text':'not ship','deadline':'next Friday','evidence_ids':[0]}],unsupported_content=False,ambiguity_notes=[])
    facts, errors = materialize(notes,parsed)
    assert errors == []
    assert facts.facts[0].source_quote in notes
    assert facts.facts[0].person == 'Ana'
    assert facts.facts[0].deadline == 'next Friday'


def test_nonliteral_fields_and_invalid_ids_rejected():
    notes = 'Ana will review by Friday.'
    for fields in ({'person':'Bo','evidence_ids':[0]}, {'person':'Ana','evidence_ids':[-1]}, {'person':'Ana','evidence_ids':[8]}):
        parsed = IndexedClassification(facts=[{'kind':'action','text':'review','deadline':'Friday',**fields}],unsupported_content=False,ambiguity_notes=[])
        facts,errors = materialize(notes,parsed)
        assert not facts.facts and errors


def test_evidence_order_and_duplicates_preserved():
    notes = 'Ana will review. Bo will ship.'
    f = {'kind':'action','person':'Ana','text':'review','deadline':'','evidence_ids':[0]}
    parsed = IndexedClassification(facts=[f,f],unsupported_content=False,ambiguity_notes=[])
    facts,errors = materialize(notes,parsed)
    assert len(facts.facts) == 2 and not errors
    parsed.facts[0].evidence_ids = [1,0]
    assert materialize(notes,parsed)[1]


def test_selected_sentences_copy_all_qualifications_and_context():
    notes = 'Morgan reports for Ana. Ana will not ship before next Friday.\nBo only proposes it.'
    parsed = SelectedClassification(facts=[{'kind':'action','evidence_ids':[0,1]}],unsupported_content=True,ambiguity_notes=[])
    facts,errors = materialize_selection(notes,parsed)
    assert not errors
    assert facts.facts[0].text == 'Morgan reports for Ana. Ana will not ship before next Friday.'
    assert facts.facts[0].source_quote == facts.facts[0].text
    assert facts.facts[0].person == ''  # No invented assignee from the speaker.
    for ids in ([],[-1],[99],[1,0],[0,0]):
        parsed.facts[0].evidence_ids = ids
        rejected,errors = materialize_selection(notes,parsed)
        assert not rejected.facts and errors


def test_paired_diagnostics_do_not_count_raw_fallback_as_compression_success():
    raw = [{'question_id':'a','correct':True},{'question_id':'b','correct':True}]
    candidate = [{'question_id':'b','correct':True,'route':'raw'},
                 {'question_id':'a','correct':False,'route':'encoded'}]
    result = paired_diagnostics(raw,candidate)
    assert result['new_errors_vs_raw'] == ['a']
    assert result['fallback_correct'] == 1 and result['encoded_correct'] == 0
    assert not result['no_new_errors']
    for bad in (candidate[:1], candidate + candidate[:1]):
        with pytest.raises(ValueError):
            paired_diagnostics(raw,bad)


def test_multiline_or_delimited_selected_evidence_rejects_without_crashing():
    for notes in ('Ana will review.\nBo will ship.', 'Ana will review A | B.'):
        spans = source_spans(notes)
        parsed = SelectedClassification(facts=[{'kind':'action','evidence_ids':list(range(len(spans)))}],unsupported_content=False,ambiguity_notes=[])
        facts,errors = materialize_selection(notes,parsed)
        assert not facts.facts and errors
