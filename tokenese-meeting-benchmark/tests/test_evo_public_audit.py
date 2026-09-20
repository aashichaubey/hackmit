"""Regression checks from the public benchmark correctness review."""
import pytest
import json
from contextlib import nullcontext
from types import SimpleNamespace
from pydantic import ValidationError

from tokenese.evo_public_dialogue import DialogueAnswer, attribution_score


@pytest.mark.parametrize('indices', [[True], [1, 999], [-1]])
def test_invalid_citations_cannot_earn_reference_overlap(indices):
    result = attribution_score(indices, [{'startIndex': 1, 'endIndex': 1}], 3)
    assert not result['valid_citations']
    assert all(result[key] == 0 for key in ('precision', 'recall', 'f1', 'iou'))


def test_dialogue_schema_rejects_boolean_citation_indices():
    with pytest.raises(ValidationError):
        DialogueAnswer(response='A response', segment_indices=[True])


def test_singleton_attribution_is_inclusive_and_duplicate_predictions_are_a_set():
    result = attribution_score([1, 1], [{'startIndex': 1, 'endIndex': 1}], 3)
    assert result['valid_citations']
    assert result['f1'] == 1


def test_saved_request_drift_fails_reproduction_even_when_scores_match(tmp_path, monkeypatch):
    from tokenese import evo_public_report as audit
    from tokenese.evo_public_eval import SpanAnswer, aggregate, prompt, score

    case = {'id': 'q', 'context': 'Plain notes.', 'question': 'Deadline?',
            'expected_found': False, 'references': [[]]}
    rows = []
    for method in ('raw', 'encoded'):
        result = {'answer': {'found': False, 'spans': []}, 'error': None,
                  'usage': {'input_tokens': 10, 'output_tokens': 2},
                  'request': {'input': prompt(case['context'], case['question']),
                              'schema': SpanAnswer.model_json_schema(), 'max_output_tokens': 512}}
        rows.append({'id': 'q', 'dataset': 'x', 'meeting_id': 'm', 'family_id': 'm',
                     'method': method, 'profile': 'raw', 'result': result, 'score': score(case, result)})
    report = {'sources': {}, 'max_output_tokens': 512, 'advance': False,
              'methods': {m: aggregate([r for r in rows if r['method'] == m]) for m in ('raw', 'encoded')}}
    monkeypatch.setattr(audit, 'REPORTS', tmp_path)
    monkeypatch.setattr(audit, 'load_training', lambda: ({'x': [case]}, [], {}))
    (tmp_path / 'pilot.json').write_text(json.dumps(report))
    path = tmp_path / 'pilot-rows.json'
    path.write_text(json.dumps(rows))
    assert audit.run()['all_reproduced']
    rows[1]['result']['request']['input'] = 'A different experiment prompt'
    path.write_text(json.dumps(rows))
    assert not audit.run()['all_reproduced']


@pytest.mark.parametrize('broken', ['catalog.json', 'local-search.json', 'dialogue.json'])
def test_public_ui_reports_corrupt_artifacts_without_crashing(monkeypatch, broken):
    from tokenese import evo_view

    warnings = []
    artifacts = {'catalog.json': {'counts': {}}}
    artifacts[broken] = {'artifact_error': f'Cannot read {broken}'}
    monkeypatch.setattr(evo_view, 'read_artifact', lambda name, directory: artifacts.get(name, {}))
    monkeypatch.setattr(evo_view, 'st', SimpleNamespace(
        warning=warnings.append, expander=lambda *args, **kwargs: nullcontext(),
        write=lambda *args: None, caption=lambda *args: None, markdown=lambda *args: None,
        dataframe=lambda *args, **kwargs: None))
    evo_view.render_public_benchmarks()
    assert warnings == [f'Cannot read {broken}']


def test_tradeoff_table_keeps_quality_loss_and_cost_separate(tmp_path):
    from tokenese.evo_tradeoffs import load_tradeoffs
    import json
    raw={'questions':64,'actual_input_tokens':1000,'macro_f1':.5,'balanced_answerability':.6,'invalid_answers':0,'actual_usd':.01}
    encoded={**raw,'actual_input_tokens':900,'macro_f1':.48,'actual_usd':.011}
    (tmp_path/'heldout.json').write_text(json.dumps({'methods':{'raw':raw,'encoded':encoded}}))
    row=load_tradeoffs(tmp_path)[0]
    assert row['input saving %']==pytest.approx(10)
    assert row['F1 loss (pp)']==pytest.approx(2)
    assert row['cost saving %']==pytest.approx(-10)
