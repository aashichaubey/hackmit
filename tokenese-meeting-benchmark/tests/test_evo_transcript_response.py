"""Source-first restoration and symmetric product grounding regressions."""
from copy import deepcopy

import pytest

from tokenese.evo_transcript import TranscriptEncoding, compile_minimal_transcript
from tokenese.evo_transcript_blocks import compile_blocks
from tokenese.evo_transcript_response import restore_response


def response(spans, found=True):
    return {'answer': {'found': found, 'spans': spans}, 'error': None,
            'request': {'input': 'saved provider input'}, 'usage': {'input_tokens': 42},
            'replayed': True, 'response_id': 'saved'}


def raw_encoding(source):
    return TranscriptEncoding(source, 'raw', 1, 1, 0, 'raw_selected', 'raw')


def test_minimal_accepts_original_labels_without_double_expansion_or_mutation():
    source = '& SPEAKER_0: Not Friday. & SPEAKER_1: Monday.'
    encoding = compile_minimal_transcript(source)
    result = response(['& SPEAKER_1: Monday.'])
    original = deepcopy(result)
    restored = restore_response(source, encoding, result, 'minimal')
    assert restored['answer']['spans'] == ['& SPEAKER_1: Monday.']
    assert not restored['error']
    assert restored is not result and restored['answer'] is not result['answer']
    assert result == original
    assert {k: v for k, v in restored.items() if k != 'answer'} == {
        k: v for k, v in result.items() if k != 'answer'}


def test_minimal_restores_compact_label_and_raw_fallback():
    source = '& SPEAKER_0: Not Friday. & SPEAKER_1: Monday.'
    restored = restore_response(source, compile_minimal_transcript(source),
                                response(['SPEAKER_0: Not Friday. SPEAKER_1: Monday.']), 'minimal')
    assert restored['answer']['spans'] == [source]
    assert not restored['error']
    # A partial original label already exists in the source and needs no expansion.
    partial = restore_response(source, compile_minimal_transcript(source),
                               response(['SPEAKER_1: Monday.']), 'minimal')
    assert partial['answer']['spans'] == ['SPEAKER_1: Monday.']
    assert not restore_response('Plain notes.', raw_encoding('Plain notes.'),
                                response(['Plain notes.']), 'minimal')['error']


@pytest.mark.parametrize('method', ['raw', 'minimal', 'blocks'])
def test_every_method_rejects_ungrounded_quotes(method):
    result = restore_response('No decision.', raw_encoding('No decision.'), response(['Friday.']), method)
    assert result['error']


@pytest.mark.parametrize('answer', [
    None, {}, {'found': 'true', 'spans': ['Yes.']}, {'found': 1, 'spans': ['Yes.']},
    {'found': False, 'spans': ['Yes.']}, {'found': True, 'spans': []},
    {'found': True, 'spans': ['']}, {'found': True, 'spans': ['   ']},
    {'found': True, 'spans': [1]}, {'found': True, 'spans': 'Yes.'},
])
def test_malformed_or_contradictory_answers_fail_closed(answer):
    result = {'answer': answer, 'error': None}
    original = deepcopy(result)
    assert restore_response('Yes.', raw_encoding('Yes.'), result, 'raw')['error']
    assert result == original


def test_valid_abstention_and_provider_failure_are_preserved():
    result = response([], False)
    assert restore_response('No decision.', raw_encoding('No decision.'), result, 'raw') == result
    failed = {**result, 'error': 'Provider timeout', 'answer': None}
    assert restore_response('No decision.', raw_encoding('No decision.'), failed, 'raw') == failed


def test_blocks_accept_original_and_restore_contextual_bullet_quotes():
    source = '& SPEAKER_1: A. & SPEAKER_1: Same. & SPEAKER_2: B. & SPEAKER_2: Same.'
    encoding = compile_blocks(source)
    assert encoding.profile == 'MeeQA'
    valid = restore_response(source, encoding, response(['& SPEAKER_1: Same.']), 'blocks')
    assert not valid['error']
    ambiguous = restore_response(source, encoding, response(['  - Same.']), 'blocks')
    assert 'ambiguous_inherited_quote' in ambiguous['error']
    body = restore_response(source, encoding, response(['Same.']), 'blocks')
    assert not body['error']


def test_unknown_method_is_a_configuration_error():
    with pytest.raises(ValueError, match='Unknown transcript method'):
        restore_response('Yes.', raw_encoding('Yes.'), response(['Yes.']), 'other')
