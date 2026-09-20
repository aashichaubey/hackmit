import pytest

from tokenese.evo_transcript_search import compile_inheritance, decode_inheritance


def test_inheritance_preserves_turns_words_and_exact_speaker_identity():
    source = '& SPEAKER_00: Never. & SPEAKER_00: Not Friday. & SPEAKER_0: Friday. & SPEAKER_0: Friday.'
    encoded = compile_inheritance(source)
    assert encoded.text == '& SPEAKER_00: Never. ↳ Not Friday. & SPEAKER_0: Friday. ↳ Friday.'
    assert decode_inheritance(encoded.text, encoded.profile) == source
    assert encoded.encoded_tokens < encoded.source_tokens
    assert encoded.turns == 4


def test_inheritance_handles_initial_continued_speaker_and_multiline_content():
    source = 'Speaker 12: C++ costs $50.\n Speaker 12: Łukasz said no.\n Speaker 2: No.\n Speaker 2: No.\n'
    encoded = compile_inheritance(source)
    assert decode_inheritance(encoded.text, encoded.profile) == source
    assert 'Łukasz said no.' in encoded.text
    assert encoded.text.count('No.') == 2


def test_ambiguous_literal_markers_and_unsupported_text_remain_raw():
    for source in ('Speaker 0: literal ↳ is data.', 'Plain prose.',
                   'Speaker 0: A & SPEAKER_1: B',
                   'Speaker 0: Literal Speaker 1: text.\n Speaker 0: Again.'):
        encoded = compile_inheritance(source)
        assert encoded.profile == 'raw'
        assert encoded.text == source
    with pytest.raises(ValueError):
        decode_inheritance('↳ Missing preceding speaker.', 'MeeQA')


def test_changes_are_independent_of_question_or_reference_metadata():
    source = '& SPEAKER_9: yes. & SPEAKER_9: yes.'
    first = compile_inheritance(source)
    second = compile_inheritance(source)
    assert first == second
    assert first.source_tokens >= first.encoded_tokens


def test_quote_starting_with_inheritance_uses_full_transcript_context():
    from tokenese.evo_transcript_search import decode_quote
    text = '& SPEAKER_00: First. ↳ Second. ↳ Third.'
    assert decode_quote(text, 'MeeQA', '↳ Second.') == '& SPEAKER_00: Second.'
    assert decode_quote(text, 'MeeQA', 'Second. ↳ Third.') == 'Second. & SPEAKER_00: Third.'
    assert decode_quote(text, 'MeeQA', text) == decode_inheritance(text, 'MeeQA')


def test_quote_occurrences_must_agree_on_inherited_identity():
    from tokenese.evo_transcript_search import decode_quote
    text = '& SPEAKER_0: A. ↳ Same. & SPEAKER_1: B. ↳ Same.'
    with pytest.raises(ValueError, match='ambiguous_inherited_quote'):
        decode_quote(text, 'MeeQA', '↳ Same.')
    assert decode_quote(text, 'MeeQA', 'Same.') == 'Same.'
    assert decode_quote(text.replace('SPEAKER_1', 'SPEAKER_0'), 'MeeQA', '↳ Same.') == '& SPEAKER_0: Same.'


def test_quote_allows_unchanged_partial_labels_but_rejects_missing_and_bad_profiles():
    from tokenese.evo_transcript_search import decode_quote
    text = 'Speaker 12: A.\n ↳ B.'
    for quote in ('Speaker 1', '12: A.', 'peaker 12: A.'):
        assert decode_quote(text, 'meetingqa', quote) == quote
    with pytest.raises(ValueError, match='ungrounded_quote'):
        decode_quote(text, 'meetingqa', 'Missing')
    with pytest.raises(ValueError, match='unsupported_quote_profile'):
        decode_quote(text, 'other', 'A.')
    with pytest.raises(ValueError, match='empty_quote'):
        decode_quote(text, 'meetingqa', '')
    with pytest.raises(ValueError, match='continuation_without_speaker'):
        decode_quote('↳ Orphan.', 'meetingqa', 'Orphan.')
    assert decode_quote(text, 'raw', '↳ B.') == '↳ B.'
    assert decode_quote(text, 'meetingqa', '↳ B.') == 'Speaker 12: B.'


def test_quote_shared_by_unchanged_label_and_body_is_unambiguous():
    from tokenese.evo_transcript_search import decode_quote
    text = '& SPEAKER_1: The count is 1. ↳ Agreed.'
    assert decode_quote(text, 'MeeQA', '1') == '1'
