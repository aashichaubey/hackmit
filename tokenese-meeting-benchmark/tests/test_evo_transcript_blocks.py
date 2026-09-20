import pytest

from tokenese.evo_transcript_blocks import compile_blocks, decode_blocks


def test_blocks_preserve_every_turn_and_identity_for_inline_transcripts():
    source = '& SPEAKER_00: No. & SPEAKER_00: Never. & SPEAKER_00: Never. & SPEAKER_1: Yes. & SPEAKER_1: Friday.'
    encoded = compile_blocks(source)
    assert encoded.profile == 'MeeQA'
    assert '\n  - Never.' in encoded.text
    assert '\n& SPEAKER_1:' in encoded.text
    assert encoded.turns == 5
    assert decode_blocks(encoded.text, encoded.profile) == source
    assert encoded.encoded_tokens < encoded.source_tokens


def test_named_speaker_blocks_keep_unicode_body_and_multiline_turns():
    source = 'Alice Smith: No.\nAlice Smith: Łukasz paid $50.\nMore detail.\nAlice Smith: No.\nBob Jones: Friday.\nBob Jones: Not Monday.'
    encoded = compile_blocks(source)
    assert encoded.profile == 'named'
    assert '\n  - Łukasz paid $50.\nMore detail.' in encoded.text
    assert decode_blocks(encoded.text, encoded.profile) == source


def test_collision_or_unrecognized_framing_remains_raw():
    for source in ('plain text', 'Alice: Heading.', 'Notes:\nAlice: A.\nAlice: B.',
                   '& SPEAKER_1: literal   - list.',
                   'Speaker 0: A & SPEAKER_1: B'):
        encoded = compile_blocks(source)
        assert encoded.profile == 'raw'
        assert encoded.text == source


def test_nonexpansion_and_explicit_decode_errors():
    source = 'Speaker 1: A.\n Speaker 1: B.\n Speaker 2: No.'
    encoded = compile_blocks(source)
    assert encoded.encoded_tokens <= encoded.source_tokens
    assert decode_blocks(encoded.text, encoded.profile) == source
    with pytest.raises(ValueError):
        decode_blocks('  - orphan', 'named')
    with pytest.raises(ValueError):
        decode_blocks('x', 'unsupported')


def test_contextual_quotes_expand_bullets_and_check_all_occurrences():
    from tokenese.evo_transcript_blocks import decode_blocks_quote
    text = '& SPEAKER_1: A. \n  - Same. \n& SPEAKER_2: B. \n  - Same.'
    assert decode_blocks_quote(text, 'MeeQA', '1') == '1'
    assert decode_blocks_quote(text, 'MeeQA', 'Same.') == 'Same.'
    assert decode_blocks_quote(text, 'MeeQA', text) == decode_blocks(text, 'MeeQA')
    with pytest.raises(ValueError, match='ambiguous_inherited_quote'):
        decode_blocks_quote(text, 'MeeQA', '  - Same.')
    with pytest.raises(ValueError, match='partial_synthetic_marker_quote'):
        decode_blocks_quote(text, 'MeeQA', '- Same.')
    with pytest.raises(ValueError, match='ungrounded_quote'):
        decode_blocks_quote(text, 'MeeQA', 'missing')
    named = 'Alice Smith: First.\n  - Second.\nBob Jones: Third.'
    assert decode_blocks_quote(named, 'named', '  - Second.') == 'Alice Smith: Second.'
    assert decode_blocks_quote(named, 'named', 'Second.\nBob Jones:') == 'Second.\nBob Jones:'
