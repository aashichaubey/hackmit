from tokenese.evo_source_groups import compile_groups,decode_groups


def test_consecutive_actor_inheritance_preserves_complete_source():
    source = '- The Steering Council discussed A. - The Steering Council did not approve B. - The Steering Council proposed C for next Friday. - The Steering Council discussed D.\n\nBudget: $50; status unclear.'
    result = compile_groups(source)
    assert result.route == 'source_groups'
    assert result.encoded_tokens < result.source_tokens
    assert decode_groups(result.text) == source
    assert 'did not approve B.' in result.text
    assert 'next Friday' in result.text and 'Budget: $50; status unclear.' in result.text


def test_grouping_never_moves_nonconsecutive_actors_or_erases_duplicates():
    source = '- Ana will review. - Ana will review. - Ana will review. - Bo will ship. - Ana will wait.\r\n'
    result = compile_groups(source)
    assert decode_groups(result.text) == source
    assert result.text.count('will review.') == 3
    assert result.text.index('Bo will ship.') < result.text.index('Ana will wait.')


def test_literal_delimiters_and_code_are_not_interpreted_as_groups():
    for source in ('- Ana: {x;y}', '- Ana will use `x`. - Ana will use y. - Ana will use z.',
                   '- Ana will use x;y. - Ana will use z. - Ana will use p.',
                   '- "Ana will use x. - Ana will use y. - Ana will use z."'):
        result = compile_groups(source)
        assert result.text == source and result.route == 'raw'


def test_fenced_code_and_inline_literals_remain_exact():
    line = '- The Steering Council uses `C++`. - The Steering Council uses "A". - The Steering Council uses B. - The Steering Council uses C.'
    encoded = compile_groups(line)
    assert decode_groups(encoded.text) == line
    assert '`C++`' in encoded.text and '"A"' in encoded.text
    for text in ('```\n'+line+'\n```','```\n'+line):
        assert compile_groups(text).text == text
