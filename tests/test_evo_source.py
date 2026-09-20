from tokenese.evo_source import compile_source, decode_source


def test_source_grammar_preserves_all_information_and_reverses_without_trace():
    notes = 'Ana decided to retain C++ until next Friday.\nBudget is $50. Bo has NOT approved it.'
    result = compile_source(notes)
    assert result.route == 'source_grammar'
    assert result.encoded_tokens < result.source_tokens
    assert decode_source(result.text) == notes
    assert 'C++ until next Friday' in result.text
    assert 'Budget is $50. Bo has NOT approved it.' in result.text
    assert result.edits[0].kind == 'decision'


def test_source_grammar_preserves_quotes_code_and_literal_markers():
    for notes in ('"Ana decided to wait."', '`Ana decided to wait.`',
                  '```\nAna decided to wait.\n```', '“Ana decided to wait.”',
                  "'Ana decided to wait.'", 'Ana decided yes. Bo decided to go.',
                  '"Ana decided to wait.', '```Ana decided to wait.'):
        result = compile_source(notes)
        assert result.route == 'raw' and result.text == notes


def test_source_grammar_preserves_multiplicity_order_and_negation():
    notes = 'Ana decided to not ship. Bo proposed to ship. Ana decided to not ship.'
    result = compile_source(notes)
    assert decode_source(result.text) == notes
    assert [e.kind for e in result.edits] == ['decision','proposal','decision']


def test_source_grammar_unchanged_fallback_for_arbitrary_documents():
    for notes in ('', 'status: stalled\nowner: Ana', '日本語の議事録', 'A | B', 'proposal to review'):
        result = compile_source(notes)
        assert result.route == 'raw'
        assert result.text == notes and result.encoded_tokens == result.source_tokens
