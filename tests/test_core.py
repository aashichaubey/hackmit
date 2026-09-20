import json
import random

import pytest
import tiktoken

from tokenmix import Entry, compress, load_dictionary
from tokenmix.cli import SEED


def pair(en="as soon as possible", zh="尽快", **kwargs):
    return Entry("test", en, zh, enabled=True, **kwargs)


def test_direction_depends_on_tokenizer():
    entry = pair("artificial intelligence", "人工智能")
    assert compress(entry.en, [entry], encoding="o200k_base").text == entry.zh
    assert compress(entry.en, [entry], encoding="cl100k_base").text == entry.en
    assert compress(entry.zh, [entry], encoding="cl100k_base").text == entry.en


def test_ties_keep_original():
    assert compress("machine learning", [pair("machine learning", "机器学习")]).text == "machine learning"


def test_full_context_overrules_isolated_saving():
    # At the start of a string this loses the leading-space tokenization benefit.
    # Find a real boundary case instead of assuming word counts are additive.
    enc = tiktoken.get_encoding("o200k_base")
    entry = pair("artificial intelligence", "人工智能")
    found = False
    for left in ("Explain ", "About ", "Topic: ", "\n", "(", " "):
        for right in (".", ")", "", "!", "?", "\n"):
            original = left + entry.en + right
            candidate = left + entry.zh + right
            if len(enc.encode_ordinary(candidate)) >= len(enc.encode_ordinary(original)):
                found = True
                assert compress(original, [entry]).text == original
    assert found, "fixture must include an actual contextual tokenization counterexample"


@pytest.mark.parametrize("text", [
    '`as soon as possible`', '``as soon as possible``',
    '```text\nas soon as possible\n```', '~~~\nas soon as possible\n~~~',
    '```text\nas soon as possible',
    '"as soon as possible"', "'as soon as possible'", '“as soon as possible”',
    '「as soon as possible」', 'https://example.org/人工智能',
    '{{artificial intelligence}}', '${artificial intelligence}',
])
def test_protected_regions(text):
    assert compress(text, load_dictionary(SEED)).text == text


def test_numbers_and_custom_literals():
    entry = pair("in 30 days", "三十天后")
    assert compress(entry.en, [entry]).text == entry.en
    entry = pair("thirty days", "30天")
    assert compress(entry.en, [entry]).text == entry.en
    original = "Explain artificial intelligence as soon as possible."
    result = compress(original, load_dictionary(SEED), protect=["artificial intelligence"])
    assert "artificial intelligence" in result.text
    assert "尽快" in result.text


def test_do_not_match_inside_words_or_identifiers():
    entry = pair("operating system", "操作系统")
    original = "cooperating systems operating system_name"
    assert compress(original, [entry]).text == original


def test_ambiguous_dictionary_abstains_in_both_directions():
    entries = [pair(), Entry("other", "as soon as possible", "尽早", enabled=True)]
    assert compress("as soon as possible", entries).text == "as soon as possible"
    entries = [pair(), Entry("other", "at the earliest opportunity", "尽快", enabled=True)]
    assert compress("尽快", entries).text == "尽快"


def test_disabled_and_other_domains_abstain():
    entry = Entry("disabled", "as soon as possible", "尽快")
    assert compress(entry.en, [entry]).text == entry.en
    entry = pair(domain="technical")
    assert compress(entry.en, [entry]).text == entry.en
    assert compress(entry.en, [entry], domain="technical").text == entry.zh


def test_prefix_overhead_falls_back_to_original():
    original = "as soon as possible"
    result = compress(original, [pair()], prefix="Read the following bilingual text and reply in English.\n")
    assert result.text == original
    assert result.prefix == ""
    assert result.saved_tokens == 0
    assert not result.changes


def test_prefix_counted_when_savings_pay_for_it():
    original = "as soon as possible; " * 20
    result = compress(original, [pair()], prefix="Answer in English.\n")
    assert result.saved_tokens > 0
    assert result.prefix == "Answer in English.\n"
    enc = tiktoken.get_encoding(result.encoding)
    assert result.payload_tokens == len(enc.encode_ordinary(result.to_dict()["payload"]))


def test_overlapping_phrases_and_repeated_occurrences():
    entries = [pair(), Entry("long", "as soon as possible please", "请尽快", enabled=True)]
    original = "as soon as possible please; as soon as possible please"
    result = compress(original, entries)
    ordered = sorted(result.changes, key=lambda c: c.start)
    assert all(a.end <= b.start for a, b in zip(ordered, ordered[1:]))
    assert all(original[c.start:c.end] == c.original for c in ordered)
    assert result.saved_tokens > 0


def test_empty_and_special_token_text_are_ordinary_input():
    for original in ("", "<|endoftext|>", "🐍🦋 café\n"):
        result = compress(original, [pair()])
        assert result.text == original
        assert result.saved_tokens == 0


def test_no_expansion_and_accounting_across_mixed_inputs():
    rng = random.Random(7)
    entries = load_dictionary(SEED)
    snippets = [e.en for e in entries] + [e.zh for e in entries] + ["42", "`artificial intelligence`", ""]
    for encoding in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(encoding)
        for _ in range(40):
            original = "\n".join(rng.choices(snippets, k=5))
            result = compress(original, entries, encoding=encoding, prefix="Reply in English.\n")
            assert result.payload_tokens == len(enc.encode_ordinary(result.prefix + result.text))
            assert result.original_tokens == len(enc.encode_ordinary(original))
            assert result.payload_tokens <= result.original_tokens
            assert result.saved_tokens == result.original_tokens - result.payload_tokens


def test_max_changes_and_invalid_inputs():
    assert compress("as soon as possible " * 5, [pair()], max_changes=1).changes.__len__() == 1
    assert not compress("as soon as possible", [pair()], max_changes=0).changes
    with pytest.raises(ValueError, match="max_changes"):
        compress("", [], max_changes=-1)
    with pytest.raises(ValueError, match="must not be empty"):
        compress("", [], protect=[""])
    with pytest.raises(ValueError, match="nonempty"):
        Entry("x", "", "中")
    with pytest.raises(ValueError, match="boolean"):
        Entry("x", "text", "中", enabled="false")


def test_dictionary_errors_include_line_number(tmp_path):
    path = tmp_path / "pairs.jsonl"
    row = json.dumps({"id": "same", "en": "text", "zh": "文字"})
    path.write_text(row + "\n" + row, encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: duplicate id"):
        load_dictionary(path)
