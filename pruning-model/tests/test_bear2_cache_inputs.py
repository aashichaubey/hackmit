from copy import deepcopy
from types import SimpleNamespace

import pytest

from pruning_model.bear2_cache_inputs import compress_sessions
from pruning_model.cache_repair_benchmark import prepare_external_edit
from pruning_model.kv_benchmark import clean_sessions
from test_kv_benchmark import CharacterTokenizer, case


class FakeClient:
    def __init__(self):
        self.calls = []

    def compress(self, text, **kwargs):
        self.calls.append((text, kwargs))
        output = text.replace("old fact", "OLD replacement")
        return SimpleNamespace(output=output, input_tokens=len(text), output_tokens=len(output),
                               tokens_saved=len(text) - len(output))


def test_bear_receives_only_history_and_exact_model_settings():
    client, records, source = FakeClient(), [], case()
    outputs, elapsed = compress_sessions(source, client, aggressiveness=.2, record=records.append)
    assert len(client.calls) == len(records) == 2
    assert all(kwargs == {"model": "bear-2", "aggressiveness": .2} for _, kwargs in client.calls)
    assert "SECRET" not in str(client.calls)
    assert "has_answer" not in str(client.calls)
    assert outputs == [r["output"] for r in records]
    assert elapsed >= 0


def test_exact_api_output_preserved_and_rewrites_do_not_reuse_wrong_tokens():
    source, tokenizer = case(), CharacterTokenizer()
    replacements = [s["notes"].replace("old fact", "invented ZZ") for s in clean_sessions(source)]
    edit = prepare_external_edit(source, replacements, tokenizer)
    for output in replacements:
        assert output in edit["compressed_text"]
    assert tokenizer(edit["compressed_text"])["input_ids"] == edit["new_ids"] + edit["query_ids"]
    assert None in edit["mapping"]
    mapped = [p for p in edit["mapping"] if p is not None]
    assert mapped == sorted(set(mapped))
    for i, old in enumerate(edit["mapping"]):
        if old is not None:
            assert edit["old_ids"][old] == edit["new_ids"][i]


def test_external_alignment_ignores_final_question_and_gold():
    a, b = case(), deepcopy(case())
    b["question"], b["answer"] = "changed final question", "different gold"
    replacements = [s["notes"].replace("old fact", "") for s in clean_sessions(a)]
    edits = [prepare_external_edit(c, replacements, CharacterTokenizer()) for c in (a, b)]
    for key in ("old_ids", "new_ids", "mapping", "common_prefix_tokens"):
        assert edits[0][key] == edits[1][key]
    assert edits[0]["query_ids"] != edits[1]["query_ids"]


def test_failed_call_is_recorded_without_retry_or_secret_error_text():
    class FailingClient:
        calls = 0

        def compress(self, *_args, **_kwargs):
            self.calls += 1
            raise ValueError("sensitive-server-error")

    client, records = FailingClient(), []
    with pytest.raises(RuntimeError, match="no automatic retry") as error:
        compress_sessions(case(), client, aggressiveness=.2, record=records.append)
    assert client.calls == len(records) == 1
    assert records[0]["status"] == "failed"
    assert "sensitive-server-error" not in str(records) + str(error.value)
