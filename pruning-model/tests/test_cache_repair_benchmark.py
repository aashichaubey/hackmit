from copy import deepcopy

from pruning_model.cache_repair_benchmark import prepare_edit
from pruning_model.kv_benchmark import prepare_plan
from pruning_model.spans import assemble, segment
from test_kv_benchmark import CharacterTokenizer, case


class FixedPruner:
    def prune(self, question, notes):
        spans = segment(notes)
        return assemble(question, notes, spans, [float("question" in s.text) for s in spans], .5)


def test_uses_actual_compressed_text_and_preserves_exact_source_alignment():
    source, tokenizer = case(), CharacterTokenizer()
    plan = prepare_plan(source, tokenizer, FixedPruner(), recent_tokens=1)
    edit = prepare_edit(source, plan, tokenizer)
    assert "old fact" in edit["original_text"]
    assert "old fact" not in edit["compressed_text"]
    for result in plan["pruning"]:
        assert result["notes"] in edit["compressed_text"]
    assert tokenizer(edit["compressed_text"])["input_ids"] == edit["new_ids"] + edit["query_ids"]
    assert len(edit["new_ids"]) < len(edit["old_ids"])
    assert None in edit["mapping"]  # Newline joins are conservatively recomputed.
    for i, original in enumerate(edit["mapping"]):
        if original is not None:
            assert edit["new_ids"][i] == edit["old_ids"][original]


def test_future_question_and_gold_do_not_affect_repair_inputs():
    one, tokenizer = case(), CharacterTokenizer()
    two = deepcopy(one)
    two["question"], two["answer"] = "Different future question", "Different gold"
    for session in two["haystack_sessions"]:
        for turn in session:
            turn["has_answer"] = not turn.get("has_answer", False)
    edits = [prepare_edit(c, prepare_plan(c, tokenizer, FixedPruner(), recent_tokens=1), tokenizer)
             for c in (one, two)]
    for key in ("old_ids", "new_ids", "mapping", "common_prefix_tokens"):
        assert edits[0][key] == edits[1][key]
    assert edits[0]["query_ids"] != edits[1]["query_ids"]
