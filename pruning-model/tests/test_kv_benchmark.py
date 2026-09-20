from copy import deepcopy

from pruning_model.kv_benchmark import clean_sessions, prepare_plan
from pruning_model.spans import PruneResult


def case():
    return {"question_id": "example", "question_type": "knowledge-update",
            "question": "SECRET FINAL QUESTION", "answer": "SECRET GOLD",
            "question_date": "2023/04/12 (Wed) 12:00",
            "haystack_session_ids": ["later", "earlier"],
            "haystack_dates": ["2023/04/11 (Tue) 12:00", "2023/04/10 (Mon) 12:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "new question", "has_answer": True},
                 {"role": "assistant", "content": "new fact"}],
                [{"role": "user", "content": "old question"},
                 {"role": "assistant", "content": "old fact"}]]}


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nUSER\n" + messages[1]["content"] + "\nASSISTANT\n"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class RecordingPruner:
    def __init__(self):
        self.calls = []

    def prune(self, question, notes):
        self.calls.append((question, notes))
        start = notes.index("fact")
        return PruneResult(question, notes[:start] + notes[start + 4:],
                           [{"start": start, "end": start + 4, "kept": False}], [], "selected", 1)


def test_sessions_sorted_and_policy_does_not_receive_gold_or_future_query():
    source = case()
    snapshot = deepcopy(source)
    sessions = clean_sessions(source)
    assert [s["id"] for s in sessions] == ["earlier", "later"]
    assert "has_answer" not in str(sessions)
    pruner = RecordingPruner()
    plan = prepare_plan(source, CharacterTokenizer(), pruner, recent_tokens=1)
    assert [q for q, _ in pruner.calls] == ["old question", "new question"]
    assert "SECRET" not in str(pruner.calls)
    assert source == snapshot
    assert all(step["boundary"] < plan["prompt"].index("SECRET FINAL QUESTION") for step in plan["steps"])
    assert plan["compression_ms"] == 2
    assert len(plan["final_retained_positions"]) < plan["original_prompt_tokens"]


def test_final_question_changes_never_change_historical_retention():
    one, two = case(), case()
    two["question"], two["answer"] = "OTHER FUTURE QUERY", "OTHER GOLD"
    a = prepare_plan(one, CharacterTokenizer(), RecordingPruner(), recent_tokens=1)
    b = prepare_plan(two, CharacterTokenizer(), RecordingPruner(), recent_tokens=1)
    assert a["steps"] == b["steps"]
    assert a["pruning"] == b["pruning"]
