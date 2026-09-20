import json
from dataclasses import asdict

import pytest
from pruning_model.api import JournaledAPI
from pruning_model.benchmark import (
    bootstrap,
    grade_answer,
    lexical_prune,
    paired_row,
    summarize,
    text_changes,
)
from pruning_model.data import validate_examples
from pruning_model.spans import assemble, segment

from tokenmix.evaluation.adapters import GenerationResult, Message
from tokenmix.evaluation.storage import append_jsonl, digest


def example(split="test", meeting="one"):
    return {
        "id": meeting + ":question",
        "meeting_id": meeting,
        "split": split,
        "kind": "notes",
        "question": "When is the release?",
        "notes": "# Plan\n- Release is Friday only if approved.\n- Lunch is at noon.",
        "keep_ids": [1],
        "answerable": True,
        "reference_answer": "Friday only if approved.",
        "facts": [{"claim": "Friday only if approved", "evidence_ids": [1]}],
    }


def generation(tokens=100, **changes):
    return asdict(
        GenerationResult(
            "Friday only if approved.",
            input_tokens=tokens,
            token_count_trustworthy=tokens is not None,
            finish_reason="stop",
            model="fake",
            provider="fake",
            **changes,
        )
    )


def test_protected_headings_unicode_and_atomic_bullets():
    notes = "# Résumé\n## Actions\n- Zoë owns migration.\n  Only if security approves.\n## Other\n- Lunch at noon."
    spans = segment(notes)
    result = assemble("Who owns migration?", notes, spans, [0, 0, 0.9, 0, 0], 0.5)
    assert result.question == "Who owns migration?"
    assert result.retained_ids == [0, 1, 2]
    assert "Only if security approves." in result.notes
    assert "Other" not in result.notes
    for span in result.spans:
        assert notes[span["start"] : span["end"]] == span["text"]


def test_passthrough_is_byte_preserving_and_threshold_validated():
    notes = "\n# Notes\n\n- A.  \n- B.\n"
    spans = segment(notes)
    assert assemble("q", notes, spans, [0] * len(spans), 0.5).notes == notes
    assert assemble("q", notes, spans, [1] * len(spans), 0.5).notes == notes
    with pytest.raises(ValueError):
        assemble("q", notes, spans, [float("nan")] * len(spans), 0.5)
    with pytest.raises(ValueError):
        assemble("q", notes, spans, [0] * len(spans), 1.1)


def test_sentence_splitting_avoids_abbreviation():
    spans = segment("Dr. Smith owns it. Friday is conditional.")
    assert [s.text for s in spans] == ["Dr. Smith owns it.", "Friday is conditional."]


def test_character_diff_exposes_partial_passage_deletion():
    original = "- Approval is not yet granted.\n- Friday only if approved."
    compressed = "Approval granted.\nFriday approved."
    changes = text_changes(original, compressed)
    recovered = original
    for change in reversed(changes):
        assert (
            original[change["source_start"] : change["source_end"]] == change["removed"]
        )
        recovered = (
            recovered[: change["source_start"]]
            + change["inserted"]
            + recovered[change["source_end"] :]
        )
    assert recovered == compressed
    assert any("not yet" in change["removed"] for change in changes)


def test_dataset_leakage_and_bad_evidence_rejected():
    train, test = example("train"), example("test")
    test["id"] = "different"
    with pytest.raises(ValueError, match="more than one split"):
        validate_examples([train, test])
    with pytest.raises(ValueError, match="expected train"):
        validate_examples([test], require_split="train")
    test["keep_ids"] = [0]
    with pytest.raises(ValueError, match="content spans"):
        validate_examples([test])


def test_lexical_selection_keeps_conditions_and_heading():
    case = example()
    result = lexical_prune(case["question"], case["notes"], rate=0.3)
    assert "Friday only if approved" in result.notes
    assert result.retained_ids == [0, 1]


def test_measured_savings_loss_and_incomplete_usage():
    case = example()
    pruned = lexical_prune(case["question"], case["notes"])
    row = paired_row(
        case,
        "test",
        generation(100),
        generation(60),
        {"correct": True, "completeness": 1},
        {"correct": False, "completeness": 0.5},
        pruned,
    )
    assert row["tokens_saved"] == 40
    assert row["percent_saved"] == 40
    metrics = summarize([row])
    assert metrics["accuracy_loss_pp"] == 100
    assert metrics["mean_completeness_loss_pp"] == 50
    assert metrics["regression_rate_given_original_correct"] == 1
    missing = dict(
        row,
        tokens_saved=None,
        original_input_tokens=None,
        compressed_input_tokens=None,
        percent_saved=None,
    )
    assert summarize([row, missing])["overall_percent_saved"] is None
    assert summarize([row, missing])["token_measurement_coverage"] == 0.5
    assert bootstrap([row], resamples=10)["intervals"] is None
    second = dict(row, meeting_id="two")
    assert bootstrap([row, second], resamples=10)["accuracy_loss_pp_95ci"] == [100, 100]


class FakeTarget:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0

    def request(self, messages, settings):
        return {"messages": [asdict(m) for m in messages], **settings}

    def generate(self, messages, settings):
        self.calls += 1
        return next(self.results)


def api_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"generation": {"model": "fake", "max_tokens": 100}}))
    return config


def test_api_journals_retries_counts_them_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr("pruning_model.api.time.sleep", lambda _: None)
    target = FakeTarget(
        [
            GenerationResult("", error="HTTP 429 temporarily limited"),
            GenerationResult("ok", finish_reason="stop"),
        ]
    )
    path = tmp_path / "api.jsonl"
    api = JournaledAPI(api_config(tmp_path), path, max_calls=2, target=target)
    messages = (Message("user", "public notes"),)
    assert api.call(messages, purpose="p")["output"] == "ok"
    assert target.calls == 2
    resumed = JournaledAPI(api_config(tmp_path), path, max_calls=2, target=target)
    assert resumed.call(messages, purpose="p")["output"] == "ok"
    with pytest.raises(ValueError, match="cap"):
        resumed.call(messages, purpose="another")


def test_unresolved_request_is_not_silently_retried(tmp_path):
    target = FakeTarget([])
    config, path = api_config(tmp_path), tmp_path / "api.jsonl"
    api = JournaledAPI(config, path, max_calls=2, target=target)
    messages = (Message("user", "notes"),)
    generation = dict(api.generation, max_tokens=1024)
    key = digest(
        {
            "messages": [asdict(m) for m in messages],
            "generation": generation,
            "purpose": "p",
        }
    )
    append_jsonl(path, {"event": "started", "key": "attempt", "base_key": key})
    resumed = JournaledAPI(config, path, max_calls=2, target=target)
    with pytest.raises(ValueError, match="unknown consumption"):
        resumed.call(messages, purpose="p")
    assert target.calls == 0


def test_concurrent_api_budget_and_duplicate_requests(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    class Target(FakeTarget):
        def generate(self, messages, settings):
            self.calls += 1
            time.sleep(0.01)
            return GenerationResult("ok", finish_reason="stop")

    target = Target([])
    api = JournaledAPI(
        api_config(tmp_path), tmp_path / "api.jsonl", max_calls=1, target=target
    )
    messages = (Message("user", "notes"),)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: api.call(messages, purpose="same"), range(3)))
    assert all(r["output"] == "ok" for r in results)
    assert target.calls == api.used == 1


def test_anonymous_judge_uses_reference_not_baseline():
    class Judge:
        def json(self, system, payload, **kwargs):
            assert "reference_facts" in payload
            assert "baseline_answer" not in payload
            return {
                "covered_fact_ids": [0],
                "contradicted_fact_ids": [],
                "unsupported_claims": [],
                "correct_abstention": False,
                "explanation": "supported",
            }

    assert grade_answer(example(), generation(), Judge(), arm="compressed")["correct"]

    class BadJudge:
        def json(self, *args, **kwargs):
            return {
                "covered_fact_ids": [5],
                "contradicted_fact_ids": [],
                "unsupported_claims": [],
                "correct_abstention": False,
            }

    with pytest.raises(ValueError, match="invalid fact"):
        grade_answer(example(), generation(), BadJudge(), arm="x")
