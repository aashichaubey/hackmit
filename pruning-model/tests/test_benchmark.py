import pytest
from pruning_model.audit import audit_grades, grade_fingerprint
from pruning_model.benchmark import benchmark
from pruning_model.spans import assemble, segment

from tokenmix.evaluation.storage import (
    file_hash,
    read_jsonl,
    strict_json,
    write_json,
    write_jsonl,
)


def test_benchmark_runs_paired_interfaces_and_writes_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")

    class Pruner:
        def __init__(self, *args, **kwargs):
            self.metadata = {
                "training_meetings": ["train"],
                "validation_meetings": ["val"],
                "thresholds": {"conservative": 0.5, "balanced": 0.5, "aggressive": 0.5},
            }

        def prune(self, question, notes, **kwargs):
            return assemble(question, notes, segment(notes), [1, 1, 0], 0.5)

    monkeypatch.setattr("pruning_model.model.ModernBertPruner", Pruner)

    class API:
        def __init__(self):
            self.generation = {"model": "fake", "temperature": 0}

        journal = tmp_path / "journal.jsonl"

        def call(self, messages, **kwargs):
            # The inference request cannot contain reference facts or labels.
            payload = strict_json(messages[1].content)
            assert set(payload) == {"question", "meeting_notes"}
            return {
                "output": "Alex",
                "input_tokens": 100 if "Lunch" in payload["meeting_notes"] else 70,
                "token_count_trustworthy": True,
                "output_tokens": 1,
                "error": None,
                "model": "fake",
                "provider": "fake",
                "finish_reason": "stop",
            }

        def json(self, *args, **kwargs):
            return {
                "covered_fact_ids": [0],
                "contradicted_fact_ids": [],
                "unsupported_claims": [],
                "correct_abstention": False,
                "explanation": "supported",
            }

    dataset = tmp_path / "test.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "id": "e",
                "meeting_id": "heldout",
                "split": "test",
                "kind": "notes",
                "notes": "# Tasks\n- Alex owns migration.\n- Lunch at noon.",
                "question": "Who owns migration?",
                "reference_answer": "Alex",
                "answerable": True,
                "keep_ids": [1],
                "facts": [{"claim": "Alex owns migration.", "evidence_ids": [1]}],
            }
        ],
    )
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "encoder").mkdir(parents=True)
    for path in [
        checkpoint / "pruner.json",
        checkpoint / "encoder" / "model.safetensors",
        checkpoint / "classifier.safetensors",
    ]:
        path.write_text("test-fingerprint")
    output = tmp_path / "benchmark"
    benchmark(dataset, checkpoint, output, API(), include_lingua=False)
    report = strict_json((output / "report.json").read_text())
    assert report["complete"]
    assert len(read_jsonl(output / "per_call.jsonl")) == 3
    assert report["variants"]["original_repeat"]["overall_percent_saved"] == 0
    assert (
        report["variants"]["modernbert_conservative+balanced+aggressive"][
            "overall_percent_saved"
        ]
        == 30
    )
    for name in (
        "per_call.csv",
        "per_call.json",
        "report.md",
        "tradeoff.png",
        "regressions.json",
    ):
        assert (output / name).stat().st_size > 0
    raw_rows = read_jsonl(output / "per_call.jsonl")
    source_hash = file_hash(output / "per_call.jsonl")
    review = tmp_path / "review.json"
    write_json(
        review,
        {
            "input_sha256": source_hash,
            "reviewer": "test reviewer",
            "reviews": [
                {
                    "case_id": "e",
                    "arm": "original",
                    "grade_sha256": grade_fingerprint(raw_rows[0], "original"),
                    "reason": "Deliberately wrong original grade to test propagation across all pairs.",
                    "verdict": {
                        "covered_fact_ids": [],
                        "contradicted_fact_ids": [],
                        "unsupported_claims": [],
                        "correct_abstention": False,
                        "explanation": "missing required fact",
                    },
                }
            ],
        },
    )
    audited = tmp_path / "audited"
    audit_grades(output, dataset, review, audited)
    audited_rows = read_jsonl(audited / "per_call.jsonl")
    assert all(
        not row["original_correct"] and row["improvement"] for row in audited_rows
    )
    assert all(row["raw_original_grade"]["correct"] for row in audited_rows)
    assert file_hash(output / "per_call.jsonl") == source_hash
    with pytest.raises(ValueError, match="new/empty"):
        audit_grades(output, dataset, review, audited)
    write_jsonl(output / "per_call.jsonl", raw_rows[:1])
    with pytest.raises(ValueError, match="stale"):
        audit_grades(output, dataset, review, tmp_path / "stale")
