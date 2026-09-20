import pytest
from pruning_model.data import (
    annotate_notes,
    apply_review,
    curate,
    import_public,
)

from tokenmix.evaluation.storage import digest, read_jsonl, write_json, write_jsonl


def test_import_generic_notes_never_uses_specific_question(tmp_path, monkeypatch):
    qmsum, explain, out = tmp_path / "qmsum", tmp_path / "explain", tmp_path / "out"
    turn = {
        "speaker": "Alex",
        "content": "Release Friday only if approved.",
        "sentence_level_content": [
            {"sent_index": 0, "dialogue_sentence": "Release Friday only if approved."}
        ],
    }
    evidence = {
        "turn_index": 0,
        "sent_index": 0,
        "content": turn["content"],
        "speaker": "Alex",
        "type": "CES",
    }
    answer = {"answer_sentence": turn["content"], "evidence": [evidence]}
    specific = {
        "query": "SECRET_QUESTION",
        "answer": "Friday",
        "explainable_answer": [answer],
    }
    generic = {
        "query": "Summarize the meeting",
        "answer": turn["content"],
        "explainable_answer": [answer],
    }
    for split in ("train", "val", "test"):
        qp = qmsum / "data" / "ALL" / split
        ep = explain / "data" / "ExplainMeetSum" / split
        qp.mkdir(parents=True)
        ep.mkdir(parents=True)
        write_json(qp / f"{split}.json", {"specific_query_list": [specific]})
        write_json(
            ep / f"{split}.json",
            {
                "meeting_transcripts": [turn],
                "explainable_qmsum": {
                    "general_query_list": [generic],
                    "specific_query_list": [specific],
                },
            },
        )
    monkeypatch.setattr("pruning_model.data.source_revision", lambda _: "test_revision")
    result = import_public(qmsum, explain, out)
    assert result["transcript_examples"] == 3
    notes = read_jsonl(out / "notes_sources.jsonl")
    assert len(notes) == 3
    assert all("SECRET_QUESTION" not in r["notes"] for r in notes)
    assert all("only if approved" in r["notes"] for r in notes)


def test_review_preserves_sources_and_requires_exact_version(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    row = {
        "id": "one",
        "meeting_id": "m",
        "split": "train",
        "kind": "notes",
        "notes": "- IBM did it.",
        "question": "Who did it?",
        "answerable": False,
        "reference_answer": "Not stated",
        "facts": [],
        "keep_ids": [],
    }
    write_jsonl(source / "notes_train.jsonl", [row])
    review = {
        "input_digest": digest([row]),
        "reviewer": "test-reviewer",
        "examples": [
            {
                "id": "one",
                "verdict": "correct",
                "reason": "IBM is explicitly named",
                "correction": {
                    "answerable": True,
                    "reference_answer": "IBM",
                    "keep_ids": [0],
                    "facts": [{"claim": "IBM did it", "evidence_ids": [0]}],
                },
            }
        ],
    }
    path = tmp_path / "review.json"
    write_json(path, review)
    out = tmp_path / "corrected"
    assert apply_review(source, path, out)["reviewed_examples"] == 1
    assert read_jsonl(out / "notes_train.jsonl")[0]["reference_answer"] == "IBM"
    assert (
        read_jsonl(source / "notes_train.jsonl")[0]["reference_answer"] == "Not stated"
    )
    review["input_digest"] = "wrong"
    write_json(path, review)
    with pytest.raises(ValueError, match="different dataset"):
        apply_review(source, path, out)


def test_annotation_rejects_invented_quotes_before_verification(tmp_path):
    row = {
        "meeting_id": "m",
        "split": "train",
        "notes": "- IBM did it.",
        "provenance": {},
        "source_evidence": {},
    }
    sources = tmp_path / "sources.jsonl"
    write_jsonl(sources, [row])

    class Fake:
        def json(self, *args, **kwargs):
            return {
                "examples": [
                    {
                        "question": "Who did it?",
                        "answerable": True,
                        "reference_answer": "IBM",
                        "facts": [
                            {
                                "claim": "IBM did it",
                                "evidence_ids": [0],
                                "quotes": ["Microsoft did it"],
                            }
                        ],
                    }
                ]
            }

    with pytest.raises(ValueError, match="not an exact source"):
        annotate_notes(
            sources,
            tmp_path / "out",
            Fake(),
            per_split={"train": 1, "val": 0, "test": 0},
        )


def test_curated_labels_cannot_move_meetings_between_splits(tmp_path):
    from tokenmix.evaluation.storage import file_hash

    source = tmp_path / "sources.jsonl"
    write_jsonl(
        source,
        [
            {
                "meeting_id": "m",
                "split": "train",
                "notes": "- Alex owns migration.",
                "provenance": {},
                "source_evidence": {},
            }
        ],
    )
    train = tmp_path / "train.jsonl"
    write_jsonl(
        train,
        [
            {
                "id": "train",
                "meeting_id": "m",
                "split": "train",
                "question": "Who?",
                "notes": "- Alex owns migration.",
                "keep_ids": [0],
                "answerable": True,
                "facts": [],
            }
        ],
    )
    questions = tmp_path / "questions.json"
    write_json(
        questions,
        {
            "sources_sha256": file_hash(source),
            "author": "test",
            "questions": [
                {
                    "meeting_id": "m",
                    "question": "Who?",
                    "reference_answer": "Alex",
                    "answerable": True,
                    "facts": [{"claim": "Alex owns migration", "evidence_ids": [0]}],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="held-out sources"):
        curate(source, questions, train, tmp_path / "out")
