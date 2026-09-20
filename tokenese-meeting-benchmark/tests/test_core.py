import pytest
from tokenese import MODEL
from tokenese.benchmark import choose_encoding, load_cases, summarize
from tokenese.encode import render_english, render_tokenese
from tokenese.facts import Answer, Fact, MeetingFacts
from tokenese.judges import grade_exact
from tokenese.prompts import answer_prompt
from tokenese.tokens import count_tokens
from tokenese.search import qualify_candidate

def test_dataset():
    cases = load_cases()
    assert len(cases) == 30
    assert [sum(case["split"] == split for case in cases) for split in ("dev", "validation", "test")] == [18, 6, 6]
    assert all(len(case["questions"]) == 4 for case in cases)
    assert len({case["id"] for case in cases}) == 30
    assert all(question["answers"] for case in cases for question in case["questions"])

def test_development_counterfactual_pairs():
    cases = load_cases()
    changes = ((0, 1, "David", "John"), (2, 3, "Thursday", "Friday"), (4, 5, "agreed", "disagreed"), (6, 7, "proposed", "decided"))
    for left, right, old, new in changes:
        assert cases[left]["split"] == cases[right]["split"] == "dev"
        assert cases[left]["notes"].replace(old, new) == cases[right]["notes"]

def test_encode_and_prompt():
    facts = MeetingFacts(facts=[Fact(kind="action", text="finish API", person="David", deadline="Thursday", source_quote="David will finish API by Thursday.")])
    assert render_tokenese(facts, "symbols") == "A|David|finish API|@Thursday"
    assert "David" in render_english(facts)
    assert "Question: Who?" in answer_prompt("hello", "Who?")
    assert count_tokens(answer_prompt("hello", "Who?"), MODEL) > 0

def test_decision_preserves_named_decider():
    facts = MeetingFacts(facts=[Fact(kind="decision", text="launch Friday", person="Sarah", source_quote="Sarah decided to launch Friday.")])
    assert render_tokenese(facts, "symbols") == "D|Sarah|launch Friday"

def test_delimiter_rejected():
    with pytest.raises(ValueError):
        Fact(kind="action", text="finish|API", source_quote="x")

def test_exact_grade():
    assert grade_exact(Answer(found=True, answer="David."), ["David"])
    assert grade_exact(Answer(found=False, answer="not found"), ["not found"])
    assert not grade_exact(Answer(found=True, answer="yes"), ["not found"])

def test_fallback():
    facts = MeetingFacts(facts=[Fact(kind="decision", text="Friday", source_quote="Launch Friday")])
    _, name = choose_encoding(facts, "What day?", qualified_profiles=["symbols"])
    assert name == "english"

def test_summary_keeps_errors():
    result = {"fact_source": "gold", "questions": [{"kind": "owner", "methods": {"raw": {"error": "failed", "correct": False}}}], "extraction": None}
    summary = summarize([result])["methods"]["raw"]
    assert summary["questions"] == 1
    assert summary["errors"] == 1
    assert summary["owner_errors"] == 1

def test_critical_error_blocks_candidate():
    for field in ("owner", "deadline", "negation", "decision"):
        assert not qualify_candidate({"critical_errors": [field], "answer_accuracy": 1.0, "english_accuracy": 1.0})

def test_unknown_model_fails():
    with pytest.raises(KeyError):
        count_tokens("hello", "made-up-model")
