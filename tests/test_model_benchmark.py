from types import SimpleNamespace
import pytest
from tokenese import MODEL
from tokenese.benchmark import load_cases, matches_gold_fact, qualify_profiles, run_case, summarize
from tokenese.facts import Answer, Fact, MeetingFacts
from tokenese.model import answer_question, extract_facts

class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []
    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed, usage=SimpleNamespace(input_tokens=10, output_tokens=2))

class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)

def test_model_schema_and_usage():
    client = FakeClient(MeetingFacts(facts=[]))
    facts, usage = extract_facts(client, "Notes", MODEL)
    assert facts.facts == []
    assert usage.input_tokens == 10
    assert client.responses.calls[0]["text_format"] is MeetingFacts
    client = FakeClient(Answer(found=False, answer="not found"))
    _, usage = answer_question(client, "Prompt", MODEL)
    assert usage.output_tokens == 2
    assert client.responses.calls[0]["text_format"] is Answer

def test_missing_parsed_output():
    with pytest.raises(ValueError, match="structured output"):
        extract_facts(FakeClient(None), "Notes", MODEL)

def test_extraction_rejects_invented_quote():
    facts = MeetingFacts(facts=[Fact(kind="decision", text="launch", source_quote="We decided to launch.")])
    with pytest.raises(ValueError, match="source quote"):
        extract_facts(FakeClient(facts), "The meeting was postponed.", MODEL)

def test_run_case_failure_is_counted():
    case = load_cases()[0]
    result = run_case(FakeClient(Answer(found=False, answer="not found")), case, methods=("raw", "deletion"))
    assert len(result["questions"]) == 4
    assert all(row["methods"]["deletion"]["error"] for row in result["questions"])
    assert all(row["methods"]["raw"]["usage"]["input_tokens"] == 10 for row in result["questions"])

def test_raw_answer_survives_extraction_failure():
    case = load_cases()[0]
    client = FakeClient(None)
    result = run_case(client, case, fact_source="parsed", methods=("raw", "english"))
    assert result["errors"]
    assert len(result["questions"]) == 4
    assert len(client.responses.calls) == 5
    assert all(row["methods"]["raw"]["error"] for row in result["questions"])
    assert all(row["methods"]["english"]["error"] == "Fact extraction failed" for row in result["questions"])

def test_qualification_requires_token_gain():
    row = {"kind": "owner", "methods": {"english": {"correct": True, "full_input_tokens": 10}, "symbols": {"correct": True, "full_input_tokens": 12}, "mixed": {"correct": False, "full_input_tokens": 8}}}
    assert qualify_profiles([{"questions": [row]}]) == []

def test_extraction_failure_counts_gold_facts_as_misses():
    case = load_cases()[0]
    result = run_case(FakeClient(None), case, fact_source="parsed", methods=("raw", "english"))
    summary = summarize([result])
    assert summary["extraction_misses"] == len(case["facts"])

def test_extraction_match_tolerates_longer_text_but_not_wrong_owner():
    gold = {"kind": "action", "person": "David", "deadline": "Thursday", "text": "finish the API"}
    found = {"kind": "action", "person": "David", "deadline": "Thursday", "text": "David will finish the API by Thursday"}
    assert matches_gold_fact(found, gold)
    assert not matches_gold_fact({**found, "person": "John"}, gold)
    decision = {"kind": "decision", "person": "", "deadline": "", "text": "launch plan"}
    assert matches_gold_fact({**decision, "person": "Sarah"}, decision)
    proposal = {"kind": "proposal", "person": "Maya", "deadline": "", "text": "move the launch", "source_quote": "I propose moving the launch."}
    extracted = {**proposal, "text": "moving the launch"}
    assert matches_gold_fact(extracted, proposal)
    assert not matches_gold_fact({**extracted, "kind": "decision"}, proposal)
