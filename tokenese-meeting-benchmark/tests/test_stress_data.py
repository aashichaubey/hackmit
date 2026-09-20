import json
from tokenese.benchmark import ROOT
from tokenese.facts import MeetingFacts


def test_speaker_labeled_stress_cases():
    cases = json.loads((ROOT / "data" / "stress_meetings.json").read_text())["meetings"]
    assert len(cases) == 6
    assert len({case["id"] for case in cases}) == 6
    for case in cases:
        assert len(case["questions"]) == 4
        facts = MeetingFacts(facts=case["facts"])
        assert all(fact.source_quote in case["notes"] for fact in facts.facts)
