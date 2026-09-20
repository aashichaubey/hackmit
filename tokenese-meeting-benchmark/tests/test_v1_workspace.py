from types import SimpleNamespace

from tokenese.facts import Fact, MeetingFacts
from tokenese.v1_workspace import compile_once


class FakeResponses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        notes = kwargs["input"]
        facts = MeetingFacts(facts=[
            Fact(kind="decision", text="ship", source_quote=notes.split("<meeting>\n", 1)[1].split("\n</meeting>", 1)[0])
        ])
        return SimpleNamespace(
            output_parsed=facts,
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_compile_once_reuses_facts_for_the_same_notes():
    state = {}
    client = FakeClient()

    first = compile_once(state, "We decided to ship.", client)
    second = compile_once(state, "We decided to ship.", client)

    assert not first.reused
    assert second.reused
    assert first.facts == second.facts
    assert len(client.responses.calls) == 1


def test_compile_once_invalidates_when_notes_change():
    state = {}
    client = FakeClient()

    compile_once(state, "We decided to ship.", client)
    result = compile_once(state, "We decided to wait.", client)

    assert not result.reused
    assert result.facts.facts[0].source_quote == "We decided to wait."
    assert len(client.responses.calls) == 2
