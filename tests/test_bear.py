from types import SimpleNamespace
from tokenese import MODEL
from tokenese.bear import BearBaseline


class FakeBearClient:
    def __init__(self):
        self.calls = []

    def compress(self, notes, *, model, aggressiveness):
        self.calls.append((notes, model, aggressiveness))
        return SimpleNamespace(output=f"compressed {aggressiveness}", input_tokens=50, output_tokens=20)


def test_bear_uses_pinned_model_and_reuses_compressions():
    baseline = BearBaseline.__new__(BearBaseline)
    baseline.client = FakeBearClient()
    baseline.cache = {}
    result = baseline.compress("meeting notes", "Who owns it?", 100, MODEL)
    baseline.compress("meeting notes", "When is it due?", 100, MODEL)
    assert len(baseline.client.calls) == 3
    assert all(call[1] == "bear-2" for call in baseline.client.calls)
    assert result["bear_input_tokens"] == 50
    assert result["bear_output_tokens"] == 20
    assert result["bear_trial_input_tokens"] == 150
    assert result["bear_trial_output_tokens"] == 60
