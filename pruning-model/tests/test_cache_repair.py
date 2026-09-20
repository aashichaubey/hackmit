import pytest
import torch

from pruning_model.cache_repair import continue_after_edit, repair_cache, repair_positions
from pruning_model.kv_cache import QwenKVSession
from test_kv_cache import tiny_model


def test_full_repair_matches_fresh_shortened_prompt_and_preserves_original():
    model = tiny_model()
    old, new = [1, 2, 3, 4, 5, 6, 7], [1, 2, 4, 19, 6, 7]
    mapping = [0, 1, 3, None, 5, 6]
    session = QwenKVSession(model)
    session.append(old)
    before = [(l.keys.clone(), l.values.clone()) for l in session.cache.layers]
    repaired, stats = repair_cache(model, session.cache, old, new, mapping, fraction=1)
    fresh = QwenKVSession(model)
    fresh.append(new)
    for actual, expected, original, saved in zip(repaired.layers, fresh.cache.layers, session.cache.layers, before):
        torch.testing.assert_close(actual.keys, expected.keys, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.values, expected.values, atol=1e-6, rtol=1e-5)
        assert torch.equal(original.keys, saved[0])
        assert torch.equal(original.values, saved[1])
    continuation = QwenKVSession(model, cache=repaired, positions=list(range(len(new))), seen_tokens=len(new))
    torch.testing.assert_close(continuation.append([20, 21]), fresh.append([20, 21]), atol=1e-6, rtol=1e-5)
    assert stats["recomputed_tokens"] == len(new)


def test_rephasing_alone_repairs_first_layer_but_not_contextual_layers():
    model = tiny_model()
    old, mapping = [1, 2, 3, 4, 5, 6, 7], [0, 1, 4, 5, 6]
    new = [old[i] for i in mapping]
    session, fresh = QwenKVSession(model), QwenKVSession(model)
    session.append(old)
    fresh.append(new)
    repaired, stats = repair_cache(model, session.cache, old, new, mapping, fraction=0)
    torch.testing.assert_close(repaired.layers[0].keys, fresh.cache.layers[0].keys, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(repaired.layers[0].values, fresh.cache.layers[0].values, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(repaired.layers[1].values, fresh.cache.layers[1].values, atol=1e-6)
    assert stats["recomputed_tokens"] == 0


def test_identity_edit_preserves_all_values_and_new_tokens_are_always_repaired():
    model = tiny_model()
    old = [1, 2, 3, 4]
    session = QwenKVSession(model)
    session.append(old)
    repaired, _ = repair_cache(model, session.cache, old, old, list(range(4)), fraction=0)
    for actual, expected in zip(repaired.layers, session.cache.layers):
        assert torch.equal(actual.keys, expected.keys)
        assert torch.equal(actual.values, expected.values)
    assert repair_positions([0, 1, None, 3, None], 0) == [2, 4]
    assert repair_positions([0, 1, 5, 6, 9, 10], .5) == [2, 3, 4]


@pytest.mark.parametrize("new, expected_reuse", [
    ([1, 2, 4, 5], 2), ([19, 2, 3, 4, 5], 0), ([1, 2], 2),
    ([1, 2, 3, 4, 5], 5), ([1, 2, 3, 4, 5, 6], 5),
])
def test_default_transition_matches_fresh_inference_without_changing_source(new, expected_reuse):
    model, old, query = tiny_model(), [1, 2, 3, 4, 5], [20, 21]
    original = QwenKVSession(model)
    original.append(old)
    before = [(layer.keys.clone(), layer.values.clone()) for layer in original.cache.layers]
    session, stats, logits = continue_after_edit(model, original, old, new, query)
    fresh = QwenKVSession(model)
    expected = fresh.append(new + query)
    torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
    assert stats["reused_tokens"] == expected_reuse
    assert session.forward_tokens == len(new) - expected_reuse + len(query)
    for actual, saved in zip(original.cache.layers, before):
        assert torch.equal(actual.keys, saved[0])
        assert torch.equal(actual.values, saved[1])


def test_repairing_entire_affected_suffix_is_exact_even_with_reused_prefix():
    model = tiny_model()
    old, mapping = [1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 4, 5, 6]
    new = [old[i] for i in mapping]
    original, fresh = QwenKVSession(model), QwenKVSession(model)
    original.append(old)
    _, stats, logits = continue_after_edit(model, original, old, new, [20, 21],
                                          mapping=mapping, fraction=.5)
    torch.testing.assert_close(logits, fresh.append(new + [20, 21]), atol=1e-6, rtol=1e-5)
    assert stats["selected_positions"] == [3, 4, 5]


def test_transition_rejects_evicted_source_and_missing_provenance():
    model, old = tiny_model(), [1, 2, 3, 4]
    original = QwenKVSession(model)
    original.append(old)
    with pytest.raises(ValueError, match="mapping"):
        continue_after_edit(model, original, old, old, [20], fraction=.1)
    original.evict({0, 3})
    with pytest.raises(ValueError, match="contiguous"):
        continue_after_edit(model, original, old, old, [20])


@pytest.mark.parametrize("fraction", [-.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_repair_budget(fraction):
    with pytest.raises(ValueError, match="fraction"):
        repair_positions([0, 1, None], fraction)


@pytest.mark.parametrize("mapping", [[0, 2], [1, 0], [0, 9], [0, 0]])
def test_rejects_wrong_alignment(mapping):
    model = tiny_model()
    session = QwenKVSession(model)
    session.append([1, 2, 3])
    with pytest.raises(ValueError, match="mapping"):
        repair_cache(model, session.cache, [1, 2, 3], [1, 2], mapping, fraction=.5)
