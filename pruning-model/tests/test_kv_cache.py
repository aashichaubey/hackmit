import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from pruning_model.kv_cache import QwenKVSession, cache_bytes, select_cache


def tiny_model():
    torch.manual_seed(9)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=256,
                        use_sliding_window=False)
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).eval()


def test_chunked_prefill_equals_unmodified_model():
    model = tiny_model()
    ids = list(range(1, 30))
    session = QwenKVSession(model, chunk_size=7)
    actual = session.append(ids)
    with torch.inference_mode():
        expected = model(torch.tensor([ids])).logits[0, -1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert session.forward_tokens == len(ids)


def test_eviction_preserves_values_and_matches_masked_full_cache_continuation():
    """Independent reference: keep full KV, hide evicted columns with a mask."""
    model = tiny_model()
    prefix, suffix = list(range(1, 22)), [31, 32, 33, 34]
    session = QwenKVSession(model, chunk_size=64)
    session.append(prefix)
    keep = [0, 1, 5, 9, 15, 20]
    old = [(layer.keys.clone(), layer.values.clone()) for layer in session.cache.layers]
    old_bytes = cache_bytes(session.cache)
    event = session.evict(set(keep))
    assert session.forward_tokens == len(prefix)
    assert event["recomputed_tokens"] == 0
    assert cache_bytes(session.cache) * len(prefix) == old_bytes * len(keep)
    for layer, (keys, values) in zip(session.cache.layers, old):
        assert torch.equal(layer.keys, keys[:, :, keep, :])
        assert torch.equal(layer.values, values[:, :, keep, :])
    actual = session.append(suffix)
    with torch.inference_mode():
        reference = model(torch.tensor([prefix]), use_cache=True)
        mask = torch.tensor([[int(i in keep) for i in range(len(prefix))] + [1] * len(suffix)])
        expected = model(torch.tensor([suffix]), past_key_values=reference.past_key_values,
                         attention_mask=mask,
                         position_ids=torch.arange(len(prefix), len(prefix) + len(suffix))[None],
                         cache_position=torch.arange(len(prefix), len(prefix) + len(suffix)),
                         use_cache=True).logits[0, -1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert session.positions == keep + list(range(len(prefix), len(prefix) + len(suffix)))
    assert session.forward_tokens == len(prefix) + len(suffix)


def test_multiple_evictions_do_not_reintroduce_dropped_positions():
    session = QwenKVSession(tiny_model())
    session.append([1, 2, 3, 4, 5])
    session.evict({0, 4})
    session.append([6, 7, 8])
    session.evict({0, 2, 5, 7})
    assert session.positions == [0, 5, 7]
    assert session.seen_tokens == session.forward_tokens == 8
    session.append([9, 10])
    assert session.positions == [0, 5, 7, 8, 9]


@pytest.mark.parametrize("indices", [[], [2, 1], [1, 1], [-1], [9]])
def test_invalid_selection_fails_without_modifying_cache(indices):
    session = QwenKVSession(tiny_model())
    session.append([1, 2, 3])
    with pytest.raises(ValueError):
        select_cache(session.cache, indices)
    assert session.cache.get_seq_length() == 3


def test_original_position_limit_still_applies_after_eviction():
    model = tiny_model()
    model.config.max_position_embeddings = 5
    session = QwenKVSession(model)
    session.append([1, 2, 3, 4])
    session.evict({0})
    with pytest.raises(ValueError, match="logical positions"):
        session.append([5, 6])
