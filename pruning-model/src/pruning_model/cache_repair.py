"""Experimental cache reuse after editing text, for batch-one full-attention Qwen2.

Unlike KV eviction, positions here belong to the newly tokenized, shortened text.
Reused deeper-layer states are approximate because their preceding text changed.
This is a simple fixed-selection experiment, not an implementation of CacheBlend.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, rotate_half

from .kv_cache import QwenKVSession, select_cache


@torch.inference_mode()
def continue_after_edit(model, original: QwenKVSession, old_ids: list[int],
                        new_ids: list[int], query_ids: list[int], *,
                        mapping: list[int | None] | None = None,
                        fraction: float | None = None) -> tuple[QwenKVSession, dict, torch.Tensor]:
    """Continue an edited history without mutating its original cache.

    The default reuses only the identical prefix and recomputes the suffix using
    the ordinary model forward pass. Supplying a repair fraction explicitly opts
    into experimental selective repair and requires source-token provenance.
    Query tokens are always processed fresh, after history has been repaired.
    The source must be a contiguous cache, not a previously evicted session.
    Exactness additionally assumes it was computed exactly from old_ids; reusing
    an already approximate source does not restore its correctness.
    """
    if (not old_ids or not new_ids or not query_ids
            or original.model is not model
            or original.seen_tokens != len(old_ids)
            or original.positions != list(range(len(old_ids)))
            or len(original.cache.layers) != len(model.model.layers)
            or any(type(layer) is not DynamicLayer or layer.keys.shape[0] != 1
                   or layer.get_seq_length() != len(old_ids) for layer in original.cache.layers)):
        raise ValueError("requires nonempty history/query and a contiguous original model cache")
    if len(new_ids) + len(query_ids) > model.config.max_position_embeddings:
        raise ValueError("edited history and query exceed context limit")
    if fraction is not None:
        if mapping is None:
            raise ValueError("selective repair requires source-token mapping")
        cache, stats = repair_cache(model, original.cache, old_ids, new_ids, mapping,
                                    fraction=fraction)
        session = QwenKVSession(model, cache=cache, positions=list(range(len(new_ids))),
                                seen_tokens=len(new_ids), chunk_size=len(query_ids))
        stats["mode"] = "selective_repair"
        return session, stats, session.append(query_ids)
    common = 0
    for before, after in zip(old_ids, new_ids):
        if before != after:
            break
        common += 1
    session = QwenKVSession(model, chunk_size=len(new_ids) + len(query_ids))
    if common:
        session.cache = select_cache(original.cache, list(range(common)))
        session.positions, session.seen_tokens = list(range(common)), common
    logits = session.append(new_ids[common:] + query_ids)
    return session, {"mode": "exact_prefix", "history_tokens": len(new_ids),
                     "recomputed_tokens": len(new_ids) - common,
                     "reused_tokens": common}, logits


def repair_positions(mapping: list[int | None], fraction: float) -> list[int]:
    """Repair unmatched tokens plus a fixed budget ranked by distance after edits.

    Uses only the source alignment: no answer, future query, or freshly rebuilt
    reference cache. Mandatory new tokens may exceed the requested fraction.
    """
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("repair fraction must be between zero and one")
    mandatory = {i for i, old in enumerate(mapping) if old is None}
    candidates = []
    distance = len(mapping)
    edited = False
    for i, old in enumerate(mapping):
        previous = mapping[i - 1] if i else -1
        if old is None or previous is None or old != previous + 1:
            edited, distance = True, 0
        if edited and i not in mandatory:
            candidates.append((distance, i))
        distance += 1
    budget = max(len(mandatory), math.ceil(len(mapping) * fraction))
    chosen = mandatory | {i for _, i in sorted(candidates)[:budget - len(mandatory)]}
    # A full-repair control must include the unchanged prefix too.
    if fraction == 1:
        chosen = set(range(len(mapping)))
    return sorted(chosen)


@torch.inference_mode()
def repair_cache(model, original: DynamicCache, old_ids: list[int], new_ids: list[int],
                 mapping: list[int | None], *, fraction: float) -> tuple[DynamicCache, dict]:
    """Rephase surviving keys and recompute selected token states at every layer.

    The original cache is read-only. Mapping identifies the original source token
    for each new token; retokenized joins and new text must use None. Selecting
    every position is a correctness control equivalent to a fresh prefill, up to
    floating-point error. Partial selection has no exactness guarantee.
    """
    config = model.config
    if (config.model_type != "qwen2" or config.use_sliding_window
            or getattr(config, "rope_scaling", None) or model.training):
        raise ValueError("requires eval-mode Qwen2 with full attention and default RoPE")
    if not old_ids or not new_ids or len(new_ids) != len(mapping):
        raise ValueError("one mapping entry is required per new token")
    if max(len(old_ids), len(new_ids)) > config.max_position_embeddings:
        raise ValueError("context limit exceeded")
    mapped = [p for p in mapping if p is not None]
    if (any(type(p) is not int or p < 0 or p >= len(old_ids) for p in mapped)
            or mapped != sorted(set(mapped))
            or any(old is not None and old_ids[old] != new_ids[i] for i, old in enumerate(mapping))):
        raise ValueError("mapping must identify matching source tokens in increasing order")
    if (len(original.layers) != len(model.model.layers)
            or any(type(layer) is not DynamicLayer or layer.keys.shape[0] != 1
                   or layer.get_seq_length() != len(old_ids) for layer in original.layers)):
        raise ValueError("original cache must contain the aligned batch-one source")
    chosen = repair_positions(mapping, fraction)
    device = next(model.parameters()).device
    n = len(new_ids)
    selection = torch.tensor(chosen, dtype=torch.long, device=device)
    source = torch.tensor([p if p is not None else 0 for p in mapping], device=device)
    changed = torch.tensor([p is not None and p != i for i, p in enumerate(mapping)], device=device)
    delta = torch.arange(n, device=device) - source
    dummy = torch.zeros((1, n, 1), dtype=torch.float32, device=device)
    delta_cos, delta_sin = model.model.rotary_emb(dummy, delta[None])
    output = DynamicCache()
    if chosen:
        ids = torch.tensor([[new_ids[i] for i in chosen]], device=device)
        hidden = model.model.embed_tokens(ids)
        rope = model.model.rotary_emb(hidden, selection[None])
        mask = (torch.arange(n, device=device)[None, :] <= selection[:, None])[None, None]
    for index, layer in enumerate(model.model.layers):
        cached = original.layers[index]
        keys = cached.keys.index_select(-2, source)
        values = cached.values.index_select(-2, source)
        if len(chosen) != n:
            # Default RoPE is an orthogonal rotation. Rephase in FP32 to reduce
            # rounding, and leave identical-position keys bitwise untouched.
            k32 = keys.float()
            shifted = (k32 * delta_cos[:, None] + rotate_half(k32) * delta_sin[:, None]).to(keys.dtype)
            keys = torch.where(changed[None, None, :, None], shifted, keys)
        if chosen:
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = layer.self_attn
            shape = (1, len(chosen), -1, attention.head_dim)
            queries = attention.q_proj(normalized).view(shape).transpose(1, 2)
            new_keys = attention.k_proj(normalized).view(shape).transpose(1, 2)
            new_values = attention.v_proj(normalized).view(shape).transpose(1, 2)
            queries, new_keys = apply_rotary_pos_emb(queries, new_keys, *rope)
            keys.index_copy_(-2, selection, new_keys)
            values.index_copy_(-2, selection, new_values)
            # All missing entries are mandatory and have now been overwritten.
            attended = F.scaled_dot_product_attention(
                queries, repeat_kv(keys, attention.num_key_value_groups),
                repeat_kv(values, attention.num_key_value_groups),
                attn_mask=mask, dropout_p=0.0, scale=attention.scaling,
            )
            hidden = residual + attention.o_proj(attended.transpose(1, 2).reshape(1, len(chosen), -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        output.update(keys, values, index)
    return output, {"history_tokens": n, "recomputed_tokens": len(chosen),
                    "reused_tokens": n - len(chosen),
                    "mandatory_new_tokens": sum(p is None for p in mapping),
                    "selected_positions": chosen,
                    "recomputed_token_layers": len(chosen) * len(model.model.layers)}
