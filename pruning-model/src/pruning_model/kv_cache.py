"""Experimental batch-one Qwen2 KV eviction; never re-encodes retained tokens.

Physical cache slots are compacted, but RoPE positions remain the original
monotonic positions. This is approximate inference, not an exact cache for a
rewritten prompt. Only full-attention Qwen2 DynamicCache is supported.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_bytes(cache: DynamicCache) -> int:
    return sum(t.numel() * t.element_size() for layer in cache.layers
               for t in (layer.keys, layer.values) if t is not None)


def select_cache(cache: DynamicCache, indices: list[int]) -> DynamicCache:
    """Copy selected slots in original order, preserving their K/V values."""
    if not indices or indices != sorted(set(indices)):
        raise ValueError("cache selection must be nonempty, unique and ordered")
    length = cache.get_seq_length()
    if indices[0] < 0 or indices[-1] >= length:
        raise ValueError("cache index out of bounds")
    if any(type(layer) is not DynamicLayer or layer.keys.shape[0] != 1
           or layer.get_seq_length() != length for layer in cache.layers):
        raise ValueError("only aligned batch-one full-attention DynamicLayer is supported")
    selected = DynamicCache()
    for i, layer in enumerate(cache.layers):
        index = torch.tensor(indices, dtype=torch.long, device=layer.keys.device)
        selected.update(layer.keys.index_select(-2, index),
                        layer.values.index_select(-2, index), i)
    return selected


@dataclass
class QwenKVSession:
    model: object
    chunk_size: int = 256
    cache: DynamicCache = field(default_factory=DynamicCache)
    positions: list[int] = field(default_factory=list)
    seen_tokens: int = 0
    forward_tokens: int = 0
    peak_cache_bytes: int = 0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        config = self.model.config
        if config.model_type != "qwen2" or config.use_sliding_window:
            raise ValueError("this prototype supports Qwen2 full attention only")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def append(self, ids: list[int]) -> torch.Tensor:
        if not ids:
            raise ValueError("append requires at least one token")
        if self.seen_tokens + len(ids) > self.model.config.max_position_embeddings:
            raise ValueError("original logical positions exceed the model context limit")
        for start in range(0, len(ids), self.chunk_size):
            part = ids[start:start + self.chunk_size]
            physical = len(self.positions)
            logical = self.seen_tokens
            result = self.model(
                input_ids=torch.tensor([part], device=self.device),
                past_key_values=self.cache,
                use_cache=True,
                # Slots determine causal masking; positions determine RoPE.
                cache_position=torch.arange(physical, physical + len(part), device=self.device),
                position_ids=torch.arange(logical, logical + len(part), device=self.device)[None],
                attention_mask=torch.ones((1, physical + len(part)), dtype=torch.long,
                                          device=self.device),
                logits_to_keep=1,
            )
            self.cache = result.past_key_values
            self.positions.extend(range(logical, logical + len(part)))
            self.seen_tokens += len(part)
            self.forward_tokens += len(part)
            self.peak_cache_bytes = max(self.peak_cache_bytes, cache_bytes(self.cache))
        return result.logits[0, -1]

    def evict(self, keep_positions: set[int]) -> dict:
        indices = [i for i, pos in enumerate(self.positions) if pos in keep_positions]
        if not indices:
            raise ValueError("cannot evict the entire cache")
        before_tokens, before_bytes = len(self.positions), cache_bytes(self.cache)
        synchronize(self.device)
        started = time.perf_counter()
        if len(indices) != before_tokens:
            self.cache = select_cache(self.cache, indices)
            self.positions = [self.positions[i] for i in indices]
        synchronize(self.device)
        event = {
            "seen_tokens": self.seen_tokens, "before_tokens": before_tokens,
            "after_tokens": len(self.positions), "before_bytes": before_bytes,
            "after_bytes": cache_bytes(self.cache),
            "eviction_ms": (time.perf_counter() - started) * 1000,
            "recomputed_tokens": 0,
        }
        self.events.append(event)
        return event

    def generate(self, logits: torch.Tensor, *, max_new_tokens: int,
                 eos_token_ids: set[int]) -> tuple[list[int], str]:
        output = []
        for step in range(max_new_tokens):
            token = int(logits.argmax())
            if token in eos_token_ids:
                return output, "stop"
            output.append(token)
            if step + 1 < max_new_tokens:
                logits = self.append([token])
        return output, "length"
