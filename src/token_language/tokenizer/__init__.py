"""Tokenizer registry.

Config selects a backend by `provider`, so experiments can swap tokenizers
without touching pipeline code:

    tokenizer:
      provider: hf
      model: deepseek-v4-flash
"""

from __future__ import annotations

from .base import CachedTokenizer, Tokenizer
from .hf import KNOWN_TOKENIZERS, HFTokenizer
from .openai import ENCODINGS, TiktokenTokenizer

__all__ = [
    "Tokenizer",
    "CachedTokenizer",
    "HFTokenizer",
    "TiktokenTokenizer",
    "get_tokenizer",
    "KNOWN_TOKENIZERS",
    "ENCODINGS",
]


def get_tokenizer(provider: str, model: str, **kwargs) -> Tokenizer:
    """Build a tokenizer from a config-style (provider, model) pair."""
    provider = provider.lower()
    if provider in ("hf", "huggingface", "deepseek"):
        return HFTokenizer.from_pretrained(model, **kwargs)
    if provider in ("openai", "tiktoken"):
        return TiktokenTokenizer(model)
    raise ValueError(
        f"unknown tokenizer provider {provider!r}; expected one of: hf, openai"
    )
