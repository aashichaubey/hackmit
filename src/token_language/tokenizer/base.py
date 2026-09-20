"""Tokenizer abstraction.

The entire project optimizes against *LLM input token count*, never character
or word count. Every token number that appears in a result must be traceable to
a concrete `Tokenizer` implementation via its `id`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache


class Tokenizer(ABC):
    """Minimal interface every tokenizer backend must satisfy."""

    #: Stable identifier recorded alongside every measurement.
    id: str

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return the token strings for `text`."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in `text`.

        Implementations should make this cheaper than `len(tokenize(text))`
        where the backend allows it.
        """

    def count_many(self, texts: list[str]) -> list[int]:
        """Batch count. Backends with real batch APIs should override."""
        return [self.count_tokens(t) for t in texts]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} id={self.id!r}>"


class CachedTokenizer(Tokenizer):
    """Mixin adding an in-process LRU cache over `count_tokens`.

    Phrase mining re-counts the same short strings many thousands of times, so
    this is a large win and keeps the pipeline free of redundant work.
    """

    def __init__(self, cache_size: int = 200_000) -> None:
        self._cached_count = lru_cache(maxsize=cache_size)(self._count_uncached)

    @abstractmethod
    def _count_uncached(self, text: str) -> int: ...

    def count_tokens(self, text: str) -> int:
        return self._cached_count(text)

    def cache_info(self):  # pragma: no cover - debug aid
        return self._cached_count.cache_info()
