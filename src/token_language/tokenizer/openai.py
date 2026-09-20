"""tiktoken backend.

Not the primary target, but essential as a *control*: comparing DeepSeek's
tokenizer against o200k_base/cl100k_base is what answers research question #12
(does the optimal representation differ between tokenizers?). Early probing
showed it does - dramatically - so this backend stays first-class.
"""

from __future__ import annotations

import tiktoken

from .base import CachedTokenizer

# Friendly name -> tiktoken encoding.
ENCODINGS: dict[str, str] = {
    "o200k_base": "o200k_base",  # GPT-4o / GPT-4.1 / GPT-5 family
    "cl100k_base": "cl100k_base",  # GPT-4 / GPT-3.5-turbo
}


class TiktokenTokenizer(CachedTokenizer):
    def __init__(self, encoding: str = "o200k_base") -> None:
        super().__init__()
        self._enc = tiktoken.get_encoding(ENCODINGS.get(encoding, encoding))
        self.id = encoding

    @classmethod
    def for_model(cls, model: str) -> TiktokenTokenizer:
        """Resolve an OpenAI model name to its encoding."""
        enc = tiktoken.encoding_for_model(model)
        inst = cls.__new__(cls)
        CachedTokenizer.__init__(inst)
        inst._enc = enc
        inst.id = enc.name
        return inst

    def tokenize(self, text: str) -> list[str]:
        return [
            self._enc.decode_single_token_bytes(t).decode("utf-8", errors="replace")
            for t in self._enc.encode(text)
        ]

    def _count_uncached(self, text: str) -> int:
        return len(self._enc.encode(text))
