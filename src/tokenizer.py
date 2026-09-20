"""Exact token counting for a configured model/tokenizer."""

from __future__ import annotations

import os


class TargetTokenizer:
    def __init__(self, model: str | None = None, encoding: str | None = None) -> None:
        self.model = model or os.getenv("TARGET_MODEL", "gpt-4.1-mini")
        self.encoding_name = encoding or os.getenv("TOKENIZER_ENCODING", "")
        self._encoding = None
        if self.model == "mock-byte-tokenizer":
            return
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("tiktoken is required for real target models; run: pip install -r requirements.txt") from exc
        if self.encoding_name:
            self._encoding = tiktoken.get_encoding(self.encoding_name)
        else:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("o200k_base")

    @property
    def name(self) -> str:
        return "utf8-bytes (mock only)" if self._encoding is None else self._encoding.name

    def encode(self, text: str) -> list[int]:
        if self._encoding is None:
            return list(text.encode("utf-8"))
        return self._encoding.encode(text)

    def count(self, text: str) -> int:
        return len(self.encode(text))
