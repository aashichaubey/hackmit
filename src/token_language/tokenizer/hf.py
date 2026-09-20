"""HuggingFace `tokenizers` backend.

This is the backend used for the primary target model (DeepSeek V4 Flash),
whose `tokenizer.json` is published publicly. Loading it locally means every
token count in this project is exact, free, and reproducible offline - no API
calls and no estimation.
"""

from __future__ import annotations

import os
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizerImpl

from .base import CachedTokenizer

# Target model -> HuggingFace repo that publishes a compatible tokenizer.json.
KNOWN_TOKENIZERS: dict[str, str] = {
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
}

_DEFAULT_CACHE = Path("data/raw/tokenizers")


class HFTokenizer(CachedTokenizer):
    """Wraps a `tokenizer.json` file.

    Note that we deliberately do *not* add special/BOS tokens when counting.
    We are measuring the marginal cost of a piece of *content*, and chat
    templates add a fixed per-message overhead that is identical for the
    English and Mandarin variants. Including it would dilute the measured
    difference without changing which side wins.
    """

    def __init__(self, tokenizer_path: str | Path, tokenizer_id: str) -> None:
        super().__init__()
        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(
                f"tokenizer file not found: {path}. "
                "Run `python -m scripts.fetch_tokenizer` to download it."
            )
        self._tk = HFTokenizerImpl.from_file(str(path))
        self._tk.no_truncation()
        self._tk.no_padding()
        self.id = tokenizer_id

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        cache_dir: str | Path = _DEFAULT_CACHE,
        repo: str | None = None,
    ) -> HFTokenizer:
        """Load `model`, downloading its tokenizer.json on first use."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / f"{model}.tokenizer.json"

        if not local.exists():
            repo_id = repo or KNOWN_TOKENIZERS.get(model)
            if repo_id is None:
                raise KeyError(
                    f"unknown tokenizer {model!r}; pass `repo=` explicitly or add it "
                    f"to KNOWN_TOKENIZERS. Known: {sorted(KNOWN_TOKENIZERS)}"
                )
            _download(repo_id, local)

        return cls(local, tokenizer_id=model)

    def tokenize(self, text: str) -> list[str]:
        return self._tk.encode(text, add_special_tokens=False).tokens

    def _count_uncached(self, text: str) -> int:
        return len(self._tk.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: list[str]) -> list[int]:
        """Use the Rust batch encoder - materially faster for corpus-scale work."""
        if not texts:
            return []
        encoded = self._tk.encode_batch(texts, add_special_tokens=False)
        return [len(e.ids) for e in encoded]


def _download(repo_id: str, dest: Path) -> None:
    """Fetch tokenizer.json from the HF CDN using stdlib only."""
    import urllib.request

    url = f"https://huggingface.co/{repo_id}/resolve/main/tokenizer.json"
    req = urllib.request.Request(url, headers={"User-Agent": "token-language/0.1"})
    if token := os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")

    tmp = dest.with_suffix(".tmp")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
    tmp.replace(dest)  # atomic: never leave a half-written tokenizer behind
