"""Multilingual sentence embeddings with an on-disk cache.

LaBSE is the default because it is trained specifically for bitext mining -
aligning translation pairs across languages - which is exactly the judgement
required here. Lighter alternatives are configurable for fast iteration.

Embeddings are cached to disk keyed by (model, text) so that re-running an
experiment costs no recomputation. This matters: the encoder queries similarity
for the same phrases repeatedly.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODELS = {
    # Best quality for English-Chinese bitext equivalence (~1.8GB).
    "labse": "sentence-transformers/LaBSE",
    # ~470MB, much faster, slightly weaker cross-lingual alignment.
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}

_DEFAULT_CACHE = Path("data/cache/embeddings")


class EmbeddingModel:
    """Lazily-loaded sentence embedder with a persistent cache."""

    def __init__(
        self,
        model: str = "labse",
        cache_dir: str | Path = _DEFAULT_CACHE,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_id = MODELS.get(model, model)
        self.name = model
        self.batch_size = batch_size
        self._device = device
        self._model = None  # loaded on first use

        self.cache_dir = Path(cache_dir) / model.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "vectors.pkl"
        self._cache: dict[str, np.ndarray] = self._load_cache()
        self._dirty = False

    def _load_cache(self) -> dict[str, np.ndarray]:
        if self._cache_file.exists():
            try:
                with self._cache_file.open("rb") as fh:
                    return pickle.load(fh)
            except (pickle.PickleError, EOFError):  # pragma: no cover
                log.warning("embedding cache corrupt; rebuilding")
        return {}

    def save_cache(self) -> None:
        if not self._dirty:
            return
        tmp = self._cache_file.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(self._cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._cache_file)
        self._dirty = False

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s (first run downloads it)", self.model_id)
            self._model = SentenceTransformer(self.model_id, device=self._device)
        return self._model

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings, computing only cache misses."""
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        keys = [self._key(t) for t in texts]
        missing = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in self._cache]

        if missing:
            model = self._ensure_model()
            vecs = model.encode(
                [t for _, t in missing],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(missing) > 500,
            )
            for (i, _), v in zip(missing, vecs):
                self._cache[keys[i]] = v.astype(np.float32)
            self._dirty = True

        return np.vstack([self._cache[k] for k in keys])

    def similarity(self, a: list[str], b: list[str]) -> np.ndarray:
        """Row-wise cosine similarity between two equal-length lists."""
        if len(a) != len(b):
            raise ValueError("similarity() requires equal-length inputs")
        if not a:
            return np.zeros(0, dtype=np.float32)
        va, vb = self.encode(a), self.encode(b)
        # Vectors are already normalized, so the dot product is the cosine.
        return np.sum(va * vb, axis=1)

    def __enter__(self) -> EmbeddingModel:
        return self

    def __exit__(self, *exc) -> None:
        self.save_cache()
