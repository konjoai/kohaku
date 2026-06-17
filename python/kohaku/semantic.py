"""Semantic encoding — project dense embeddings into HDC space.

The default text encoder (:func:`kohaku.encode_text`) bundles per-*token*
hypervectors, so similarity is essentially token overlap: *"the user enjoys
merlot"* barely matches *"User prefers Italian wine"*. This module adds an
**opt-in** semantic path: take a dense embedding (sentence-transformers,
OpenAI, anything) and project it into a bipolar hypervector via a fixed random
projection. This is SimHash — sign-of-random-projection — which approximately
preserves cosine similarity, so embeddings that are close in dense space land
close in HDC space.

Nothing here is imported by default and there is no hard dependency on any
embedding library: :class:`EmbeddingEncoder` accepts any ``embed_fn`` callable
(``str -> 1-D float array``). The sentence-transformers path is a lazily-loaded
convenience — absent the package, instantiating without an ``embed_fn`` raises
a clear :class:`ImportError`, never an import-time crash.

    >>> from kohaku.semantic import EmbeddingEncoder
    >>> from kohaku import Memory
    >>> enc = EmbeddingEncoder(model_name="all-MiniLM-L6-v2")   # doctest: +SKIP
    >>> mem = Memory(encoder=enc)                               # doctest: +SKIP
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from kohaku._pure import DIMS, HyperVector

logger = logging.getLogger(__name__)

# Fixed seed for the random projection so the same embedding maps to the same
# hypervector across processes — required for save/load and cache parity.
DEFAULT_PROJECTION_SEED = 0x5EED_C0DE

EmbedFn = Callable[[str], "np.ndarray"]

# Cache projection matrices keyed by (embedding_dim, dims, seed). Building a
# 384×10000 matrix is ~15 MB and pure overhead to recompute per call.
_PROJECTION_CACHE: Dict[Tuple[int, int, int], "np.ndarray"] = {}


def _projection_matrix(embedding_dim: int, dims: int, seed: int) -> "np.ndarray":
    """Return (and cache) a deterministic ``(embedding_dim, dims)`` Gaussian matrix."""
    key = (embedding_dim, dims, seed)
    cached = _PROJECTION_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(seed)
        cached = rng.standard_normal((embedding_dim, dims)).astype(np.float32)
        _PROJECTION_CACHE[key] = cached
    return cached


def project_to_hypervector(
    embedding: "np.ndarray",
    dims: int = DIMS,
    *,
    seed: int = DEFAULT_PROJECTION_SEED,
) -> HyperVector:
    """Project a dense embedding into a bipolar hypervector via SimHash.

    ``hv[i] = sign(embedding · R[:, i])`` for a fixed Gaussian matrix ``R``.
    The probability that two embeddings agree on a given bit is
    ``1 - angle/π`` — so HDC cosine tracks dense cosine. Zero projections map
    to ``+1`` to keep the output strictly bipolar.
    """
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if emb.size == 0:
        raise ValueError("embedding must be non-empty")
    if not np.all(np.isfinite(emb)):
        raise ValueError("embedding contains non-finite values")
    matrix = _projection_matrix(emb.shape[0], dims, seed)
    projected = emb @ matrix
    components = np.where(projected >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
    return HyperVector(components)


class EmbeddingEncoder:
    """Encode text into HDC space via a dense embedding + random projection.

    Parameters
    ----------
    embed_fn:
        Callable ``str -> 1-D float array``. When ``None``, a
        sentence-transformers model named ``model_name`` is lazily loaded the
        first time text is encoded.
    model_name:
        sentence-transformers model id (only used when ``embed_fn`` is None).
    dims:
        Output hypervector dimensionality.
    seed:
        Random-projection seed (fix it to keep encodings stable across runs).

    The encoder is callable, so it drops straight into
    :class:`kohaku.Memory(encoder=...)` or anywhere a ``str -> HyperVector``
    function is expected.
    """

    def __init__(
        self,
        *,
        embed_fn: Optional[EmbedFn] = None,
        model_name: str = "all-MiniLM-L6-v2",
        dims: int = DIMS,
        seed: int = DEFAULT_PROJECTION_SEED,
    ) -> None:
        self._embed_fn = embed_fn
        self._model_name = model_name
        self._model = None  # lazily constructed sentence-transformers model
        self.dims = dims
        self.seed = seed

    def _ensure_embed_fn(self) -> EmbedFn:
        if self._embed_fn is not None:
            return self._embed_fn
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "EmbeddingEncoder needs either an `embed_fn` or the optional "
                "`sentence-transformers` package. Install it with "
                "`pip install kohaku[semantic]` or pass `embed_fn=...`."
            ) from exc
        logger.info("loading sentence-transformers model %r", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._embed_fn = lambda text: self._model.encode(text)  # type: ignore[union-attr]
        return self._embed_fn

    def embed(self, text: str) -> "np.ndarray":
        """Return the raw dense embedding for ``text``."""
        return np.asarray(self._ensure_embed_fn()(text), dtype=np.float32).reshape(-1)

    def encode(self, text: str) -> HyperVector:
        """Encode ``text`` into a bipolar hypervector (semantic SimHash)."""
        if not text or not text.strip():
            raise ValueError("cannot encode empty text")
        return project_to_hypervector(self.embed(text), self.dims, seed=self.seed)

    def __call__(self, text: str) -> HyperVector:
        return self.encode(text)
