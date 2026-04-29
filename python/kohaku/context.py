"""Context window memory manager — sliding-window episodic store sized to an LLM context limit."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kohaku._pure import (
    HyperVector,
    EpisodicMemory,
    DIMS,
    _LCG_MUL,
    _LCG_ADD,
    _MASK64,
    _SEED_XOR,
)


# ---------------------------------------------------------------------------
# Deterministic text → HyperVector encoding
# ---------------------------------------------------------------------------

def _hash_str(s: str) -> int:
    """Stable, deterministic 64-bit hash of a string using the same LCG seed path."""
    h: int = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & _MASK64
    return h


def _encode_word(word: str, dims: int) -> HyperVector:
    """Deterministically encode a single word as a bipolar hypervector."""
    seed = _hash_str(word) & _MASK64
    # Replicate HyperVector.random LCG with XOR preamble
    state = (seed ^ _SEED_XOR) & _MASK64
    bits = np.empty(dims, dtype=np.int8)
    for i in range(dims):
        state = (state * _LCG_MUL + _LCG_ADD) & _MASK64
        bits[i] = np.int8(1) if (state >> 63) == 0 else np.int8(-1)
    return HyperVector(bits)


def _encode_text_to_hv(text: str, dims: int = DIMS) -> HyperVector:
    """Encode text as a hypervector: split into words, encode each, bundle with majority vote.

    The encoding is fully deterministic — same text always produces the same hypervector.
    Single-word (or empty-after-split) inputs degenerate gracefully to the word's own vector
    or a zero-seeded vector respectively.
    """
    words = text.split()
    if not words:
        # Return a deterministic fallback for empty text
        return HyperVector.random(dims, seed=0)
    word_vecs = [_encode_word(w, dims) for w in words]
    if len(word_vecs) == 1:
        return word_vecs[0]
    return HyperVector.bundle_all(word_vecs)


# ---------------------------------------------------------------------------
# ContextConfig + ContextMemoryManager
# ---------------------------------------------------------------------------

@dataclass
class ContextConfig:
    """Configuration for the context-window memory manager."""
    max_tokens: int = 4096         # LLM context window size
    tokens_per_entry: int = 50     # estimated tokens per memory entry
    top_k: int = 5                 # memories to inject per query
    similarity_threshold: float = 0.1


class ContextMemoryManager:
    """Sliding-window episodic store sized to fit inside an LLM context limit.

    Encodes text keys as deterministic hypervectors using the same LCG algorithm as the
    core HDC engine.  Internally wraps ``EpisodicMemory`` with capacity
    ``max_tokens // tokens_per_entry``.
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config: ContextConfig = config or ContextConfig()
        cap = max(1, self.config.max_tokens // self.config.tokens_per_entry)
        self._memory: EpisodicMemory = EpisodicMemory(capacity=cap)
        # Store raw (label, value) strings for context block construction
        self._labels: list[str] = []
        self._values: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, key: str, value: str, label: str = "") -> None:
        """Encode *key* as a hypervector and store *value*+*label*.

        The value hypervector also encodes the value text so similarity queries on the
        key side work correctly.  Raw strings are maintained in a parallel list for
        ``build_context_block``.
        """
        key_hv = _encode_text_to_hv(key)
        value_hv = _encode_text_to_hv(value) if value else HyperVector.random(dims=DIMS, seed=0)
        if len(self._labels) >= self._memory._capacity:
            # Mirror the FIFO eviction happening inside EpisodicMemory
            self._labels.pop(0)
            self._values.pop(0)
        self._labels.append(label)
        self._values.append(value)
        self._memory.store(key_hv, value_hv, label)

    def retrieve(
        self, query_text: str, top_k: int | None = None
    ) -> list[tuple[str, str, float]]:
        """Return list of (label, value, similarity) sorted descending.

        Parameters
        ----------
        query_text:
            The text used to form the query hypervector.
        top_k:
            Number of results to return.  Defaults to ``config.top_k``.
        """
        k = top_k if top_k is not None else self.config.top_k
        if self._memory.is_empty:
            return []
        query_hv = _encode_text_to_hv(query_text)
        entries = self._memory.entries()
        scored: list[tuple[float, int]] = [
            (e.key.cosine_similarity(query_hv), idx)
            for idx, e in enumerate(entries)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[tuple[str, str, float]] = []
        for sim, idx in scored[:k]:
            if sim >= self.config.similarity_threshold:
                entry = entries[idx]
                # Map entry index to raw string storage.  The raw lists are kept in
                # insertion order matching EpisodicMemory._entries, so the index is valid.
                raw_label = self._labels[idx] if idx < len(self._labels) else entry.label
                raw_value = self._values[idx] if idx < len(self._values) else ""
                results.append((raw_label, raw_value, sim))
        return results

    def build_context_block(self, query_text: str) -> str:
        """Return a formatted string ready to prepend to an LLM prompt.

        Format::

            Relevant memories:
            - [label]: value
            - [label]: value
            ...

        Returns an empty string when no memories exceed the similarity threshold.
        """
        hits = self.retrieve(query_text)
        if not hits:
            return ""
        lines = ["Relevant memories:"]
        for label, value, _sim in hits:
            lines.append(f"- [{label}]: {value}")
        return "\n".join(lines)

    def capacity(self) -> int:
        """Maximum number of entries this manager can hold."""
        return self._memory._capacity

    def utilization(self) -> float:
        """Fraction of capacity currently used, in [0.0, 1.0]."""
        if self._memory._capacity == 0:
            return 0.0
        return len(self._memory) / self._memory._capacity
