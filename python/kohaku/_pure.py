"""Pure-Python HDC implementation (no Rust required). Same math as the Rust core."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

DIMS = 10_000

# LCG constants — must match Rust src/hypervector.rs exactly.
_LCG_MUL: int = 6_364_136_223_846_793_005
_LCG_ADD: int = 1_442_695_040_888_963_407
_MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
# Initial state XOR applied to the seed before the first LCG step.
_SEED_XOR: int = 0xDEAD_BEEF_CAFE_BABE


def _lcg_next(state: int) -> tuple[int, int]:
    """LCG matching Rust: multiplier=6364136223846793005, inc=1442695040888963407, mod 2^64."""
    state = (state * _LCG_MUL + _LCG_ADD) & _MASK64
    return state, state


class HyperVector:
    """Bipolar (+1/-1) hypervector of DIMS dimensions."""
    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data.astype(np.int8)

    @classmethod
    def random(cls, dims: int = DIMS, seed: int = 42) -> "HyperVector":
        """Deterministic random bipolar vector, matching Rust LCG exactly.

        The Rust implementation XORs the seed with 0xDEAD_BEEF_CAFE_BABE before the
        first LCG step. The sign bit of each output value (bit 63) determines +1 vs -1.
        """
        state = (seed ^ _SEED_XOR) & _MASK64
        bits = np.empty(dims, dtype=np.int8)
        for i in range(dims):
            state = (state * _LCG_MUL + _LCG_ADD) & _MASK64
            bits[i] = np.int8(1) if (state >> 63) == 0 else np.int8(-1)
        return cls(bits)

    def bundle(self, others: list["HyperVector"]) -> "HyperVector":
        """Superposition: majority vote over self + others."""
        all_vecs = [self] + others
        matrix = np.stack([v.data for v in all_vecs], axis=0).astype(np.int32)
        summed = matrix.sum(axis=0)
        result = np.where(summed >= 0, np.int8(1), np.int8(-1)).astype(np.int8)
        return HyperVector(result)

    @classmethod
    def bundle_all(cls, vectors: list["HyperVector"]) -> "HyperVector":
        """Bundle a list of vectors (must be non-empty)."""
        if not vectors:
            raise ValueError("Cannot bundle empty list")
        return vectors[0].bundle(vectors[1:])

    def bind(self, other: "HyperVector") -> "HyperVector":
        """Element-wise multiply (XOR equiv for bipolar)."""
        return HyperVector(self.data * other.data)

    def permute(self, shift: int = 1) -> "HyperVector":
        """Circular left shift by `shift` positions."""
        return HyperVector(np.roll(self.data, -shift))

    def cosine_similarity(self, other: "HyperVector") -> float:
        """Cosine similarity in [-1, 1].

        For bipolar ±1 vectors |v|² = D, so cosine = dot(a, b) / D.
        This matches the Rust implementation exactly.
        """
        dot = float(np.dot(self.data.astype(np.float32), other.data.astype(np.float32)))
        norm_a = float(np.linalg.norm(self.data.astype(np.float32)))
        norm_b = float(np.linalg.norm(other.data.astype(np.float32)))
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return dot / (norm_a * norm_b)

    def hamming_distance(self, other: "HyperVector") -> float:
        """Fraction of differing components in [0, 1].

        Invariant: hamming = (1 - cosine) / 2 for bipolar ±1 vectors.
        """
        return float(np.sum(self.data != other.data)) / len(self.data)

    def __repr__(self) -> str:
        preview = " ".join(f"{x:+d}" for x in self.data[:8])
        return f"HyperVector([{preview} ...], dims={len(self.data)})"

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class MemoryEntry:
    id: int
    key: HyperVector
    value: HyperVector
    label: str
    timestamp: int


class EpisodicMemory:
    """Fixed-capacity episodic memory store with FIFO eviction."""

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("EpisodicMemory capacity must be > 0")
        self._capacity = capacity
        self._entries: list[MemoryEntry] = []
        self._next_id: int = 1
        self._timestamp: int = 1

    def store(self, key: HyperVector, value: HyperVector, label: str) -> int:
        """Store a key-value pair. Evicts oldest if at capacity. Returns entry ID."""
        if len(self._entries) >= self._capacity:
            self._entries.pop(0)
        entry = MemoryEntry(
            id=self._next_id,
            key=key,
            value=value,
            label=label,
            timestamp=self._timestamp,
        )
        self._entries.append(entry)
        self._next_id += 1
        self._timestamp += 1
        return entry.id

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._next_id = 1
        self._timestamp = 1
