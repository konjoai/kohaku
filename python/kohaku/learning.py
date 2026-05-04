"""Online HDC learning — item memory with prototype updates from feedback.

Item memory maps a label → a prototype hypervector that represents the centroid
of all examples seen for that label. Prototypes are maintained as float
accumulators (running sums of signed examples) and binarized on demand:

    prototype(L) = sign( Σ_i  s_i · v_i )       s_i ∈ {+1, -1}

where each ``v_i`` is an example bipolar vector and ``s_i`` is the supervision
sign (+1 = "yes this is L", −1 = "no this is not L"). This is the classic HDC
online-learning rule (see Kanerva 2009, Rachkovskij 2001) — equivalent to a
single-prototype perceptron in 10 000-D but with exact bipolar arithmetic.

Properties
----------
* Each accumulator is float32 — sign() yields a bipolar HyperVector.
* No floating-point precision concerns up to ~2²⁴ examples per label
  (well above any realistic use).
* Prototype always remains in {+1, −1}^D — no precision drift after binarization.
* New labels seed their accumulator with the first example.

Used by :class:`MemorySystem` to maintain a semantic store that survives
episodic decay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

from kohaku._pure import DIMS, HyperVector
from kohaku._query import RetrievalResult


@dataclass
class Prototype:
    """A learned prototype for one class label."""
    label: str
    accumulator: np.ndarray  # float32, shape (dims,) — running signed sum
    n_examples: int

    @property
    def vector(self) -> HyperVector:
        """The current binarized prototype: sign(accumulator), ties → +1."""
        bits = np.where(self.accumulator >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
        return HyperVector(bits)

    def __len__(self) -> int:
        return int(self.accumulator.shape[0])


class ItemMemory:
    """Online-learnable map of *label* → *prototype hypervector*.

    Prototypes update with every call to :meth:`add` or :meth:`update` and stay
    binarized to ±1 by construction. Use :meth:`predict` for top-k classification.
    """

    def __init__(self, dims: int = DIMS) -> None:
        if dims <= 0:
            raise ValueError("dims must be > 0")
        self._dims = int(dims)
        self._protos: dict[str, Prototype] = {}

    # ── basic accessors ────────────────────────────────────────────────────
    @property
    def dims(self) -> int:
        return self._dims

    def __len__(self) -> int:
        return len(self._protos)

    def __contains__(self, label: str) -> bool:
        return label in self._protos

    def labels(self) -> List[str]:
        return list(self._protos.keys())

    def get(self, label: str) -> Optional[Prototype]:
        return self._protos.get(label)

    def clear(self) -> None:
        self._protos.clear()

    # ── learning ───────────────────────────────────────────────────────────
    def add(self, label: str, vector: HyperVector, weight: float = 1.0) -> Prototype:
        """Register *vector* as a positive example of *label*. Equivalent to
        ``update(label, vector, sign=+1, weight=weight)`` but creates the
        prototype if it does not exist yet."""
        return self.update(label, vector, sign=+1, weight=weight)

    def update(
        self,
        label: str,
        vector: HyperVector,
        sign: int = +1,
        weight: float = 1.0,
    ) -> Prototype:
        """Move *label*'s prototype toward (sign=+1) or away from (sign=−1) *vector*.

        Parameters
        ----------
        label:
            Class label.
        vector:
            Bipolar hypervector example.
        sign:
            +1 to reinforce, −1 to suppress (negative feedback).
        weight:
            Positive scalar weight on this update (e.g. learning rate). The
            prototype is binarized after each update so weight only affects
            relative magnitudes inside the accumulator, never the final ±1.

        Returns
        -------
        Prototype
            The (possibly new) prototype after the update.
        """
        if sign not in (+1, -1):
            raise ValueError("sign must be +1 or -1")
        if weight <= 0:
            raise ValueError("weight must be > 0")
        if len(vector) != self._dims:
            raise ValueError(
                f"vector dims mismatch: expected {self._dims}, got {len(vector)}"
            )

        delta = vector.data.astype(np.float32) * (sign * float(weight))
        proto = self._protos.get(label)
        if proto is None:
            proto = Prototype(
                label=label,
                accumulator=delta.copy(),
                n_examples=1,
            )
            self._protos[label] = proto
        else:
            proto.accumulator += delta
            proto.n_examples += 1
        return proto

    def train_from_feedback(
        self,
        label: str,
        vector: HyperVector,
        correct: bool,
        weight: float = 1.0,
    ) -> Prototype:
        """Convenience wrapper: ``correct=True`` → reinforce, ``False`` → suppress."""
        return self.update(label, vector, sign=+1 if correct else -1, weight=weight)

    # ── inference ──────────────────────────────────────────────────────────
    def predict(self, vector: HyperVector, top_k: int = 1) -> List[RetrievalResult]:
        """Return the *top_k* closest prototypes to *vector* by cosine similarity.

        Empty :class:`ItemMemory` returns an empty list. Result entries reuse the
        :class:`RetrievalResult` dataclass; ``entry_id`` is the index of the
        prototype in insertion order, ``label`` is the prototype's label, and
        ``value`` is the prototype's current binarized vector.
        """
        if top_k <= 0 or not self._protos:
            return []
        if len(vector) != self._dims:
            raise ValueError(
                f"vector dims mismatch: expected {self._dims}, got {len(vector)}"
            )
        scored: List[RetrievalResult] = []
        for idx, (label, proto) in enumerate(self._protos.items()):
            proto_vec = proto.vector
            sim = proto_vec.cosine_similarity(vector)
            scored.append(
                RetrievalResult(
                    entry_id=idx,
                    label=label,
                    similarity=sim,
                    value=proto_vec,
                )
            )
        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:top_k]
