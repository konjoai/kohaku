"""Analogical memory — relational reasoning over memory via VSA binding algebra.

This is the capability embedding / vector-DB memory cannot do: *algebra* over
what you remember, with no model call and no extra storage. Each record is a
superposition (bundle) of bound ``(attribute, value)`` pairs over a shared,
deterministic symbol vocabulary::

    record("USA") = bundle( bind(country, usa), bind(currency, dollar),
                            bind(capital, washington), ... )

Because binding (element-wise multiply of bipolar ±1 vectors) is its own inverse
and distributes over the bundle, two operations fall out for free:

* **Attribute query** — :meth:`AnalogicalMemory.get` ("USA", "currency") unbinds
  the attribute from the record (``record ⊛ currency``) and cleans the noisy
  result up against the value codebook → ``"dollar"``.
* **Analogical transfer** — :meth:`AnalogicalMemory.analogy` ("USA", "Mexico",
  "dollar") builds the USA→Mexico mapping (``record(USA) ⊛ record(Mexico)``),
  applies it to ``dollar``, and cleans up → ``"peso"``. The classic "What is the
  dollar of Mexico?" (Kanerva 2010), now over an agent's own memory.

Cleanup uses the packed :class:`~kohaku.RetrievalIndex`. Symbols are deterministic
random bipolar vectors seeded by a stable hash of the symbol text — orthogonal by
construction. This layer is intentionally *not* the lexical/semantic encoder:
encoders make symbols that share words similar, which would corrupt the binding
algebra. Recall is exact-cosine cleanup, so every answer carries a confidence;
analogical transfer is noisier than direct attribute lookup (see
``benchmarks/bench_analogy.py`` for the capacity curve).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from kohaku._index import RetrievalIndex
from kohaku._pure import DIMS, HyperVector
from kohaku.extraction import Triple, extract_triples


def _stable_seed(symbol: str) -> int:
    """Process-stable 64-bit seed for a symbol (unlike the salted builtin hash)."""
    return int.from_bytes(
        hashlib.blake2b(symbol.encode("utf-8"), digest_size=8).digest(), "big"
    )


@dataclass(frozen=True)
class AnalogyResult:
    """The cleaned-up answer to an attribute or analogy query.

    ``value`` is the best-matching codebook entry; ``confidence`` is its cosine
    to the (noisy) query vector. ``ranked`` holds the top candidates so callers
    can inspect runner-ups and the decision ``margin``. An empty result
    (``value == ""``) is falsy.
    """

    value: str
    confidence: float
    ranked: Tuple[Tuple[str, float], ...] = ()

    @property
    def margin(self) -> float:
        """Confidence gap between the top and second candidate (separation)."""
        if len(self.ranked) >= 2:
            return round(self.ranked[0][1] - self.ranked[1][1], 4)
        return self.confidence

    def __bool__(self) -> bool:
        return bool(self.value)


class AnalogicalMemory:
    """Relational memory supporting attribute lookup and analogical transfer.

    Parameters
    ----------
    dims:
        Hypervector dimensionality. Higher dims raise the number of
        ``(attribute, value)`` pairs a record holds before cleanup degrades
        (capacity scales roughly linearly with ``dims``).
    """

    def __init__(self, dims: int = DIMS) -> None:
        if dims <= 0:
            raise ValueError("dims must be > 0")
        self.dims = dims
        self._records: Dict[str, HyperVector] = {}
        self._fields: Dict[str, Dict[str, str]] = {}
        self._symbols: Dict[str, HyperVector] = {}
        self._values: Dict[str, HyperVector] = {}  # cleanup codebook
        self._value_order: List[str] = []
        self._index: RetrievalIndex | None = None  # cached cleanup index

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, name: str) -> bool:
        return name in self._records

    def records(self) -> List[str]:
        """Names of all stored records."""
        return list(self._records)

    def fields(self, name: str) -> Dict[str, str]:
        """The original ``{attribute: value}`` map for a record (a copy)."""
        self._require(name)
        return dict(self._fields[name])

    def to_dict(self) -> Dict[str, object]:
        """Serialise to the field maps; vectors re-derive deterministically."""
        return {
            "dims": self.dims,
            "records": {n: dict(f) for n, f in self._fields.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AnalogicalMemory":
        """Rebuild from :meth:`to_dict` output (symbols are hash-deterministic)."""
        mem = cls(dims=int(data.get("dims", DIMS)))  # type: ignore[arg-type]
        for name, fields in dict(data.get("records", {})).items():  # type: ignore[arg-type]
            mem.add_record(name, fields)
        return mem

    # ── construction ─────────────────────────────────────────────────────────
    def add_record(self, name: str, fields: Mapping[str, str]) -> None:
        """Store ``name`` as a superposition of its bound attribute/value pairs.

        Re-adding a name replaces it. Every value is registered in the cleanup
        codebook, so it becomes a possible answer to future queries.
        """
        if not fields:
            raise ValueError("record must have at least one field")
        pairs: List[HyperVector] = []
        normalised: Dict[str, str] = {}
        for attr, value in fields.items():
            a, v = str(attr), str(value)
            normalised[a] = v
            self._register_value(v)
            pairs.append(self._symbol(a).bind(self._symbol(v)))
        self._records[name] = HyperVector.bundle_all(pairs)
        self._fields[name] = normalised

    def learn(self, text: str) -> List[Triple]:
        """Extract ``(subject, attribute, value)`` triples from free text and
        fold them into records keyed by subject — so analogical reasoning works
        on what the agent *read*, not just hand-built records.

        New attributes merge into an existing subject's record (later mentions
        overwrite the same attribute). Returns the triples learned, which is
        ``[]`` when nothing parsed — the extractor never fabricates structure.
        See :mod:`kohaku.extraction` for the patterns recognised.
        """
        triples = extract_triples(text)
        by_subject: Dict[str, Dict[str, str]] = {}
        for triple in triples:
            by_subject.setdefault(triple.subject, {})[triple.attribute] = triple.value
        for subject, fields in by_subject.items():
            merged = {**self._fields.get(subject, {}), **fields}
            self.add_record(subject, merged)
        return triples

    # ── queries ──────────────────────────────────────────────────────────────
    def get(self, name: str, attribute: str, *, top_k: int = 3) -> AnalogyResult:
        """Recover the value of ``attribute`` in record ``name`` (unbind + cleanup).

        e.g. ``get("USA", "currency") -> AnalogyResult(value="dollar", ...)``.
        """
        self._require(name)
        noisy = self._records[name].bind(self._symbol(str(attribute)))
        return self._cleanup(noisy, top_k=top_k)

    def analogy(
        self, source: str, target: str, value: str, *, top_k: int = 3
    ) -> AnalogyResult:
        """Analogical transfer: "the ``value`` of ``source`` is to ``target`` as…".

        ``value`` should be an attribute value of ``source``; the result is the
        corresponding value in ``target`` for the same attribute. The "dollar of
        Mexico" construction: ``analogy("USA", "Mexico", "dollar") -> "peso"``.
        """
        self._require(source)
        self._require(target)
        mapping = self._records[source].bind(self._records[target])
        noisy = self._symbol(str(value)).bind(mapping)
        # Exclude the probe itself — the analog is a *different* value.
        return self._cleanup(noisy, top_k=top_k, exclude=(str(value),))

    # ── internals ──────────────────────────────────────────────────────────
    def _require(self, name: str) -> None:
        if name not in self._records:
            raise ValueError(f"unknown record {name!r}")

    def _symbol(self, text: str) -> HyperVector:
        hv = self._symbols.get(text)
        if hv is None:
            rng = np.random.default_rng(_stable_seed(text))
            bits = np.where(rng.random(self.dims) > 0.5, np.int8(1), np.int8(-1))
            hv = HyperVector(bits.astype(np.int8))
            self._symbols[text] = hv
        return hv

    def _register_value(self, text: str) -> None:
        if text not in self._values:
            self._values[text] = self._symbol(text)
            self._value_order.append(text)
            self._index = None  # codebook changed → rebuild on next cleanup

    def _cleanup_index(self) -> RetrievalIndex:
        if self._index is None:
            if self._value_order:
                keys = np.stack([self._values[t].data for t in self._value_order])
            else:
                keys = np.empty((0, 0), dtype=np.int8)
            self._index = RetrievalIndex(keys)
        return self._index

    def _cleanup(
        self, hv: HyperVector, *, top_k: int, exclude: Sequence[str] = ()
    ) -> AnalogyResult:
        order = self._value_order
        if not order:
            return AnalogyResult("", 0.0, ())
        ranked = self._cleanup_index().topk(hv.data, len(order))
        skip = set(exclude)
        out = [(order[i], float(s)) for i, s in ranked if order[i] not in skip]
        if not out:
            return AnalogyResult("", 0.0, ())
        head = [(t, round(s, 4)) for t, s in out[: max(1, top_k)]]
        return AnalogyResult(head[0][0], head[0][1], tuple(head))
