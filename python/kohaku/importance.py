"""Auto importance scoring — derive ``MemoryMetadata.importance`` from
observed behaviour rather than the user's initial estimate.

Four signals contribute, each normalised to [0, 1]:

* **Access frequency** — ``min(1.0, reinforcement_count / max_freq)``.
  Memories that get retrieved often are more important than ones that
  never get queried.
* **Recency** — Ebbinghaus weight ``0.5 ** (age_days / half_life)``.
  Fresh memories rank higher.
* **Uniqueness** — ``1 - max_cosine_to_other_memories``. A memory that
  duplicates many others contributes less than a memory that stands
  alone.
* **Provenance depth** — ``log1p(children_count) / log1p(max_children)``.
  Memories that have spawned downstream lineage (via consolidation,
  inference) are more important than terminal leaves.

The four signals are blended with configurable weights that default to
equal contribution (0.25 each). The result replaces — or, with
``blend_alpha < 1.0``, exponentially-smooths — the existing ``importance``
field. ``ImportanceScorer.compute()`` is read-only and returns the
proposed scores. ``ImportanceScorer.apply()`` writes them back to the
live :class:`EnrichedMemoryStore`.

A *dry-run* call returns the per-component breakdown so callers can audit
the ranking before committing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from kohaku._index import index_over
from kohaku.enriched import EnrichedMemoryStore

logger = logging.getLogger(__name__)


DEFAULT_HALF_LIFE_DAYS: float = 30.0
DEFAULT_FREQ_CAP: int = 20

# Per-signal weights; the dataclass validator enforces sum ≈ 1.
DEFAULT_WEIGHTS = {
    "frequency": 0.25,
    "recency": 0.25,
    "uniqueness": 0.25,
    "depth": 0.25,
}


@dataclass(frozen=True)
class ImportanceBreakdown:
    """Per-component contribution for one memory."""

    entry_id: int
    frequency: float
    recency: float
    uniqueness: float
    depth: float
    composite: float
    importance_before: float
    importance_after: float

    def to_dict(self) -> dict:
        return {
            "entry_id": int(self.entry_id),
            "frequency": round(float(self.frequency), 4),
            "recency": round(float(self.recency), 4),
            "uniqueness": round(float(self.uniqueness), 4),
            "depth": round(float(self.depth), 4),
            "composite": round(float(self.composite), 4),
            "importance_before": round(float(self.importance_before), 4),
            "importance_after": round(float(self.importance_after), 4),
        }


@dataclass(frozen=True)
class RescoreReport:
    """Aggregate output of a rescore pass."""

    total_memories: int
    updated: int
    skipped: int
    dry_run: bool
    blend_alpha: float
    weights: Dict[str, float]
    breakdowns: List[ImportanceBreakdown] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_memories": int(self.total_memories),
            "updated": int(self.updated),
            "skipped": int(self.skipped),
            "dry_run": bool(self.dry_run),
            "blend_alpha": round(float(self.blend_alpha), 4),
            "weights": {k: round(float(v), 4) for k, v in self.weights.items()},
            "breakdowns": [b.to_dict() for b in self.breakdowns],
        }


class ImportanceScorer:
    """Auto-importance computation.

    Parameters
    ----------
    store:
        Live :class:`EnrichedMemoryStore`.
    provenance:
        Optional :class:`kohaku.provenance.ProvenanceGraph`; required for
        the ``depth`` signal to contribute. Without it the depth signal
        falls back to 0 and the remaining three are renormalised.
    half_life_days:
        Half-life used by the recency signal.
    freq_cap:
        ``reinforcement_count`` value at which the frequency signal
        saturates at 1.0.
    weights:
        Per-signal weight dict; must contain all four keys and sum to ~1.
    blend_alpha:
        Smoothing factor in [0, 1]. 1.0 replaces ``importance`` outright;
        0.0 keeps the existing value (no-op). Default 1.0.
    """

    def __init__(
        self,
        store: EnrichedMemoryStore,
        *,
        provenance: "Optional[object]" = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        freq_cap: int = DEFAULT_FREQ_CAP,
        weights: Optional[Dict[str, float]] = None,
        blend_alpha: float = 1.0,
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be > 0")
        if freq_cap <= 0:
            raise ValueError("freq_cap must be > 0")
        if not 0.0 <= blend_alpha <= 1.0:
            raise ValueError("blend_alpha must be in [0, 1]")
        w = dict(weights or DEFAULT_WEIGHTS)
        if set(w) != set(DEFAULT_WEIGHTS):
            raise ValueError(
                f"weights must contain exactly {sorted(DEFAULT_WEIGHTS)}, "
                f"got {sorted(w)}"
            )
        if any(v < 0 for v in w.values()):
            raise ValueError("weights must be >= 0")
        total = sum(w.values())
        if total <= 0:
            raise ValueError("weights must sum > 0")
        # Renormalise to a unit sum so the score lands in [0, 1].
        self.weights = {k: v / total for k, v in w.items()}
        self.store = store
        self.provenance = provenance
        self.half_life_days = float(half_life_days)
        self.freq_cap = int(freq_cap)
        self.blend_alpha = float(blend_alpha)

    # ── public API ───────────────────────────────────────────────────────
    def compute(
        self,
        *,
        now: Optional[datetime] = None,
    ) -> List[ImportanceBreakdown]:
        """Compute the proposed new importance score for every live memory.

        Read-only. Returns one :class:`ImportanceBreakdown` per memory.
        """
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        entries = self.store.episodic.entries()
        if not entries:
            return []
        # Pre-compute the per-memory uniqueness via one O(n²) cosine pass.
        uniqueness = self._uniqueness_scores(entries)
        # Pre-compute children-count normalisation.
        children_norm = self._children_norm(entries)

        out: List[ImportanceBreakdown] = []
        for e in entries:
            meta = self.store.get_metadata(e.id)
            if meta is None:
                continue
            freq = min(1.0, meta.reinforcement_count / float(self.freq_cap))
            age_days = max(0.0, (now - meta.valid_from).total_seconds() / 86_400.0)
            recency = math.pow(0.5, age_days / self.half_life_days)
            unique = uniqueness.get(e.id, 0.0)
            depth = children_norm.get(e.id, 0.0)
            composite = (
                self.weights["frequency"] * freq
                + self.weights["recency"] * recency
                + self.weights["uniqueness"] * unique
                + self.weights["depth"] * depth
            )
            composite = max(0.0, min(1.0, composite))
            blended = (
                self.blend_alpha * composite
                + (1.0 - self.blend_alpha) * meta.importance
            )
            out.append(
                ImportanceBreakdown(
                    entry_id=e.id,
                    frequency=freq,
                    recency=recency,
                    uniqueness=unique,
                    depth=depth,
                    composite=composite,
                    importance_before=float(meta.importance),
                    importance_after=float(blended),
                )
            )
        return out

    def apply(
        self,
        *,
        now: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> RescoreReport:
        """Compute scores and write them back unless ``dry_run`` is set."""
        breakdowns = self.compute(now=now)
        total = len(breakdowns)
        updated = 0
        if not dry_run:
            for b in breakdowns:
                meta = self.store.get_metadata(b.entry_id)
                if meta is None:
                    continue
                if meta.importance != b.importance_after:
                    meta.importance = b.importance_after
                    updated += 1
        return RescoreReport(
            total_memories=total,
            updated=updated,
            skipped=total - updated,
            dry_run=bool(dry_run),
            blend_alpha=self.blend_alpha,
            weights=dict(self.weights),
            breakdowns=breakdowns,
        )

    # ── component computations ───────────────────────────────────────────
    def _uniqueness_scores(self, entries: list) -> Dict[int, float]:
        """Per-memory uniqueness = 1 - max cosine to any other memory.

        With < 2 memories every memory is fully unique (1.0).
        """
        n = len(entries)
        if n < 2:
            return {e.id: 1.0 for e in entries}
        # One batched cosine pass per row over a resident index instead of the
        # O(n²) Python double loop (Rust popcount when built, else NumPy matmul).
        idx = index_over(entries)
        scores: Dict[int, float] = {}
        for i, ei in enumerate(entries):
            sims = idx.all_scores(ei.key.data)
            sims[i] = -1.0  # exclude self before taking the max-other
            # Bipolar cosine ∈ [-1, 1]; clamp to [0, 1] for the signal.
            best_clamped = max(0.0, float(sims.max()))
            scores[ei.id] = 1.0 - best_clamped
        return scores

    def _children_norm(self, entries: list) -> Dict[int, float]:
        """Per-memory provenance-depth signal in [0, 1]."""
        if self.provenance is None:
            return {e.id: 0.0 for e in entries}
        child_counts: Dict[int, int] = {}
        for e in entries:
            try:
                # Use the provenance's BFS — descendants depth=1 gives direct children.
                descendants = self.provenance.get_descendants(e.id, max_depth=1)
            except (AttributeError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "depth signal skipped for %s (%s)",
                    e.id,
                    exc.__class__.__name__,
                )
                descendants = []
            child_counts[e.id] = len(descendants)
        max_log = math.log1p(max(child_counts.values())) if child_counts else 0.0
        if max_log <= 0:
            return {eid: 0.0 for eid in child_counts}
        return {eid: math.log1p(cnt) / max_log for eid, cnt in child_counts.items()}


def rescore_all(
    store: EnrichedMemoryStore,
    *,
    provenance: "Optional[object]" = None,
    dry_run: bool = False,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    freq_cap: int = DEFAULT_FREQ_CAP,
    weights: Optional[Dict[str, float]] = None,
    blend_alpha: float = 1.0,
    now: Optional[datetime] = None,
) -> RescoreReport:
    """One-shot helper: build a scorer with the given config and apply it."""
    scorer = ImportanceScorer(
        store,
        provenance=provenance,
        half_life_days=half_life_days,
        freq_cap=freq_cap,
        weights=weights,
        blend_alpha=blend_alpha,
    )
    return scorer.apply(now=now, dry_run=dry_run)
