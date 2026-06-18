"""Memory health dashboard — operational view over an :class:`EnrichedMemoryStore`.

Reports four classes of issue:

* **Stale**            — created ``stale_days+`` ago AND never reinforced.
* **Expired**          — ``valid_until`` is in the past.
* **Orphaned**         — present in the store but missing from the provenance graph
                         (cannot be lineage-traced).
* **Duplicate pairs**  — pairwise cosine on the bipolar key vectors ≥ a threshold
                         (default 0.95). O(n²); intended for ≤ a few thousand
                         entries, which matches the kohaku scale.

The composite :attr:`MemoryHealthReport.health_score` is in [0, 1] and is meant
for at-a-glance dashboards — it is NOT a replacement for inspecting each
contributor. Recommendations are emitted only when the underlying counts cross
sensible thresholds, so a clean store returns an empty list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence

from kohaku._index import index_over
from kohaku.enriched import EnrichedMemoryStore
from kohaku.provenance import ProvenanceGraph

logger = logging.getLogger(__name__)


def _age_from_valid_from(meta: Any, now: datetime) -> float:
    """Days since the memory's ``valid_from`` (its in-corpus lifetime).

    Distinct from :meth:`MemoryMetadata.age_days`, which measures wall-clock
    time since the row was written. For the health view we want the
    *semantic* age — when the memory's validity began — so tests and callers
    can author memories with historical timestamps.
    """
    vf = meta.valid_from
    if vf.tzinfo is None:
        vf = vf.replace(tzinfo=timezone.utc)
    delta = now - vf
    return max(0.0, delta.total_seconds() / 86_400.0)


DEFAULT_STALE_DAYS: int = 30
DEFAULT_DUPLICATE_THRESHOLD: float = 0.95
SALIENCE_BUCKET_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Health-score weights for each issue class. They sum to 1.0 so a fully-broken
# store scores 0.0 and a clean one scores 1.0.
_WEIGHTS = {"stale": 0.30, "expired": 0.20, "orphan": 0.20, "duplicate": 0.30}


# ──────────────────────────── DTOs ─────────────────────────────────────────


@dataclass(frozen=True)
class DuplicatePair:
    a_id: int
    b_id: int
    similarity: float
    label_a: str
    label_b: str

    def to_dict(self) -> dict:
        return {
            "a_id": int(self.a_id),
            "b_id": int(self.b_id),
            "similarity": round(float(self.similarity), 4),
            "label_a": self.label_a,
            "label_b": self.label_b,
        }


@dataclass(frozen=True)
class StaleMemory:
    entry_id: int
    label: str
    age_days: float
    last_accessed: Optional[str]
    reinforcement_count: int

    def to_dict(self) -> dict:
        return {
            "entry_id": int(self.entry_id),
            "label": self.label,
            "age_days": round(float(self.age_days), 2),
            "last_accessed": self.last_accessed,
            "reinforcement_count": int(self.reinforcement_count),
        }


@dataclass(frozen=True)
class MemoryHealthReport:
    total_memories: int
    stale_memories: int
    expired_memories: int
    orphaned_memories: int
    duplicate_candidates: List[DuplicatePair]
    storage_bytes: int
    avg_access_frequency: float
    salience_buckets: List[int]
    health_score: float
    recommendations: List[str] = field(default_factory=list)
    stale_days: int = DEFAULT_STALE_DAYS
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "total_memories": int(self.total_memories),
            "stale_memories": int(self.stale_memories),
            "expired_memories": int(self.expired_memories),
            "orphaned_memories": int(self.orphaned_memories),
            "duplicate_candidates": [p.to_dict() for p in self.duplicate_candidates],
            "storage_bytes": int(self.storage_bytes),
            "avg_access_frequency": round(float(self.avg_access_frequency), 3),
            "salience_buckets": [int(b) for b in self.salience_buckets],
            "salience_bucket_edges": list(SALIENCE_BUCKET_EDGES),
            "health_score": round(float(self.health_score), 4),
            "recommendations": list(self.recommendations),
            "stale_days": int(self.stale_days),
            "duplicate_threshold": round(float(self.duplicate_threshold), 4),
        }


# ──────────────────────────── analyzer ─────────────────────────────────────


class MemoryHealthAnalyzer:
    """Compute health metrics over a live :class:`EnrichedMemoryStore`.

    Pass an optional :class:`ProvenanceGraph` to enable orphan detection — a
    memory is orphaned when it is present in the store but absent from the
    provenance graph (and therefore can't be lineage-traced).
    """

    def __init__(
        self,
        store: EnrichedMemoryStore,
        *,
        provenance: Optional[ProvenanceGraph] = None,
        stale_days: int = DEFAULT_STALE_DAYS,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        max_duplicate_pairs: int = 50,
    ) -> None:
        if stale_days <= 0:
            raise ValueError("stale_days must be > 0")
        if not 0.0 < duplicate_threshold <= 1.0:
            raise ValueError("duplicate_threshold must be in (0, 1]")
        if max_duplicate_pairs < 0:
            raise ValueError("max_duplicate_pairs must be >= 0")
        self.store = store
        self.provenance = provenance
        self.stale_days = int(stale_days)
        self.duplicate_threshold = float(duplicate_threshold)
        self.max_duplicate_pairs = int(max_duplicate_pairs)

    # ── primary report ────────────────────────────────────────────────────
    def compute(self, *, now: Optional[datetime] = None) -> MemoryHealthReport:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        entries = self.store.episodic.entries()
        total = len(entries)

        stale_ids = self._stale_ids(entries, now=now)
        expired_ids = self._expired_ids(entries, now=now)
        orphan_ids = self._orphan_ids(entries)
        duplicates = self._duplicate_pairs(entries)
        salience_buckets = self._salience_buckets(now=now)
        avg_access = self._avg_access(entries)
        storage = self._storage_bytes(entries)

        score = self._health_score(
            total=total,
            stale=len(stale_ids),
            expired=len(expired_ids),
            orphans=len(orphan_ids),
            duplicates=len(duplicates),
        )
        recs = self._recommendations(
            total=total,
            stale=len(stale_ids),
            expired=len(expired_ids),
            orphans=len(orphan_ids),
            duplicates=len(duplicates),
            avg_access=avg_access,
        )

        return MemoryHealthReport(
            total_memories=total,
            stale_memories=len(stale_ids),
            expired_memories=len(expired_ids),
            orphaned_memories=len(orphan_ids),
            duplicate_candidates=duplicates,
            storage_bytes=storage,
            avg_access_frequency=avg_access,
            salience_buckets=salience_buckets,
            health_score=score,
            recommendations=recs,
            stale_days=self.stale_days,
            duplicate_threshold=self.duplicate_threshold,
        )

    # ── stale list endpoint ───────────────────────────────────────────────
    def list_stale(
        self,
        *,
        days: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> List[StaleMemory]:
        """List memories that are stale per the (optional override) threshold."""
        threshold = days if days is not None else self.stale_days
        if threshold <= 0:
            raise ValueError("days must be > 0")
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        out: List[StaleMemory] = []
        for e in self.store.episodic.entries():
            meta = self.store.get_metadata(e.id)
            if meta is None:
                continue
            age = _age_from_valid_from(meta, now)
            if age >= threshold and meta.reinforcement_count == 0:
                out.append(
                    StaleMemory(
                        entry_id=e.id,
                        label=e.label,
                        age_days=age,
                        last_accessed=None,  # not tracked yet; placeholder for v2
                        reinforcement_count=meta.reinforcement_count,
                    )
                )
        out.sort(key=lambda s: s.age_days, reverse=True)
        return out

    def delete_stale(
        self,
        *,
        days: Optional[int] = None,
        dry_run: bool = True,
        now: Optional[datetime] = None,
    ) -> dict:
        """Delete stale memories (or report what *would* be deleted under dry_run)."""
        stale = self.list_stale(days=days, now=now)
        ids = [s.entry_id for s in stale]
        deleted = 0
        if not dry_run and ids:
            target = set(ids)
            kept = [e for e in self.store.episodic._entries if e.id not in target]
            removed = len(self.store.episodic._entries) - len(kept)
            self.store.episodic._entries = kept
            if removed:
                self.store.episodic._mark_mutated()  # invalidate index cache
            for eid in ids:
                self.store._meta.pop(eid, None)
                if self.provenance is not None:
                    self.provenance.delete(eid)
            deleted = removed
        return {
            "candidates": [s.to_dict() for s in stale],
            "candidate_count": len(stale),
            "deleted_count": deleted,
            "dry_run": dry_run,
            "days": days if days is not None else self.stale_days,
        }

    # ── component computations ────────────────────────────────────────────
    def _stale_ids(self, entries: Sequence[Any], *, now: datetime) -> List[int]:
        out: List[int] = []
        for e in entries:
            meta = self.store.get_metadata(e.id)
            if meta is None:
                continue
            age = _age_from_valid_from(meta, now)
            if age >= self.stale_days and meta.reinforcement_count == 0:
                out.append(e.id)
        return out

    def _expired_ids(self, entries: Sequence[Any], *, now: datetime) -> List[int]:
        out: List[int] = []
        for e in entries:
            meta = self.store.get_metadata(e.id)
            if meta is None:
                continue
            if not meta.is_valid_at(now):
                out.append(e.id)
        return out

    def _orphan_ids(self, entries: Sequence[Any]) -> List[int]:
        if self.provenance is None:
            return []
        return [e.id for e in entries if not self.provenance.has(e.id)]

    def _duplicate_pairs(self, entries: Sequence[Any]) -> List[DuplicatePair]:
        out: List[DuplicatePair] = []
        n = len(entries)
        if n < 2 or self.max_duplicate_pairs == 0:
            return out
        sim_matrix = index_over(entries).all_pairs()
        for i in range(n):
            sims = sim_matrix[i]
            for j in range(i + 1, n):
                sim = float(sims[j])
                if sim >= self.duplicate_threshold:
                    out.append(
                        DuplicatePair(
                            a_id=entries[i].id,
                            b_id=entries[j].id,
                            similarity=sim,
                            label_a=entries[i].label,
                            label_b=entries[j].label,
                        )
                    )
                    if len(out) >= self.max_duplicate_pairs:
                        return out
        return out

    def _salience_buckets(self, *, now: datetime) -> List[int]:
        # 5 buckets between SALIENCE_BUCKET_EDGES[i] and [i+1]
        buckets = [0] * (len(SALIENCE_BUCKET_EDGES) - 1)
        for e in self.store.episodic.entries():
            meta = self.store.get_metadata(e.id)
            if meta is None:
                continue
            sal = meta.salience(
                now=now,
                half_life_days=self.store.half_life_days,
                reinforcement_k=self.store.reinforcement_k,
                trust_weights=self.store.trust_weights,
            )
            clamped = max(0.0, min(1.0, sal))
            # Map [0, 1] to bucket index in [0, len-1]
            idx = min(len(buckets) - 1, int(clamped * len(buckets)))
            buckets[idx] += 1
        return buckets

    def _avg_access(self, entries: Sequence[Any]) -> float:
        if not entries:
            return 0.0
        total = 0
        count = 0
        for e in entries:
            meta = self.store.get_metadata(e.id)
            if meta is None:
                continue
            total += meta.reinforcement_count
            count += 1
        return total / count if count else 0.0

    def _storage_bytes(self, entries: Sequence[Any]) -> int:
        # Two bipolar vectors (key + value) packed at 1 bit per component,
        # plus the entry's label UTF-8 bytes. Mirrors the .hkb on-disk format.
        per_vector = (self.store.dims + 7) // 8
        total = 0
        for e in entries:
            total += 2 * per_vector
            total += len(e.label.encode("utf-8"))
        return total

    def _health_score(
        self,
        *,
        total: int,
        stale: int,
        expired: int,
        orphans: int,
        duplicates: int,
    ) -> float:
        if total == 0:
            return 1.0
        # All four ratios are capped at 1.0; the weighted sum subtracted from 1
        # gives the score. Each ratio is computed against `total` for stable
        # behaviour as the store grows.
        ratios = {
            "stale": stale / total,
            "expired": expired / total,
            "orphan": orphans / total,
            "duplicate": (2 * duplicates) / total,  # both members of each pair count
        }
        penalty = sum(_WEIGHTS[k] * min(1.0, v) for k, v in ratios.items())
        return max(0.0, min(1.0, 1.0 - penalty))

    def _recommendations(
        self,
        *,
        total: int,
        stale: int,
        expired: int,
        orphans: int,
        duplicates: int,
        avg_access: float,
    ) -> List[str]:
        recs: List[str] = []
        if total == 0:
            return recs
        if expired > 0:
            recs.append(f"POST /memories/expire to drop {expired} expired item(s).")
        if stale / max(1, total) > 0.20:
            recs.append(
                f"Consider deleting {stale} stale items "
                f"(no reinforcement in {self.stale_days}+ days) — "
                f"DELETE /memories/stale?days={self.stale_days}&dry_run=true."
            )
        if duplicates > 0:
            recs.append(
                f"POST /consolidate could merge {duplicates} duplicate pair(s) "
                f"(cosine ≥ {self.duplicate_threshold:.2f})."
            )
        if orphans > 0:
            recs.append(
                f"{orphans} memory record(s) missing provenance — "
                "rebuild via /memories/{id}/provenance after the next write."
            )
        if avg_access < 0.5 and total > 10:
            recs.append(
                "Average reinforcement is below 0.5 — many memories are "
                "never recalled. Review what is being stored."
            )
        return recs
