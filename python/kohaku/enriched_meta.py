"""Metadata primitives for the enriched memory store.

Split out of :mod:`kohaku.enriched` so that module stays under the 500-line
quality-gate cap. Holds the per-memory metadata dataclass, the salience
formula, the source-trust table, and the small datetime/tag helpers. The
:class:`~kohaku.enriched.EnrichedMemoryStore` lives in ``enriched.py`` and
re-exports everything here for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, cast

from kohaku._pure import HyperVector

# Default trust weights per source. Agent-generated memories are trusted
# less by default — they don't fail closed, just rank lower under salience.
SOURCE_TRUST_WEIGHTS: Dict[str, float] = {
    "user_input": 1.0,
    "tool_result": 0.9,
    "web_search": 0.8,
    "agent_inference": 0.5,
}

# Default reinforcement scaling — every retrieval bumps salience by `k`.
DEFAULT_REINFORCEMENT_K: float = 0.1
DEFAULT_HALF_LIFE_DAYS: float = 30.0
DEFAULT_IMPORTANCE: float = 0.5

SortMode = Literal["similarity", "salience", "recency"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    """Promote naive datetimes to UTC-aware so comparisons are well-defined."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _normalise_tag(tag: str) -> str:
    """Lower-cased, whitespace-stripped tag (≤ 64 chars). Empty if invalid."""
    if not isinstance(tag, str):
        return ""
    cleaned = tag.strip().lower()
    return cleaned[:64]


@dataclass
class MemoryMetadata:
    """Per-memory metadata maintained outside :class:`EpisodicMemory`.

    All datetimes are stored as UTC-aware. Naive datetimes are promoted on
    write so downstream comparisons never raise ``TypeError``.
    """

    entry_id: int
    valid_from: datetime
    valid_until: Optional[datetime] = None
    source: str = "user_input"
    importance: float = DEFAULT_IMPORTANCE
    reinforcement_count: int = 0
    created_at: datetime = field(default_factory=_utcnow)
    tags: set[str] = field(default_factory=set)
    forgetting_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be in [0, 1], got {self.importance}")
        if self.reinforcement_count < 0:
            raise ValueError("reinforcement_count must be >= 0")
        if self.forgetting_rate is not None and self.forgetting_rate <= 0:
            raise ValueError("forgetting_rate must be > 0")
        self.valid_from = _aware(self.valid_from)
        if self.valid_until is not None:
            self.valid_until = _aware(self.valid_until)
            if self.valid_until < self.valid_from:
                raise ValueError("valid_until must be >= valid_from")
        self.created_at = _aware(self.created_at)
        if self.source == "":
            raise ValueError("source must be non-empty")
        # Coerce tags to a set of normalised non-empty strings. Empty tags are
        # silently dropped — they're never useful for filtering.
        self.tags = {_normalise_tag(t) for t in self.tags if _normalise_tag(t)}

    def is_valid_at(self, now: datetime) -> bool:
        """True if this memory is currently active at ``now``."""
        now = _aware(now)
        if now < self.valid_from:
            return False
        if self.valid_until is not None and now > self.valid_until:
            return False
        return True

    def age_days(self, now: datetime) -> float:
        """Days elapsed since ``created_at`` (used by the recency decay)."""
        now = _aware(now)
        delta = now - self.created_at
        return max(0.0, delta.total_seconds() / 86_400.0)

    def trust(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights if weights is not None else SOURCE_TRUST_WEIGHTS
        return w.get(self.source, 0.5)

    def salience(
        self,
        *,
        now: datetime,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        reinforcement_k: float = DEFAULT_REINFORCEMENT_K,
        trust_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Composite salience score.

        salience = importance · 0.5^(age / effective_half_life) · (1 + count · k) · trust(source)

        When ``forgetting_rate`` is set, ``effective_half_life = half_life_days / forgetting_rate``
        so rates > 1 accelerate decay and rates < 1 slow it.  High-importance memories
        naturally warrant a low forgetting rate (longer half-life).
        """
        if half_life_days <= 0:
            raise ValueError("half_life_days must be > 0")
        effective_hl = (
            half_life_days / self.forgetting_rate
            if self.forgetting_rate is not None
            else half_life_days
        )
        age = self.age_days(now)
        recency = 0.5 ** (age / effective_hl)
        reinforcement = 1.0 + self.reinforcement_count * reinforcement_k
        return cast(
            float,
            self.importance * recency * reinforcement * self.trust(trust_weights),
        )


@dataclass(frozen=True)
class EnrichedRetrievalResult:
    """One row of an enriched query result."""

    entry_id: int
    label: str
    similarity: float
    salience: float
    source: str
    importance: float
    reinforcement_count: int
    valid_from: datetime
    valid_until: Optional[datetime]
    trust: float
    value: HyperVector
    tags: tuple[str, ...] = ()  # immutable view for the frozen dataclass

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "label": self.label,
            "similarity": round(float(self.similarity), 6),
            "salience": round(float(self.salience), 6),
            "source": self.source,
            "importance": float(self.importance),
            "reinforcement_count": int(self.reinforcement_count),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "trust": float(self.trust),
            "tags": list(self.tags),
        }
