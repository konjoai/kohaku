"""Write-time validation and poisoning defense for EpisodicMemory.

Two independent gates applied before every store:

1. **Novelty check** — reject if cosine to the nearest stored entry is >=
   ``duplicate_threshold`` (catches verbatim re-submissions and near-clones).
2. **Rate limit** — reject if a named source has exceeded ``max_stores``
   within the sliding ``window_seconds`` window (per-source deque of timestamps).

Usage::

    from kohaku.validation import WriteValidator, RateLimit

    validator = WriteValidator(
        memory,
        duplicate_threshold=0.99,
        rate_limits={"agent_inference": RateLimit(max_stores=100, window_seconds=60.0)},
    )

    result = validator.validate(key_hv, source="agent_inference")
    if result.accepted:
        entry_id = memory.store(key_hv, value_hv, label)
        validator.record(source="agent_inference")

    # Or in one call:
    result, entry_id = validator.validate_and_store(key_hv, value_hv, label,
                                                    source="agent_inference")
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from kohaku._pure import EpisodicMemory, HyperVector
from kohaku._query import query


@dataclass(frozen=True)
class RateLimit:
    """Policy: at most ``max_stores`` stores from one source within ``window_seconds``."""

    max_stores: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.max_stores <= 0:
            raise ValueError("max_stores must be > 0")
        if self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be > 0")


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason: str  # "accepted" | "near_duplicate" | "rate_limit_exceeded"
    nearest_similarity: float
    nearest_label: str


class WriteValidator:
    """Validates HVs before storage, enforcing novelty and rate-limit policies.

    Note: the sliding-window deques are not locked. Use an external lock when
    calling :meth:`validate_and_store` from multiple threads simultaneously.
    """

    def __init__(
        self,
        memory: EpisodicMemory,
        *,
        duplicate_threshold: float = 0.99,
        rate_limits: Optional[Dict[str, RateLimit]] = None,
    ) -> None:
        if not (0.0 < duplicate_threshold <= 1.0):
            raise ValueError("duplicate_threshold must be in (0, 1]")
        self._memory = memory
        self._duplicate_threshold = duplicate_threshold
        self._rate_limits: Dict[str, RateLimit] = rate_limits or {}
        self._store_times: Dict[str, deque[float]] = defaultdict(deque)

    def validate(
        self,
        key_hv: HyperVector,
        source: Optional[str] = None,
    ) -> ValidationResult:
        """Check novelty and rate limits without modifying any state.

        Call :meth:`record` after a successful store to update the rate-limit
        window.
        """
        nearest_sim = 0.0
        nearest_label = ""

        if not self._memory.is_empty:
            results = query(self._memory, key_hv, top_k=1)
            if results:
                nearest_sim = results[0].similarity
                nearest_label = results[0].label
                if nearest_sim >= self._duplicate_threshold:
                    return ValidationResult(
                        accepted=False,
                        reason="near_duplicate",
                        nearest_similarity=nearest_sim,
                        nearest_label=nearest_label,
                    )

        if source and source in self._rate_limits:
            limit = self._rate_limits[source]
            now = time.time()
            dq = self._store_times[source]
            cutoff = now - limit.window_seconds
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= limit.max_stores:
                return ValidationResult(
                    accepted=False,
                    reason="rate_limit_exceeded",
                    nearest_similarity=nearest_sim,
                    nearest_label=nearest_label,
                )

        return ValidationResult(
            accepted=True,
            reason="accepted",
            nearest_similarity=nearest_sim,
            nearest_label=nearest_label,
        )

    def record(self, source: Optional[str] = None) -> None:
        """Record a successful store for rate-limit accounting."""
        if source and source in self._rate_limits:
            self._store_times[source].append(time.time())

    def validate_and_store(
        self,
        key_hv: HyperVector,
        value_hv: HyperVector,
        label: str,
        source: Optional[str] = None,
    ) -> Tuple[ValidationResult, Optional[int]]:
        """Validate and, if accepted, store to memory atomically.

        Returns ``(result, entry_id)`` where ``entry_id`` is ``None`` on
        rejection.
        """
        result = self.validate(key_hv, source)
        if result.accepted:
            entry_id = self._memory.store(key_hv, value_hv, label)
            self.record(source)
            return result, entry_id
        return result, None
