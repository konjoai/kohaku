"""Sleep-phase consolidation daemon — periodically merges episodic clusters
into semantic prototypes.

Distinct from :class:`kohaku.streaming.StreamingConsolidator`:

* ``StreamingConsolidator`` fires when memory utilisation crosses a capacity
  threshold. It's reactive — pressure-driven.
* ``SleepConsolidator`` fires on a *time* schedule (default every 60 minutes)
  and emits a structured :class:`SleepReport` per run. It models the
  hippocampus→neocortex sleep-replay loop more literally: cycles happen
  whether or not the system is under pressure.

Each run does:

1. Snapshot the episodic count and dimension footprint.
2. Run :func:`kohaku.consolidation.consolidate` at the configured cosine
   threshold (default ≥ 0.85).
3. Replace the episodic memory's entries with the cluster centroids — only
   the clusters that *actually merged* (size > 1) are counted as
   prototypes; singletons are preserved.
4. Emit a ``SleepReport`` with ``episodes_consolidated``,
   ``prototypes_created``, ``memory_freed``, ``run_seconds``, ``started_at``.

The daemon is opt-in: instantiate, call ``start()``, the background thread
runs until ``stop()`` (or the context manager exits). Manual ``run_once()``
is always available — that's what the ``POST /consolidate`` API endpoint
calls.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

from .consolidation import consolidate
from ._pure import EpisodicMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SleepReport:
    """Structured outcome of one consolidation run."""
    started_at: datetime
    run_seconds: float
    episodes_before: int
    episodes_after: int
    episodes_consolidated: int    # entries that were absorbed into a prototype
    prototypes_created: int        # clusters with size > 1
    memory_freed: int              # episodes_before - episodes_after
    similarity_threshold: float

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "run_seconds": round(float(self.run_seconds), 4),
            "episodes_before": int(self.episodes_before),
            "episodes_after": int(self.episodes_after),
            "episodes_consolidated": int(self.episodes_consolidated),
            "prototypes_created": int(self.prototypes_created),
            "memory_freed": int(self.memory_freed),
            "similarity_threshold": float(self.similarity_threshold),
        }


SleepCallback = Callable[[SleepReport], None]


class SleepConsolidator:
    """Time-scheduled consolidation thread for an :class:`EpisodicMemory`.

    Parameters
    ----------
    memory:
        The store to consolidate. Same instance used by the rest of the app —
        consolidation mutates its contents in place.
    consolidation_interval_minutes:
        Cadence (default 60). Sub-minute values are accepted for testing.
    similarity_threshold:
        Cosine cutoff for cluster membership (default 0.85).
    on_report:
        Optional callback fired after each run with the :class:`SleepReport`.
        Useful for logging, metrics, websocket push, etc.
    """

    def __init__(
        self,
        memory: EpisodicMemory,
        *,
        consolidation_interval_minutes: float = 60.0,
        similarity_threshold: float = 0.85,
        on_report: Optional[SleepCallback] = None,
    ) -> None:
        if consolidation_interval_minutes <= 0:
            raise ValueError("consolidation_interval_minutes must be > 0")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        self._memory = memory
        self._interval_s = consolidation_interval_minutes * 60.0
        self._threshold = similarity_threshold
        self._on_report = on_report
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reports: List[SleepReport] = []
        self._run_count = 0

    # ── basic accessors ────────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def last_report(self) -> Optional[SleepReport]:
        return self._reports[-1] if self._reports else None

    def reports(self) -> List[SleepReport]:
        return list(self._reports)

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="kohaku-sleep-consolidator")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None

    def __enter__(self) -> "SleepConsolidator":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ── core operation ─────────────────────────────────────────────────────
    def run_once(self) -> SleepReport:
        """Run one consolidation pass synchronously. Returns the report."""
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        with self._lock:
            episodes_before = len(self._memory)
            if episodes_before <= 1:
                # nothing to consolidate
                report = SleepReport(
                    started_at=started,
                    run_seconds=time.perf_counter() - t0,
                    episodes_before=episodes_before,
                    episodes_after=episodes_before,
                    episodes_consolidated=0,
                    prototypes_created=0,
                    memory_freed=0,
                    similarity_threshold=self._threshold,
                )
            else:
                clusters = consolidate(
                    self._memory, similarity_threshold=self._threshold
                )
                prototypes_created = sum(1 for c in clusters if c.size > 1)
                episodes_consolidated = sum(c.size for c in clusters if c.size > 1)
                episodes_after = len(clusters)
                # Rebuild memory from cluster centroids, preserving FIFO order.
                self._memory.clear()
                for c in clusters:
                    label = (
                        f"{c.label} (n={c.size})" if c.size > 1 else c.label
                    )
                    self._memory.store(c.centroid_key, c.centroid_value, label)
                report = SleepReport(
                    started_at=started,
                    run_seconds=time.perf_counter() - t0,
                    episodes_before=episodes_before,
                    episodes_after=episodes_after,
                    episodes_consolidated=episodes_consolidated,
                    prototypes_created=prototypes_created,
                    memory_freed=episodes_before - episodes_after,
                    similarity_threshold=self._threshold,
                )
            self._reports.append(report)
            self._run_count += 1

        if self._on_report is not None:
            try:
                self._on_report(report)
            except Exception:  # noqa: BLE001
                logger.warning("SleepConsolidator: on_report callback raised", exc_info=True)
        logger.info(
            "SleepConsolidator run: %d → %d entries (%d prototypes, %d freed) in %.3fs",
            report.episodes_before, report.episodes_after,
            report.prototypes_created, report.memory_freed, report.run_seconds,
        )
        return report

    # ── thread loop ────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            # Use Event.wait() so stop() interrupts the sleep cleanly.
            if self._stop_event.wait(self._interval_s):
                return
            try:
                self.run_once()
            except Exception:  # noqa: BLE001
                logger.warning("SleepConsolidator: run_once raised", exc_info=True)
