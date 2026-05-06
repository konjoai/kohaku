"""Real-time streaming consolidation — auto-consolidates memory in a background thread."""
from __future__ import annotations
import threading
import logging
import time
from typing import Optional, List
from .consolidation import consolidate_to_memory
from ._pure import EpisodicMemory, HyperVector
from ._query import RetrievalResult, query

logger = logging.getLogger(__name__)


class StreamingConsolidator:
    """Background thread that auto-consolidates an EpisodicMemory when it nears capacity.

    When the memory utilization exceeds `trigger_ratio`, the background thread runs
    consolidation and replaces the contents of the memory with cluster centroids.

    Thread safety: uses a lock around all EpisodicMemory access.
    """

    def __init__(
        self,
        memory: EpisodicMemory,
        trigger_ratio: float = 0.85,
        poll_interval_s: float = 1.0,
        similarity_threshold: float = 0.7,
    ):
        if not (0.0 < trigger_ratio <= 1.0):
            raise ValueError("trigger_ratio must be in (0, 1]")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self._memory = memory
        self._trigger = trigger_ratio
        self._interval = poll_interval_s
        self._threshold = similarity_threshold
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consolidation_count = 0

    @property
    def consolidation_count(self) -> int:
        return self._consolidation_count

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the background consolidation thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def store(self, key: HyperVector, value: HyperVector, label: str = "") -> None:
        """Thread-safe store into memory."""
        with self._lock:
            self._memory.store(key, value, label)

    def retrieve(self, query_key: HyperVector, top_k: int = 1) -> List[RetrievalResult]:
        """Thread-safe retrieve from memory."""
        with self._lock:
            return query(self._memory, query_key, top_k)

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            with self._lock:
                try:
                    self._maybe_consolidate()
                except Exception as e:
                    logger.warning("StreamingConsolidator: consolidation error: %s", e)

    def _maybe_consolidate(self) -> None:
        utilization = len(self._memory) / self._memory._capacity
        if utilization < self._trigger:
            return
        consolidated = consolidate_to_memory(
            self._memory, similarity_threshold=self._threshold
        )
        # Rebuild the memory from consolidated entries
        self._memory.clear()
        for entry in consolidated.entries():
            self._memory.store(entry.key, entry.value, entry.label)
        self._consolidation_count += 1
        logger.info(
            "StreamingConsolidator: consolidated to %d entries (utilization was %.1f%%)",
            len(self._memory),
            utilization * 100,
        )

    def __enter__(self) -> "StreamingConsolidator":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
