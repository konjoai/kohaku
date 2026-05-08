"""Tests for kohaku.streaming — background StreamingConsolidator."""
from __future__ import annotations

import time
import threading

import pytest

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku.streaming import StreamingConsolidator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mem(capacity: int = 20) -> EpisodicMemory:
    return EpisodicMemory(capacity=capacity)


def _hv(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


# ---------------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------------

def test_init_defaults() -> None:
    mem = _mem()
    sc = StreamingConsolidator(mem)
    assert sc.consolidation_count == 0
    assert not sc.is_running


def test_init_bad_trigger_ratio() -> None:
    mem = _mem()
    with pytest.raises(ValueError):
        StreamingConsolidator(mem, trigger_ratio=0.0)
    with pytest.raises(ValueError):
        StreamingConsolidator(mem, trigger_ratio=1.1)


def test_init_bad_poll_interval() -> None:
    mem = _mem()
    with pytest.raises(ValueError):
        StreamingConsolidator(mem, poll_interval_s=0.0)
    with pytest.raises(ValueError):
        StreamingConsolidator(mem, poll_interval_s=-1.0)


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

def test_start_sets_is_running() -> None:
    sc = StreamingConsolidator(_mem(), poll_interval_s=0.05)
    sc.start()
    try:
        assert sc.is_running
    finally:
        sc.stop()


def test_stop_clears_is_running() -> None:
    sc = StreamingConsolidator(_mem(), poll_interval_s=0.05)
    sc.start()
    sc.stop()
    assert not sc.is_running


def test_double_start_is_noop() -> None:
    """Calling start() twice should not spawn a second thread."""
    sc = StreamingConsolidator(_mem(), poll_interval_s=0.05)
    sc.start()
    thread_id_first = sc._thread.ident if sc._thread else None
    sc.start()  # second call — should be a no-op
    thread_id_second = sc._thread.ident if sc._thread else None
    try:
        assert thread_id_first == thread_id_second
    finally:
        sc.stop()


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------

def test_context_manager_starts_and_stops() -> None:
    mem = _mem()
    with StreamingConsolidator(mem, poll_interval_s=0.05) as sc:
        assert sc.is_running
    assert not sc.is_running


# ---------------------------------------------------------------------------
# thread-safe store and retrieve
# ---------------------------------------------------------------------------

def test_thread_safe_store_and_retrieve() -> None:
    """Concurrent stores from multiple threads must not corrupt the memory."""
    mem = _mem(capacity=200)
    sc = StreamingConsolidator(mem, trigger_ratio=0.99, poll_interval_s=10.0)
    sc.start()
    errors = []

    def worker(offset: int) -> None:
        try:
            for i in range(10):
                sc.store(_hv(offset + i), _hv(offset + i + 1000), label=f"t{offset}-{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t * 100,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    sc.stop()

    assert errors == [], f"Thread errors: {errors}"
    assert len(mem) == 50  # 5 threads × 10 stores each


# ---------------------------------------------------------------------------
# consolidation trigger
# ---------------------------------------------------------------------------

def test_consolidation_fires_when_above_trigger() -> None:
    """Fill memory past trigger_ratio; background thread should consolidate."""
    capacity = 10
    mem = EpisodicMemory(capacity=capacity)
    sc = StreamingConsolidator(
        mem,
        trigger_ratio=0.80,   # triggers at 8/10 entries
        poll_interval_s=0.05,
        similarity_threshold=0.1,  # very permissive — everything merges
    )

    # Store 9 similar entries (90% utilization → above 80% trigger)
    base = _hv(seed=1)
    import numpy as np
    for i in range(9):
        rng = np.random.default_rng(i + 10)
        data = base.data.copy()
        idx = rng.choice(DIMS, size=int(0.05 * DIMS), replace=False)
        data[idx] *= -1
        mem.store(HyperVector(data), _hv(seed=2), label="cluster")

    sc.start()
    # Wait up to 1 second for consolidation to run
    deadline = time.monotonic() + 1.0
    while sc.consolidation_count == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    sc.stop()

    assert sc.consolidation_count >= 1, "consolidation never fired"
    # After consolidation the memory should be smaller than before
    assert len(mem) < 9


def test_consolidation_count_increments() -> None:
    """Consolidation count increases each time consolidation runs."""
    capacity = 4
    mem = EpisodicMemory(capacity=capacity)
    sc = StreamingConsolidator(
        mem,
        trigger_ratio=0.5,   # triggers at 2/4
        poll_interval_s=0.05,
        similarity_threshold=0.1,
    )
    import numpy as np
    base = _hv(seed=77)
    # Pre-fill to trigger immediately
    for i in range(3):
        rng = np.random.default_rng(i)
        data = base.data.copy()
        data[rng.choice(DIMS, size=int(0.03 * DIMS), replace=False)] *= -1
        mem.store(HyperVector(data), _hv(seed=2), label="x")

    sc.start()
    deadline = time.monotonic() + 1.0
    while sc.consolidation_count == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    count_after_first = sc.consolidation_count
    sc.stop()

    assert count_after_first >= 1


def test_no_consolidation_below_trigger() -> None:
    """Memory below trigger_ratio must not be consolidated."""
    mem = EpisodicMemory(capacity=100)
    sc = StreamingConsolidator(
        mem,
        trigger_ratio=0.90,
        poll_interval_s=0.05,
    )
    # Only 5 entries in a capacity-100 store → well below 90%
    for i in range(5):
        mem.store(_hv(seed=i), _hv(seed=i + 100), label=f"e{i}")

    sc.start()
    time.sleep(0.2)  # give the thread a couple of poll cycles
    sc.stop()

    assert sc.consolidation_count == 0
    assert len(mem) == 5  # nothing evicted
