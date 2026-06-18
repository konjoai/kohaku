"""Tests for StreamingConsolidator."""

import pytest
from kohaku.streaming import StreamingConsolidator
from kohaku._pure import EpisodicMemory, HyperVector


DIM = 64


def make_memory(capacity=10):
    return EpisodicMemory(capacity=capacity)


def rand_hv(seed=0):
    return HyperVector.random(dims=DIM, seed=seed)


def test_consolidator_init():
    mem = make_memory()
    sc = StreamingConsolidator(mem)
    assert sc.consolidation_count == 0
    assert not sc.is_running


def test_consolidator_bad_trigger():
    mem = make_memory()
    with pytest.raises(ValueError):
        StreamingConsolidator(mem, trigger_ratio=0.0)
    with pytest.raises(ValueError):
        StreamingConsolidator(mem, trigger_ratio=1.5)


def test_consolidator_bad_interval():
    mem = make_memory()
    with pytest.raises(ValueError):
        StreamingConsolidator(mem, poll_interval_s=0.0)


def test_start_stop():
    mem = make_memory()
    sc = StreamingConsolidator(mem, poll_interval_s=0.05)
    sc.start()
    assert sc.is_running
    sc.stop(timeout=1.0)
    assert not sc.is_running


def test_double_start_noop():
    mem = make_memory()
    sc = StreamingConsolidator(mem, poll_interval_s=0.1)
    sc.start()
    sc.start()  # second start is a no-op
    assert sc.is_running
    sc.stop()


def test_context_manager():
    mem = make_memory()
    with StreamingConsolidator(mem, poll_interval_s=0.1) as sc:
        assert sc.is_running
    assert not sc.is_running


def test_thread_safe_store_retrieve():
    mem = make_memory(capacity=20)
    sc = StreamingConsolidator(mem)
    k = rand_hv(0)
    v = rand_hv(1)
    sc.store(k, v, label="t")
    results = sc.retrieve(k, top_k=1)
    assert len(results) == 1


def test_consolidation_triggers():
    """Force consolidation by filling memory past trigger_ratio."""
    mem = make_memory(capacity=5)
    sc = StreamingConsolidator(mem, trigger_ratio=0.8, poll_interval_s=999)
    # Fill past trigger (5/5 = 100% > 80%)
    for i in range(5):
        k = rand_hv(i)
        v = rand_hv(i + 10)
        mem.store(k, v, label=f"e{i}")
    before = len(mem)
    sc._maybe_consolidate()
    # After consolidation, count should be <= original
    assert len(mem) <= before


def test_consolidation_count_increments():
    mem = make_memory(capacity=3)
    sc = StreamingConsolidator(mem, trigger_ratio=0.5, poll_interval_s=999)
    for i in range(3):
        k = rand_hv(i)
        v = rand_hv(i + 10)
        mem.store(k, v, label=f"e{i}")
    sc._maybe_consolidate()
    assert sc.consolidation_count == 1


def test_no_consolidation_below_trigger():
    mem = make_memory(capacity=10)
    sc = StreamingConsolidator(mem, trigger_ratio=0.9)
    # Only 2 of 10 entries = 20% utilization
    for i in range(2):
        mem.store(rand_hv(i), rand_hv(i + 5), label=f"e{i}")
    sc._maybe_consolidate()
    assert sc.consolidation_count == 0
