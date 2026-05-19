"""Tests for kohaku.sleep — sleep-phase consolidation daemon."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku.sleep import SleepConsolidator, SleepReport


def _noisy(base: HyperVector, frac: float, seed: int) -> HyperVector:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(base), size=int(frac * len(base)), replace=False)
    data = base.data.copy()
    data[idx] *= -1
    return HyperVector(data)


# ──────────────────────── construction / validation ──────────────────────────

def test_invalid_interval_raises() -> None:
    mem = EpisodicMemory(capacity=10)
    with pytest.raises(ValueError, match="consolidation_interval"):
        SleepConsolidator(mem, consolidation_interval_minutes=0)
    with pytest.raises(ValueError, match="consolidation_interval"):
        SleepConsolidator(mem, consolidation_interval_minutes=-1)


def test_invalid_threshold_raises() -> None:
    mem = EpisodicMemory(capacity=10)
    with pytest.raises(ValueError, match="similarity_threshold"):
        SleepConsolidator(mem, similarity_threshold=2.0)


# ───────────────────────────── run_once mechanics ────────────────────────────

def test_run_once_on_empty_memory_returns_zero_report() -> None:
    mem = EpisodicMemory(capacity=10)
    sleeper = SleepConsolidator(mem)
    report = sleeper.run_once()
    assert isinstance(report, SleepReport)
    assert report.episodes_before == 0
    assert report.episodes_after == 0
    assert report.prototypes_created == 0
    assert report.memory_freed == 0


def test_run_once_merges_high_similarity_cluster() -> None:
    mem = EpisodicMemory(capacity=20)
    base = HyperVector.random(DIMS, seed=42)
    val = HyperVector.random(DIMS, seed=43)
    # 5 variants with 2% flips → pairwise cosine ~0.92 (above 0.85 threshold)
    for i in range(5):
        mem.store(_noisy(base, 0.02, seed=100 + i), val, "cat")
    # One orthogonal outlier
    mem.store(HyperVector.random(DIMS, seed=999), val, "dog")

    sleeper = SleepConsolidator(mem, similarity_threshold=0.85)
    report = sleeper.run_once()

    assert report.episodes_before == 6
    assert report.prototypes_created == 1
    assert report.episodes_after == 2
    assert report.memory_freed == 4
    assert report.run_seconds > 0


def test_run_once_no_merge_below_threshold() -> None:
    mem = EpisodicMemory(capacity=10)
    val = HyperVector.random(DIMS, seed=1)
    # 4 mutually-orthogonal entries
    for s in range(4):
        mem.store(HyperVector.random(DIMS, seed=1000 + s), val, f"x{s}")
    sleeper = SleepConsolidator(mem, similarity_threshold=0.85)
    r = sleeper.run_once()
    assert r.prototypes_created == 0
    assert r.episodes_after == r.episodes_before == 4
    assert r.memory_freed == 0


def test_singleton_memory_emits_zero_report() -> None:
    mem = EpisodicMemory(capacity=10)
    hv = HyperVector.random(DIMS, seed=1)
    mem.store(hv, hv, "lonely")
    sleeper = SleepConsolidator(mem)
    r = sleeper.run_once()
    assert r.episodes_before == 1
    assert r.episodes_after == 1
    assert r.prototypes_created == 0


def test_run_count_and_reports_accumulate() -> None:
    mem = EpisodicMemory(capacity=10)
    hv = HyperVector.random(DIMS, seed=1)
    mem.store(hv, hv, "x")
    sleeper = SleepConsolidator(mem)
    sleeper.run_once()
    sleeper.run_once()
    sleeper.run_once()
    assert sleeper.run_count == 3
    assert len(sleeper.reports()) == 3
    assert sleeper.last_report is sleeper.reports()[-1]


# ───────────────────────────────── callback ───────────────────────────────────

def test_on_report_callback_fires() -> None:
    mem = EpisodicMemory(capacity=10)
    received: list[SleepReport] = []
    sleeper = SleepConsolidator(mem, on_report=received.append)
    r = sleeper.run_once()
    assert received == [r]


def test_callback_exception_does_not_break_run() -> None:
    mem = EpisodicMemory(capacity=10)

    def bad_cb(_: SleepReport) -> None:
        raise RuntimeError("boom")

    sleeper = SleepConsolidator(mem, on_report=bad_cb)
    # Should not raise — exception in callback is logged and swallowed.
    r = sleeper.run_once()
    assert isinstance(r, SleepReport)


# ───────────────────────────────── lifecycle ──────────────────────────────────

def test_start_stop_lifecycle() -> None:
    mem = EpisodicMemory(capacity=10)
    sleeper = SleepConsolidator(mem, consolidation_interval_minutes=10)
    assert not sleeper.is_running
    sleeper.start()
    assert sleeper.is_running
    sleeper.stop(timeout=2.0)
    assert not sleeper.is_running


def test_double_start_is_idempotent() -> None:
    mem = EpisodicMemory(capacity=10)
    sleeper = SleepConsolidator(mem, consolidation_interval_minutes=10)
    sleeper.start()
    t1 = sleeper._thread
    sleeper.start()
    assert sleeper._thread is t1
    sleeper.stop()


def test_context_manager_runs_then_stops() -> None:
    mem = EpisodicMemory(capacity=10)
    with SleepConsolidator(mem, consolidation_interval_minutes=10) as sleeper:
        assert sleeper.is_running
    assert not sleeper.is_running


def test_background_thread_fires_at_short_interval() -> None:
    """Use a 0.05-min interval (=3s) is too long for a unit test; we use
    an absurd 1ms / 60_000 = 1/60000 of a minute — but our floor is positive
    not millisecond-min. Use 0.001 min = 60ms."""
    mem = EpisodicMemory(capacity=10)
    base = HyperVector.random(DIMS, seed=1)
    val = HyperVector.random(DIMS, seed=2)
    for i in range(3):
        mem.store(_noisy(base, 0.02, 50 + i), val, "g")

    triggered = threading.Event()

    def cb(_: SleepReport) -> None:
        triggered.set()

    sleeper = SleepConsolidator(
        mem,
        consolidation_interval_minutes=0.001,   # 60ms
        similarity_threshold=0.85,
        on_report=cb,
    )
    sleeper.start()
    try:
        assert triggered.wait(timeout=3.0), "background run did not fire"
    finally:
        sleeper.stop(timeout=2.0)
    assert sleeper.run_count >= 1


# ────────────────────────── SleepReport contract ─────────────────────────────

def test_report_to_dict_is_json_serializable() -> None:
    import json
    mem = EpisodicMemory(capacity=10)
    hv = HyperVector.random(DIMS, seed=1)
    mem.store(hv, hv, "x")
    sleeper = SleepConsolidator(mem)
    r = sleeper.run_once()
    d = r.to_dict()
    s = json.dumps(d)
    assert "started_at" in d
    assert "run_seconds" in d
    assert isinstance(json.loads(s), dict)
