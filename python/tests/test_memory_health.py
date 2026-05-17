"""Tests for kohaku.memory_health — operational dashboard analytics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kohaku import EnrichedMemoryStore, encode_text
from kohaku.memory_health import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_STALE_DAYS,
    MemoryHealthAnalyzer,
    MemoryHealthReport,
)
from kohaku.provenance import ProvenanceGraph


def _store_with(provenance: bool = True) -> tuple[EnrichedMemoryStore, ProvenanceGraph]:
    pg = ProvenanceGraph() if provenance else None
    store = EnrichedMemoryStore(capacity=50, provenance=pg)
    return store, pg


def test_empty_store_reports_clean() -> None:
    store, pg = _store_with()
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    report = analyzer.compute()
    assert isinstance(report, MemoryHealthReport)
    assert report.total_memories == 0
    assert report.stale_memories == 0
    assert report.expired_memories == 0
    assert report.orphaned_memories == 0
    assert report.duplicate_candidates == []
    assert report.health_score == 1.0
    assert report.recommendations == []


def test_stale_detection() -> None:
    store, pg = _store_with()
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(days=DEFAULT_STALE_DAYS + 5)
    h_fresh = encode_text("recent thought")
    h_stale = encode_text("old forgotten thing")
    store.store(h_fresh, h_fresh, label="fresh", valid_from=now)
    store.store(h_stale, h_stale, label="stale", valid_from=long_ago)
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    report = analyzer.compute(now=now)
    assert report.total_memories == 2
    assert report.stale_memories == 1
    stale = analyzer.list_stale(now=now)
    assert len(stale) == 1
    assert stale[0].label == "stale"


def test_expired_detection() -> None:
    store, pg = _store_with()
    now = datetime.now(timezone.utc)
    h = encode_text("temp memo")
    store.store(h, h, label="expired",
                valid_from=now - timedelta(days=2),
                valid_until=now - timedelta(days=1))
    store.store(h, h, label="still good",
                valid_from=now - timedelta(days=1),
                valid_until=now + timedelta(days=1))
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    report = analyzer.compute(now=now)
    assert report.expired_memories == 1


def test_duplicate_candidates_flagged() -> None:
    store, pg = _store_with()
    hv = encode_text("identical phrase used twice")
    store.store(hv, hv, label="copy-a")
    store.store(hv, hv, label="copy-b")
    analyzer = MemoryHealthAnalyzer(store, provenance=pg,
                                    duplicate_threshold=0.9)
    report = analyzer.compute()
    assert report.duplicate_candidates, "identical HVs should be flagged"
    pair = report.duplicate_candidates[0]
    assert pair.similarity >= 0.9
    assert {pair.label_a, pair.label_b} == {"copy-a", "copy-b"}


def test_orphans_appear_when_provenance_missing() -> None:
    # Build a store with NO provenance attached, then attach a fresh graph
    # at analysis time — every entry should be orphaned.
    store = EnrichedMemoryStore(capacity=10)
    hv = encode_text("first")
    store.store(hv, hv, label="a")
    store.store(hv, hv, label="b")
    pg = ProvenanceGraph()
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    report = analyzer.compute()
    assert report.orphaned_memories == 2


def test_orphans_zero_without_provenance() -> None:
    store = EnrichedMemoryStore(capacity=10)
    hv = encode_text("anything")
    store.store(hv, hv, label="m")
    analyzer = MemoryHealthAnalyzer(store)
    report = analyzer.compute()
    # When no graph is attached, orphans cannot be detected — reported as 0.
    assert report.orphaned_memories == 0


def test_health_score_decreases_with_issues() -> None:
    store, pg = _store_with()
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(days=DEFAULT_STALE_DAYS + 1)
    # 4 stale memories
    for i in range(4):
        h = encode_text(f"stale {i}")
        store.store(h, h, label=f"old-{i}", valid_from=long_ago)
    # 1 fresh
    h = encode_text("alive")
    store.store(h, h, label="fresh", valid_from=now)
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    report = analyzer.compute(now=now)
    assert 0.0 <= report.health_score < 1.0
    # 4 of 5 stale → ratio 0.8, weight 0.3 → penalty 0.24
    assert report.health_score == pytest.approx(1.0 - 0.3 * 0.8, abs=1e-6)


def test_salience_buckets_sum_to_total() -> None:
    store, pg = _store_with()
    now = datetime.now(timezone.utc)
    for i in range(8):
        h = encode_text(f"phrase {i}")
        store.store(h, h, label=f"m-{i}", importance=i / 8.0, valid_from=now)
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    report = analyzer.compute(now=now)
    assert sum(report.salience_buckets) == 8
    assert len(report.salience_buckets) == 5


def test_recommendations_appear_when_issues_present() -> None:
    store, pg = _store_with()
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(days=DEFAULT_STALE_DAYS + 1)
    # Trigger every recommendation branch we can.
    for i in range(6):
        h = encode_text(f"old item {i}")
        store.store(h, h, label=f"old-{i}", valid_from=long_ago)
    # Add an expired memory.
    h = encode_text("expired memo")
    store.store(h, h, label="expired",
                valid_from=now - timedelta(days=1),
                valid_until=now - timedelta(hours=1))
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    report = analyzer.compute(now=now)
    text = " ".join(report.recommendations)
    assert "stale" in text
    assert "expired" in text


def test_avg_access_tracks_reinforcement() -> None:
    store, pg = _store_with()
    for i in range(3):
        h = encode_text(f"m{i}")
        eid = store.store(h, h, label=f"m{i}")
        store.reinforce(eid, delta=i)  # 0, 1, 2
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    report = analyzer.compute()
    assert report.avg_access_frequency == pytest.approx(1.0)


def test_delete_stale_dry_run_reports_only() -> None:
    store, pg = _store_with()
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(days=DEFAULT_STALE_DAYS + 1)
    h = encode_text("forgotten")
    store.store(h, h, label="forgotten", valid_from=long_ago)
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    result = analyzer.delete_stale(dry_run=True, now=now)
    assert result["candidate_count"] == 1
    assert result["deleted_count"] == 0
    assert len(store) == 1


def test_delete_stale_actually_deletes() -> None:
    store, pg = _store_with()
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(days=DEFAULT_STALE_DAYS + 1)
    h1 = encode_text("forgotten 1")
    h2 = encode_text("forgotten 2")
    h3 = encode_text("recent")
    e1 = store.store(h1, h1, label="forgotten 1", valid_from=long_ago)
    e2 = store.store(h2, h2, label="forgotten 2", valid_from=long_ago)
    store.store(h3, h3, label="recent", valid_from=now)
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    result = analyzer.delete_stale(dry_run=False, now=now)
    assert result["deleted_count"] == 2
    assert len(store) == 1
    assert e1 not in store and e2 not in store
    # Provenance rows should also be gone.
    assert not pg.has(e1) and not pg.has(e2)


def test_storage_bytes_grows_with_entries() -> None:
    store, pg = _store_with()
    before = MemoryHealthAnalyzer(store, provenance=pg).compute().storage_bytes
    h = encode_text("hello")
    store.store(h, h, label="hello")
    after = MemoryHealthAnalyzer(store, provenance=pg).compute().storage_bytes
    assert after > before


def test_constructor_validation() -> None:
    store, _ = _store_with()
    with pytest.raises(ValueError):
        MemoryHealthAnalyzer(store, stale_days=0)
    with pytest.raises(ValueError):
        MemoryHealthAnalyzer(store, duplicate_threshold=1.5)
    with pytest.raises(ValueError):
        MemoryHealthAnalyzer(store, max_duplicate_pairs=-1)


def test_duplicate_threshold_constant_is_used_by_default() -> None:
    store, pg = _store_with()
    analyzer = MemoryHealthAnalyzer(store, provenance=pg)
    assert analyzer.duplicate_threshold == DEFAULT_DUPLICATE_THRESHOLD
