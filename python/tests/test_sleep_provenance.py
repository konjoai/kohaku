"""Tests for the sleep ↔ provenance lineage wiring + consolidation history."""

from __future__ import annotations

import numpy as np
import pytest

from kohaku import (
    EnrichedMemoryStore,
    ProvenanceGraph,
    SleepConsolidator,
    encode_text,
)
from kohaku._pure import DIMS, HyperVector


def _noisy(base: HyperVector, flip_frac: float, seed: int) -> HyperVector:
    rng = np.random.default_rng(seed)
    n_flip = int(flip_frac * len(base))
    idx = rng.choice(len(base), size=n_flip, replace=False)
    data = base.data.copy()
    data[idx] *= -1
    return HyperVector(data)


def test_sleep_without_provenance_still_runs() -> None:
    store = EnrichedMemoryStore(capacity=10)
    h = encode_text("alpha")
    store.store(h, h, label="alpha", source="user_input")
    sl = SleepConsolidator(store.episodic, similarity_threshold=0.30)
    report = sl.run_once()
    assert report.episodes_before == 1
    assert report.prototypes_created == 0


def test_sleep_records_consolidation_lineage() -> None:
    pg = ProvenanceGraph()
    store = EnrichedMemoryStore(capacity=20, provenance=pg)
    base = HyperVector.random(DIMS, seed=42)
    for i in range(4):
        v = _noisy(base, 0.05, seed=100 + i)
        store.store(v, v, label=f"variant-{i}", source="user_input")
    sl = SleepConsolidator(
        store.episodic,
        similarity_threshold=0.30,
        provenance=pg,
    )
    report = sl.run_once()
    assert report.prototypes_created == 1
    # After consolidation: one consolidation node in the graph with
    # source_type='consolidation'.
    consolidation_nodes = []
    for e in store.episodic.entries():
        full = pg.get_full_graph(e.id)
        for n in full.nodes:
            if n.source_type == "consolidation":
                consolidation_nodes.append(n)
                break
    assert consolidation_nodes, "at least one consolidation lineage edge expected"


def test_consolidation_metadata_carries_threshold_and_cluster_size() -> None:
    pg = ProvenanceGraph()
    store = EnrichedMemoryStore(capacity=20, provenance=pg)
    base = HyperVector.random(DIMS, seed=7)
    for i in range(3):
        v = _noisy(base, 0.05, seed=200 + i)
        store.store(v, v, label=f"m-{i}", source="user_input")
    sl = SleepConsolidator(
        store.episodic,
        similarity_threshold=0.30,
        provenance=pg,
    )
    sl.run_once()
    # Find the consolidation node — it carries cluster_size + threshold metadata
    for e in store.episodic.entries():
        for n in pg.get_full_graph(e.id).nodes:
            if n.source_type == "consolidation":
                assert n.metadata.get("similarity_threshold") == pytest.approx(0.30)
                assert n.metadata.get("cluster_size") >= 2
                return
    raise AssertionError("no consolidation node found")


def test_singleton_clusters_do_not_record_lineage() -> None:
    pg = ProvenanceGraph()
    store = EnrichedMemoryStore(capacity=10, provenance=pg)
    # Three orthogonal random vectors — each becomes its own cluster.
    for s in range(3):
        v = HyperVector.random(DIMS, seed=1000 + s)
        store.store(v, v, label=f"orth-{s}", source="user_input")
    sl = SleepConsolidator(
        store.episodic,
        similarity_threshold=0.30,
        provenance=pg,
    )
    sl.run_once()
    # No consolidation source_type nodes should exist — each cluster is size 1.
    has_consolidation = False
    for e in store.episodic.entries():
        for n in pg.get_full_graph(e.id).nodes:
            if n.source_type == "consolidation":
                has_consolidation = True
    assert has_consolidation is False


def test_sleep_reports_accumulate() -> None:
    store = EnrichedMemoryStore(capacity=5)
    h = encode_text("once")
    store.store(h, h, label="once", source="user_input")
    sl = SleepConsolidator(store.episodic, similarity_threshold=0.85)
    assert sl.run_count == 0
    sl.run_once()
    sl.run_once()
    sl.run_once()
    assert sl.run_count == 3
    assert len(sl.reports()) == 3
    # Returned list is a copy — mutating doesn't affect internal state
    snapshot = sl.reports()
    snapshot.clear()
    assert len(sl.reports()) == 3


def test_self_parent_collision_is_filtered() -> None:
    """EpisodicMemory.clear() resets _next_id, so the merged_id may match a
    pre-clear source id. The wiring must filter that case without raising."""
    pg = ProvenanceGraph()
    store = EnrichedMemoryStore(capacity=20, provenance=pg)
    base = HyperVector.random(DIMS, seed=99)
    for i in range(2):
        v = _noisy(base, 0.05, seed=300 + i)
        store.store(v, v, label=f"x-{i}", source="user_input")
    sl = SleepConsolidator(
        store.episodic,
        similarity_threshold=0.30,
        provenance=pg,
    )
    # Must not raise; lineage edge may be a singleton if one parent collided
    # with the new id, but the consolidation node itself must exist.
    sl.run_once()
    assert sl.run_count == 1
