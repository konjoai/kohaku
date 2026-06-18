"""Tests for kohaku.importance — auto importance scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kohaku import EnrichedMemoryStore, ProvenanceGraph, encode_text
from kohaku.importance import (
    DEFAULT_WEIGHTS,
    ImportanceScorer,
    RescoreReport,
    rescore_all,
)


def _store(
    provenance: bool = False,
) -> tuple[EnrichedMemoryStore, ProvenanceGraph | None]:
    pg = ProvenanceGraph() if provenance else None
    return EnrichedMemoryStore(capacity=30, provenance=pg), pg


def test_empty_store_returns_empty_breakdowns() -> None:
    s, _ = _store()
    scorer = ImportanceScorer(s)
    assert scorer.compute() == []
    r = scorer.apply()
    assert r.total_memories == 0
    assert r.updated == 0


def test_default_weights_sum_to_one() -> None:
    assert pytest.approx(sum(DEFAULT_WEIGHTS.values())) == 1.0


def test_invalid_constructor_args_raise() -> None:
    s, _ = _store()
    with pytest.raises(ValueError):
        ImportanceScorer(s, half_life_days=0)
    with pytest.raises(ValueError):
        ImportanceScorer(s, freq_cap=0)
    with pytest.raises(ValueError):
        ImportanceScorer(s, blend_alpha=1.5)
    with pytest.raises(ValueError):
        ImportanceScorer(s, weights={"frequency": 0.5})  # missing keys
    with pytest.raises(ValueError):
        ImportanceScorer(s, weights={k: -1.0 for k in DEFAULT_WEIGHTS})


def test_weights_renormalised_on_construction() -> None:
    s, _ = _store()
    scorer = ImportanceScorer(s, weights={k: 1.0 for k in DEFAULT_WEIGHTS})
    assert pytest.approx(sum(scorer.weights.values())) == 1.0


def test_frequency_signal_responds_to_reinforcement() -> None:
    s, _ = _store()
    h = encode_text("alpha")
    e1 = s.store(h, h, label="alpha", source="user_input", importance=0.5)
    e2 = s.store(
        encode_text("beta"),
        encode_text("beta"),
        label="beta",
        source="user_input",
        importance=0.5,
    )
    for _ in range(20):
        s.reinforce(e1)
    scorer = ImportanceScorer(s)
    bd = scorer.compute()
    by_id = {b.entry_id: b for b in bd}
    assert by_id[e1].frequency == pytest.approx(1.0)
    assert by_id[e2].frequency == pytest.approx(0.0)
    assert by_id[e1].composite > by_id[e2].composite


def test_recency_signal_decays_with_age() -> None:
    s, _ = _store()
    now = datetime.now(timezone.utc)
    h = encode_text("ok")
    e1 = s.store(h, h, label="ok", valid_from=now)
    e2 = s.store(
        encode_text("ok2"),
        encode_text("ok2"),
        label="ok2",
        valid_from=now - timedelta(days=30),
    )  # 1 half-life
    scorer = ImportanceScorer(s, half_life_days=30.0)
    bd = scorer.compute(now=now)
    by_id = {b.entry_id: b for b in bd}
    assert by_id[e1].recency == pytest.approx(1.0)
    assert by_id[e2].recency == pytest.approx(0.5, abs=1e-3)


def test_uniqueness_signal_low_for_duplicates() -> None:
    s, _ = _store()
    h = encode_text("the cat sat on the mat")
    e1 = s.store(h, h, label="cat-1", source="user_input")
    e2 = s.store(h, h, label="cat-2", source="user_input")
    e3 = s.store(
        encode_text("espresso brewed dark and bitter"),
        encode_text("espresso brewed dark and bitter"),
        label="espresso",
        source="user_input",
    )
    scorer = ImportanceScorer(s)
    bd = {b.entry_id: b for b in scorer.compute()}
    # The two duplicates have very low uniqueness; the espresso entry is high.
    assert bd[e1].uniqueness < 0.2
    assert bd[e2].uniqueness < 0.2
    assert bd[e3].uniqueness > 0.5


def test_depth_signal_requires_provenance() -> None:
    s, _ = _store()  # no provenance
    h = encode_text("ok")
    s.store(h, h, label="ok")
    scorer = ImportanceScorer(s)  # no provenance arg
    bd = scorer.compute()
    assert all(b.depth == 0.0 for b in bd)


def test_depth_signal_picks_up_consolidation_children() -> None:
    s, pg = _store(provenance=True)
    h = encode_text("ok")
    s.store(h, h, label="parent")
    # Manually create a child via the provenance graph
    pg.record("100", parent_ids=["1"], source_type="consolidation")
    scorer = ImportanceScorer(s, provenance=pg)
    bd = {b.entry_id: b for b in scorer.compute()}
    assert bd[1].depth > 0.0


def test_apply_writes_back_to_metadata() -> None:
    s, _ = _store()
    h = encode_text("ok")
    eid = s.store(h, h, label="ok", importance=0.5)
    before = s.get_metadata(eid).importance
    report = rescore_all(s, dry_run=False)
    assert isinstance(report, RescoreReport)
    after = s.get_metadata(eid).importance
    # Single-memory: full uniqueness + full recency + zero frequency / depth.
    # With default weights the composite should be ~0.5 — definitely changed.
    assert after != before


def test_dry_run_does_not_mutate() -> None:
    s, _ = _store()
    h = encode_text("ok")
    eid = s.store(h, h, label="ok", importance=0.5)
    report = rescore_all(s, dry_run=True)
    assert report.dry_run is True
    assert report.updated == 0
    assert s.get_metadata(eid).importance == 0.5


def test_blend_alpha_zero_is_noop() -> None:
    s, _ = _store()
    h = encode_text("ok")
    eid = s.store(h, h, label="ok", importance=0.3)
    rescore_all(s, blend_alpha=0.0)
    assert s.get_metadata(eid).importance == pytest.approx(0.3)


def test_breakdown_to_dict_shape() -> None:
    s, _ = _store()
    h = encode_text("ok")
    s.store(h, h, label="ok")
    bd = ImportanceScorer(s).compute()
    assert bd
    d = bd[0].to_dict()
    assert {
        "entry_id",
        "frequency",
        "recency",
        "uniqueness",
        "depth",
        "composite",
        "importance_before",
        "importance_after",
    } <= d.keys()


def test_report_to_dict_includes_weights() -> None:
    s, _ = _store()
    s.store(encode_text("a"), encode_text("a"), label="a")
    report = ImportanceScorer(s).apply()
    d = report.to_dict()
    assert set(d["weights"]) == set(DEFAULT_WEIGHTS)
    assert isinstance(d["breakdowns"], list)
