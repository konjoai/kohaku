"""Tests for Graphiti and Mem0 export dialects (Phase 15)."""

from __future__ import annotations

import json
from pathlib import Path

from kohaku._pure import EpisodicMemory, HyperVector
from kohaku.graph_export import (
    GraphExportConfig,
    MemoryGraph,
    MemoryGraphExporter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episodic(n: int = 4, seed_base: int = 0) -> EpisodicMemory:
    mem = EpisodicMemory(capacity=100)
    for i in range(n):
        hv = HyperVector.random(dims=256, seed=seed_base + i)
        mem.store(hv, hv, label=f"memory_{i}")
    return mem


def _graph(n: int = 3, threshold: float = -1.0) -> MemoryGraph:
    mem = _make_episodic(n)
    cfg = GraphExportConfig(similarity_threshold=threshold)
    return MemoryGraphExporter(cfg).export(mem)


# ---------------------------------------------------------------------------
# Graphiti — dict structure
# ---------------------------------------------------------------------------


def test_to_graphiti_top_level_keys() -> None:
    """to_graphiti() must have all required top-level keys."""
    d = _graph().to_graphiti()
    assert d["format"] == "graphiti"
    assert d["version"] == "1.0"
    assert "exported_at" in d
    assert "similarity_threshold" in d
    assert isinstance(d["episodes"], list)
    assert isinstance(d["entities"], list)
    assert isinstance(d["relations"], list)


def test_to_graphiti_episode_count_matches_nodes() -> None:
    g = _graph(n=5)
    d = g.to_graphiti()
    assert len(d["episodes"]) == g.n_nodes == 5


def test_to_graphiti_episode_fields() -> None:
    """Each episode must carry uuid, name, content, source, timestamps, attributes."""
    d = _graph(n=2).to_graphiti()
    ep = d["episodes"][0]
    assert {
        "uuid",
        "name",
        "content",
        "source",
        "created_at",
        "valid_at",
        "invalid_at",
        "attributes",
    } <= ep.keys()
    assert ep["invalid_at"] is None
    assert isinstance(ep["attributes"], dict)


def test_to_graphiti_entities_always_empty() -> None:
    """entities list must be empty — kohaku memories map to episodes not entities."""
    d = _graph(n=4).to_graphiti()
    assert d["entities"] == []


def test_to_graphiti_relations_count_matches_edges() -> None:
    g = _graph(n=4, threshold=-1.0)
    d = g.to_graphiti()
    assert len(d["relations"]) == g.n_edges


def test_to_graphiti_relation_fields() -> None:
    g = _graph(n=3, threshold=-1.0)
    d = g.to_graphiti()
    if d["relations"]:
        rel = d["relations"][0]
        assert {
            "uuid",
            "name",
            "fact",
            "source_node_uuid",
            "target_node_uuid",
            "created_at",
            "expired_at",
            "weight",
        } <= rel.keys()
        assert rel["name"] == "similar_to"
        assert rel["expired_at"] is None
        assert -1.0 <= rel["weight"] <= 1.0


def test_to_graphiti_json_is_valid_json() -> None:
    payload = _graph().to_graphiti_json()
    parsed = json.loads(payload)
    assert parsed["format"] == "graphiti"


def test_save_graphiti_creates_file(tmp_path: Path) -> None:
    g = _graph()
    exp = MemoryGraphExporter()
    dest = tmp_path / "graph.graphiti.json"
    exp.save_graphiti(g, dest)
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["format"] == "graphiti"


# ---------------------------------------------------------------------------
# Mem0 — dict structure
# ---------------------------------------------------------------------------


def test_to_mem0_top_level_keys() -> None:
    d = _graph().to_mem0()
    assert d["format"] == "mem0"
    assert d["version"] == "1.0"
    assert "exported_at" in d
    assert isinstance(d["memories"], list)


def test_to_mem0_memory_count_matches_nodes() -> None:
    g = _graph(n=6)
    d = g.to_mem0()
    assert len(d["memories"]) == g.n_nodes == 6


def test_to_mem0_memory_fields() -> None:
    d = _graph(n=2).to_mem0()
    mem = d["memories"][0]
    assert {
        "id",
        "memory",
        "hash",
        "metadata",
        "score",
        "created_at",
        "updated_at",
    } <= mem.keys()
    assert isinstance(mem["hash"], str) and len(mem["hash"]) == 16
    assert isinstance(mem["metadata"], dict)
    assert 0.0 <= mem["score"] <= 1.0


def test_to_mem0_hash_is_deterministic() -> None:
    g = _graph(n=2)
    d1 = g.to_mem0()
    d2 = g.to_mem0()
    assert d1["memories"][0]["hash"] == d2["memories"][0]["hash"]


def test_to_mem0_score_fallback_when_no_decay() -> None:
    """Without a DecayConfig, score must default to 1.0."""
    d = _graph().to_mem0()
    for mem in d["memories"]:
        assert mem["score"] == 1.0


def test_to_mem0_json_is_valid_json() -> None:
    payload = _graph().to_mem0_json()
    parsed = json.loads(payload)
    assert parsed["format"] == "mem0"


def test_save_mem0_creates_file(tmp_path: Path) -> None:
    g = _graph()
    exp = MemoryGraphExporter()
    dest = tmp_path / "export.mem0.json"
    exp.save_mem0(g, dest)
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["format"] == "mem0"
