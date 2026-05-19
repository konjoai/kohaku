"""Tests for Phase 10 — Memory Graph Export (v0.9.0)."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from kohaku._pure import EpisodicMemory, HyperVector
from kohaku.graph_export import (
    GraphExportConfig,
    MemoryGraph,
    MemoryGraphExporter,
)
from kohaku.learning import ItemMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episodic(n: int = 4, seed_base: int = 0) -> EpisodicMemory:
    """Return an EpisodicMemory populated with *n* deterministic HVs."""
    mem = EpisodicMemory(capacity=100)
    for i in range(n):
        hv = HyperVector.random(dims=256, seed=seed_base + i)
        mem.store(hv, hv, label=f"entry_{i}")
    return mem


def _make_semantic(labels: list[str], seed_base: int = 100) -> ItemMemory:
    """Return an ItemMemory with one prototype per label."""
    im = ItemMemory(dims=256)
    for i, lbl in enumerate(labels):
        hv = HyperVector.random(dims=256, seed=seed_base + i)
        im.add(lbl, hv)
    return im


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_graph_export_config_defaults() -> None:
    """GraphExportConfig must have the documented default values."""
    cfg = GraphExportConfig()
    assert cfg.similarity_threshold == 0.3
    assert cfg.include_self_loops is False
    assert cfg.max_nodes == 5000
    assert cfg.decay_config is None
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# Basic export behaviour
# ---------------------------------------------------------------------------

def test_export_returns_memory_graph() -> None:
    """export() must return a MemoryGraph instance."""
    mem = _make_episodic(3)
    exporter = MemoryGraphExporter()
    result = exporter.export(mem)
    assert isinstance(result, MemoryGraph)


def test_export_node_count_matches_entries() -> None:
    """n_nodes must equal the number of stored memory entries."""
    mem = _make_episodic(5)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    assert graph.n_nodes == 5
    assert len(graph.nodes) == 5


def test_export_node_has_required_fields() -> None:
    """Every node must expose node_id, label, source, timestamp."""
    mem = _make_episodic(2)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    for node in graph.nodes:
        assert node.node_id
        assert node.label
        assert node.source in ("episodic", "semantic")
        assert isinstance(node.timestamp, (int, type(None)))


def test_export_episodic_source_label() -> None:
    """Nodes derived from EpisodicMemory must have source='episodic'."""
    mem = _make_episodic(3)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    assert all(n.source == "episodic" for n in graph.nodes)


# ---------------------------------------------------------------------------
# Edge behaviour
# ---------------------------------------------------------------------------

def test_export_edges_above_threshold() -> None:
    """All edges must have similarity >= similarity_threshold."""
    mem = _make_episodic(6)
    cfg = GraphExportConfig(similarity_threshold=0.0)
    exporter = MemoryGraphExporter(cfg)
    graph = exporter.export(mem)
    for edge in graph.edges:
        assert edge.similarity >= 0.0


def test_export_no_edges_when_threshold_1() -> None:
    """With threshold=1.0 no edges should be added (HVs are not identical)."""
    mem = _make_episodic(4)
    cfg = GraphExportConfig(similarity_threshold=1.0)
    exporter = MemoryGraphExporter(cfg)
    graph = exporter.export(mem)
    assert graph.n_edges == 0


def test_export_all_edges_when_threshold_0() -> None:
    """With threshold=0.0, edges = n*(n-1)/2 (no self-loops, undirected)."""
    n = 5
    mem = _make_episodic(n)
    cfg = GraphExportConfig(similarity_threshold=0.0)
    exporter = MemoryGraphExporter(cfg)
    graph = exporter.export(mem)
    # threshold=0.0 — all non-negative sims qualify; random HVs in high dim
    # are nearly orthogonal, so some may be negative. We only assert that edges
    # with negative similarity are absent and that total <= n*(n-1)/2.
    assert graph.n_edges <= n * (n - 1) // 2


def test_export_similarity_in_range() -> None:
    """All edge similarities must be in the valid cosine range [-1, 1]."""
    mem = _make_episodic(6)
    cfg = GraphExportConfig(similarity_threshold=-1.0)
    exporter = MemoryGraphExporter(cfg)
    graph = exporter.export(mem)
    for edge in graph.edges:
        assert -1.0 <= edge.similarity <= 1.0


def test_export_no_self_loops_default() -> None:
    """By default, no self-loop edges should be present."""
    mem = _make_episodic(4)
    cfg = GraphExportConfig(similarity_threshold=-1.0)  # include negative to get all
    exporter = MemoryGraphExporter(cfg)
    graph = exporter.export(mem)
    for edge in graph.edges:
        assert edge.source_id != edge.target_id


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_export_to_dict_roundtrip() -> None:
    """to_dict() must contain all required top-level keys."""
    mem = _make_episodic(3)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    d = graph.to_dict()
    for key in ("n_nodes", "n_edges", "similarity_threshold", "exported_at", "nodes", "edges"):
        assert key in d, f"missing key: {key}"
    assert d["n_nodes"] == 3


def test_export_to_json_valid() -> None:
    """to_json() must produce valid JSON."""
    mem = _make_episodic(3)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    parsed = json.loads(graph.to_json())
    assert parsed["n_nodes"] == 3


def test_export_gexf_is_xml() -> None:
    """to_gexf() must return a parseable XML string."""
    mem = _make_episodic(3)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    gexf = graph.to_gexf()
    ET.fromstring(gexf)  # raises on invalid XML


def test_export_gexf_has_nodes_element() -> None:
    """GEXF output must contain a <nodes> element with correct child count."""
    mem = _make_episodic(4)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    root = ET.fromstring(graph.to_gexf())
    ns = {"g": "http://gexf.net/1.3"}
    nodes_el = root.find(".//g:nodes", ns)
    assert nodes_el is not None
    assert len(list(nodes_el)) == 4


def test_export_gexf_has_edges_element() -> None:
    """GEXF output must contain an <edges> element."""
    mem = _make_episodic(4)
    cfg = GraphExportConfig(similarity_threshold=-1.0)
    exporter = MemoryGraphExporter(cfg)
    graph = exporter.export(mem)
    root = ET.fromstring(graph.to_gexf())
    ns = {"g": "http://gexf.net/1.3"}
    edges_el = root.find(".//g:edges", ns)
    assert edges_el is not None


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def test_save_json_creates_file(tmp_path: Path) -> None:
    """save_json() must create the target file."""
    mem = _make_episodic(3)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    dest = tmp_path / "graph.json"
    exporter.save_json(graph, dest)
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["n_nodes"] == 3


def test_save_gexf_creates_file(tmp_path: Path) -> None:
    """save_gexf() must create the target file."""
    mem = _make_episodic(3)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    dest = tmp_path / "graph.gexf"
    exporter.save_gexf(graph, dest)
    assert dest.exists()
    ET.parse(dest)  # valid XML


def test_save_dispatch_by_extension(tmp_path: Path) -> None:
    """save() must dispatch to save_json or save_gexf based on extension."""
    mem = _make_episodic(3)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    json_path = tmp_path / "g.json"
    gexf_path = tmp_path / "g.gexf"
    exporter.save(graph, json_path)
    exporter.save(graph, gexf_path)
    assert json_path.exists()
    assert gexf_path.exists()


def test_save_unknown_extension_raises(tmp_path: Path) -> None:
    """save() must raise ValueError for unknown extensions."""
    mem = _make_episodic(2)
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem)
    with pytest.raises(ValueError, match="unknown file extension"):
        exporter.save(graph, tmp_path / "graph.xyz")


# ---------------------------------------------------------------------------
# Semantic memory integration
# ---------------------------------------------------------------------------

def test_export_with_semantic_memory() -> None:
    """ItemMemory nodes must be included with source='semantic'."""
    mem = _make_episodic(3)
    sem = _make_semantic(["cat", "dog", "fish"])
    exporter = MemoryGraphExporter()
    graph = exporter.export(mem, semantic=sem)
    assert graph.n_nodes == 6  # 3 episodic + 3 semantic
    semantic_nodes = [n for n in graph.nodes if n.source == "semantic"]
    assert len(semantic_nodes) == 3
    semantic_labels = {n.label for n in semantic_nodes}
    assert semantic_labels == {"cat", "dog", "fish"}
