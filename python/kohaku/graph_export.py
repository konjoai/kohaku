"""Memory graph export — serialise episodic/semantic memory as a graph.

Graph model:
  - Nodes: each memory entry (episodic or semantic)
  - Edges: cosine similarity between HVs >= similarity_threshold
  - Node attributes: entry_id, label, timestamp, source ("episodic"|"semantic"),
                     decay_weight (if DecayConfig provided), cluster_id (if clusters provided)
  - Edge attributes: similarity (float)

Cosine similarity: computed in FP32 via np.einsum.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import numpy as np

from kohaku._pure import EpisodicMemory
from kohaku.decay import DecayConfig, decay_weight as _decay_weight
from kohaku.learning import ItemMemory


@dataclass(frozen=True)
class GraphExportConfig:
    """Configuration for memory graph export.

    Attributes
    ----------
    similarity_threshold:
        Minimum cosine similarity for an edge to be added.
    include_self_loops:
        If True, self-loop edges are added (sim of node with itself = 1.0).
    max_nodes:
        Safety cap. Raises ValueError if the total node count exceeds this.
    decay_config:
        If set, compute decay_weight for each node and add it as a node attribute.
    seed:
        Seed logged with the export for reproducibility.
    """

    similarity_threshold: float = 0.3
    include_self_loops: bool = False
    max_nodes: int = 5000
    decay_config: Optional[DecayConfig] = None
    seed: int = 42


@dataclass(frozen=True)
class MemoryNode:
    """A single node in the exported memory graph."""

    node_id: str          # "e_{entry_id}" or "s_{label}"
    label: str
    source: str           # "episodic" | "semantic"
    timestamp: Optional[int]
    decay_weight: Optional[float]  # None if no DecayConfig
    cluster_id: Optional[int]      # None if no clustering info


@dataclass(frozen=True)
class MemoryEdge:
    """A single edge in the exported memory graph."""

    source_id: str
    target_id: str
    similarity: float


@dataclass
class MemoryGraph:
    """Complete exported memory graph with nodes and edges."""

    nodes: list[MemoryNode]
    edges: list[MemoryEdge]
    n_nodes: int
    n_edges: int
    similarity_threshold: float
    exported_at: str  # ISO-8601 UTC

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "similarity_threshold": self.similarity_threshold,
            "exported_at": self.exported_at,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "label": n.label,
                    "source": n.source,
                    "timestamp": n.timestamp,
                    "decay_weight": n.decay_weight,
                    "cluster_id": n.cluster_id,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "similarity": e.similarity,
                }
                for e in self.edges
            ],
        }

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_gexf(self) -> str:
        """Return a GEXF 1.3 XML string representing the graph.

        Structure:
          <gexf version="1.3">
            <graph defaultedgetype="undirected" mode="static">
              <attributes class="node"> ... </attributes>
              <nodes> <node id="..." label="..."> <attvalues>...</attvalues> </node> </nodes>
              <edges> <edge id="..." source="..." target="..." weight="..."/> </edges>
            </graph>
          </gexf>
        """
        root = _build_gexf_root()
        graph_el = _build_gexf_graph(root)
        _add_gexf_node_attributes(graph_el)
        _add_gexf_nodes(graph_el, self.nodes)
        _add_gexf_edges(graph_el, self.edges)
        raw = tostring(root, encoding="unicode")
        return minidom.parseString(raw).toprettyxml(indent="  ")


def _build_gexf_root() -> Element:
    """Create the root GEXF element."""
    root = Element("gexf")
    root.set("xmlns", "http://gexf.net/1.3")
    root.set("version", "1.3")
    return root


def _build_gexf_graph(root: Element) -> Element:
    """Append a static undirected graph element to root."""
    graph_el = SubElement(root, "graph")
    graph_el.set("defaultedgetype", "undirected")
    graph_el.set("mode", "static")
    return graph_el


def _add_gexf_node_attributes(graph_el: Element) -> None:
    """Append node attribute declarations to the graph element."""
    attrs = SubElement(graph_el, "attributes")
    attrs.set("class", "node")
    _attr(attrs, "0", "source", "string")
    _attr(attrs, "1", "timestamp", "integer")
    _attr(attrs, "2", "decay_weight", "float")
    _attr(attrs, "3", "cluster_id", "integer")


def _attr(parent: Element, attr_id: str, title: str, attr_type: str) -> None:
    """Append a single attribute declaration."""
    a = SubElement(parent, "attribute")
    a.set("id", attr_id)
    a.set("title", title)
    a.set("type", attr_type)


def _add_gexf_nodes(graph_el: Element, nodes: list[MemoryNode]) -> None:
    """Append all node elements to the graph."""
    nodes_el = SubElement(graph_el, "nodes")
    for node in nodes:
        n = SubElement(nodes_el, "node")
        n.set("id", node.node_id)
        n.set("label", node.label)
        avs = SubElement(n, "attvalues")
        _attval(avs, "0", node.source)
        _attval(avs, "1", str(node.timestamp) if node.timestamp is not None else "")
        _attval(avs, "2", str(node.decay_weight) if node.decay_weight is not None else "")
        _attval(avs, "3", str(node.cluster_id) if node.cluster_id is not None else "")


def _attval(parent: Element, attr_id: str, value: str) -> None:
    """Append a single attvalue element."""
    av = SubElement(parent, "attvalue")
    av.set("for", attr_id)
    av.set("value", value)


def _add_gexf_edges(graph_el: Element, edges: list[MemoryEdge]) -> None:
    """Append all edge elements to the graph."""
    edges_el = SubElement(graph_el, "edges")
    for idx, edge in enumerate(edges):
        e = SubElement(edges_el, "edge")
        e.set("id", str(idx))
        e.set("source", edge.source_id)
        e.set("target", edge.target_id)
        e.set("weight", str(round(edge.similarity, 6)))


class MemoryGraphExporter:
    """Exports EpisodicMemory (and optionally ItemMemory) as a MemoryGraph."""

    def __init__(self, config: Optional[GraphExportConfig] = None) -> None:
        self._config = config or GraphExportConfig()

    def export(
        self,
        episodic: EpisodicMemory,
        semantic: Optional[ItemMemory] = None,
    ) -> MemoryGraph:
        """Build the graph from memory store(s).

        Steps:
        1. Collect all entries from episodic (and semantic prototypes if provided).
        2. Compute pairwise cosine similarities in FP32 via np.einsum — O(N²).
        3. Add edges where similarity >= threshold (excluding self-loops unless configured).
        4. Return MemoryGraph.

        Raises
        ------
        ValueError
            If n_nodes exceeds config.max_nodes.
        """
        cfg = self._config
        nodes, hvs = _collect_nodes(episodic, semantic, cfg)
        n = len(nodes)
        if n > cfg.max_nodes:
            raise ValueError(
                f"n_nodes={n} exceeds max_nodes={cfg.max_nodes}; "
                "reduce scope or raise GraphExportConfig.max_nodes"
            )
        edges = _compute_edges(nodes, hvs, cfg)
        return MemoryGraph(
            nodes=nodes,
            edges=edges,
            n_nodes=n,
            n_edges=len(edges),
            similarity_threshold=cfg.similarity_threshold,
            exported_at=datetime.now(timezone.utc).isoformat(),
        )

    def save_json(self, graph: MemoryGraph, path: str | Path) -> None:
        """Write graph.to_json() atomically to path. Creates parent dirs."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(graph.to_json(), encoding="utf-8")
        os.replace(tmp, dest)

    def save_gexf(self, graph: MemoryGraph, path: str | Path) -> None:
        """Write graph.to_gexf() atomically to path. Creates parent dirs."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(graph.to_gexf(), encoding="utf-8")
        os.replace(tmp, dest)

    def save(self, graph: MemoryGraph, path: str | Path) -> None:
        """Dispatch by extension: .json -> save_json, .gexf -> save_gexf.

        Raises
        ------
        ValueError
            On unknown file extension.
        """
        dest = Path(path)
        ext = dest.suffix.lower()
        if ext == ".json":
            self.save_json(graph, dest)
        elif ext == ".gexf":
            self.save_gexf(graph, dest)
        else:
            raise ValueError(
                f"unknown file extension {ext!r}; supported: .json, .gexf"
            )


def _collect_nodes(
    episodic: EpisodicMemory,
    semantic: Optional[ItemMemory],
    cfg: GraphExportConfig,
) -> tuple[list[MemoryNode], list[np.ndarray]]:
    """Collect MemoryNode list and parallel FP32 HV arrays from both stores."""
    nodes: list[MemoryNode] = []
    hvs: list[np.ndarray] = []
    now_clock = episodic._timestamp - 1

    for entry in episodic.entries():
        dw = None
        if cfg.decay_config is not None:
            age = max(0, now_clock - entry.timestamp)
            dw = round(_decay_weight(age, cfg.decay_config), 6)
        nodes.append(
            MemoryNode(
                node_id=f"e_{entry.id}",
                label=entry.label,
                source="episodic",
                timestamp=entry.timestamp,
                decay_weight=dw,
                cluster_id=None,
            )
        )
        hvs.append(entry.key.data.astype(np.float32))

    if semantic is not None:
        for label in semantic.labels():
            proto = semantic.get(label)
            if proto is None:
                continue
            nodes.append(
                MemoryNode(
                    node_id=f"s_{label}",
                    label=label,
                    source="semantic",
                    timestamp=None,
                    decay_weight=None,
                    cluster_id=None,
                )
            )
            hvs.append(proto.vector.data.astype(np.float32))

    return nodes, hvs


def _compute_edges(
    nodes: list[MemoryNode],
    hvs: list[np.ndarray],
    cfg: GraphExportConfig,
) -> list[MemoryEdge]:
    """Compute pairwise cosine similarities and return edges above threshold."""
    n = len(nodes)
    if n == 0:
        return []
    mat = np.stack(hvs, axis=0)  # (n, D), FP32
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    normed = mat / norms  # (n, D)
    sims = np.einsum("id,jd->ij", normed, normed)  # (n, n), FP32

    edges: list[MemoryEdge] = []
    for i in range(n):
        start = i if cfg.include_self_loops else i + 1
        for j in range(start, n):
            if i == j and not cfg.include_self_loops:
                continue
            sim = float(sims[i, j])
            if sim >= cfg.similarity_threshold:
                edges.append(
                    MemoryEdge(
                        source_id=nodes[i].node_id,
                        target_id=nodes[j].node_id,
                        similarity=round(sim, 6),
                    )
                )
    return edges
