"""Memory graph export endpoints (/export/graph[...])."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, Query
from fastapi.responses import Response

from kohaku import GraphExportConfig, MemoryGraphExporter

from .._helpers import Ctx, RestState


def register(app: FastAPI, ctx: Ctx) -> None:
    # ── Memory graph export ──────────────────────────────────────────────
    @app.get("/export/graph")
    def export_graph(
        format: str = Query("json", description="Output format hint (json or gexf)"),
        threshold: float = Query(0.3, ge=-1.0, le=1.0),
    ) -> Dict[str, Any]:
        """Export the live episodic + semantic memory as a graph (JSON).

        Always returns JSON over HTTP. Use /export/graph/gexf for XML output.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            cfg = GraphExportConfig(similarity_threshold=threshold)
            exporter = MemoryGraphExporter(cfg)
            graph = exporter.export(rest.episodic, semantic=rest.semantic)
        return graph.to_dict()

    @app.get("/export/graph/gexf")
    def export_graph_gexf(
        threshold: float = Query(0.3, ge=-1.0, le=1.0),
    ) -> Response:
        """Export the live episodic + semantic memory as GEXF 1.3 XML."""
        rest: RestState = app.state.rest
        with rest.lock:
            cfg = GraphExportConfig(similarity_threshold=threshold)
            exporter = MemoryGraphExporter(cfg)
            graph = exporter.export(rest.episodic, semantic=rest.semantic)
        return Response(content=graph.to_gexf(), media_type="application/xml")

    @app.get("/export/graph/graphiti")
    def export_graph_graphiti(
        threshold: float = Query(0.3, ge=-1.0, le=1.0),
    ) -> Dict[str, Any]:
        """Export as a Graphiti-compatible graph (episodes + relations)."""
        rest: RestState = app.state.rest
        with rest.lock:
            cfg = GraphExportConfig(similarity_threshold=threshold)
            exporter = MemoryGraphExporter(cfg)
            graph = exporter.export(rest.episodic, semantic=rest.semantic)
        return graph.to_graphiti()

    @app.get("/export/graph/mem0")
    def export_graph_mem0(
        threshold: float = Query(0.3, ge=-1.0, le=1.0),
    ) -> Dict[str, Any]:
        """Export as a Mem0-compatible memory list."""
        rest: RestState = app.state.rest
        with rest.lock:
            cfg = GraphExportConfig(similarity_threshold=threshold)
            exporter = MemoryGraphExporter(cfg)
            graph = exporter.export(rest.episodic, semantic=rest.semantic)
        return graph.to_mem0()
