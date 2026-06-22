"""Remaining ``/memories/*`` endpoints: provenance, time-range views, health,
and versioning + consolidation history.

Registered immediately after ``memories`` so the overall ``/memories/*``
registration order is byte-for-byte identical to the original ``api/main.py``.
The Phase 17 relationships/rescore/batch routes follow in ``memories_batch``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query

from kohaku import (
    MemoryHealthAnalyzer,
    ProvenanceGraph,
    TimeFilter,
    apply_time_filter,
    bucket_timeline,
    encode_text,
    filter_recent,
    update_memory,
)

from .._helpers import Ctx, RestState


def register(app: FastAPI, ctx: Ctx) -> None:
    # ── Provenance, time-range, health (P2 wave) ───────────────────────────

    @app.get("/memories/{memory_id}/provenance")
    def memories_provenance(
        memory_id: int,
        direction: str = Query("both", description="ancestors | descendants | both"),
        max_depth: int = Query(5, ge=1, le=32),
    ) -> Dict[str, Any]:
        if direction not in ("ancestors", "descendants", "both"):
            raise HTTPException(
                status_code=400,
                detail="direction must be 'ancestors', 'descendants', or 'both'",
            )
        rest: RestState = app.state.rest
        graph: ProvenanceGraph = rest.provenance
        if not graph.has(memory_id):
            raise HTTPException(
                status_code=404,
                detail=f"no provenance record for memory_id={memory_id}",
            )
        if direction == "ancestors":
            nodes = graph.get_ancestors(memory_id, max_depth=max_depth)
            return {
                "root_id": str(memory_id),
                "direction": direction,
                "max_depth": max_depth,
                "ancestors": [n.to_dict() for n in nodes],
                "descendants": [],
                "edges": [],
                "nodes": [n.to_dict() for n in nodes],
            }
        if direction == "descendants":
            nodes = graph.get_descendants(memory_id, max_depth=max_depth)
            return {
                "root_id": str(memory_id),
                "direction": direction,
                "max_depth": max_depth,
                "ancestors": [],
                "descendants": [n.to_dict() for n in nodes],
                "edges": [],
                "nodes": [n.to_dict() for n in nodes],
            }
        result = graph.get_full_graph(memory_id, max_depth=max_depth)
        out = result.to_dict()
        out["direction"] = "both"
        out["max_depth"] = max_depth
        return out

    @app.get("/memories/search")
    def memories_search(
        q: Optional[str] = Query(None, description="Free-text query"),
        valid_after: Optional[str] = Query(None),
        valid_before: Optional[str] = Query(None),
        source: Optional[str] = Query(None),
        sort: str = Query("salience"),
        limit: int = Query(50, ge=1, le=1000),
        include_expired: bool = Query(False),
    ) -> Dict[str, Any]:
        if sort not in ("salience", "recency", "similarity"):
            raise HTTPException(
                status_code=400, detail="sort must be salience | recency | similarity"
            )
        try:
            tf = TimeFilter.from_iso(valid_after, valid_before)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rest: RestState = app.state.rest
        with rest.lock:
            base_sort = sort if sort in ("salience", "recency") else "salience"
            items = rest.enriched.list_memories(
                sort=base_sort,
                source_filter=source,
                include_expired=include_expired,
                limit=None,
            )
            items = apply_time_filter(items, tf)
            if q and sort == "similarity":
                probe = encode_text(q)
                results = rest.enriched.query(
                    probe,
                    top_k=max(limit, 1),
                    source_filter=source,
                    include_expired=include_expired,
                    reinforce_hits=False,
                )
                live_ids = {it["entry_id"] for it in items}
                items = [r.to_dict() for r in results if r.entry_id in live_ids]
            items = items[:limit]
        return {
            "items": items,
            "count": len(items),
            "sort": sort,
            "valid_after": valid_after,
            "valid_before": valid_before,
            "q": q,
            "source_filter": source,
            "include_expired": include_expired,
        }

    @app.get("/memories/timeline")
    def memories_timeline(
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None),
        bucket: str = Query("day"),
        preview_per_bucket: int = Query(5, ge=0, le=50),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            items = rest.enriched.list_memories(
                sort="recency",
                include_expired=True,
                limit=None,
            )
        try:
            buckets = bucket_timeline(
                items,
                start=start,
                end=end,
                bucket=bucket,
                preview_per_bucket=preview_per_bucket,
                text_field="label",
                id_field="entry_id",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "buckets": [b.to_dict() for b in buckets],
            "bucket": bucket,
            "start": start,
            "end": end,
            "total": sum(b.count for b in buckets),
        }

    @app.get("/memories/recent")
    def memories_recent(
        limit: int = Query(20, ge=1, le=500),
        since_hours: float = Query(24.0, gt=0.0, le=24.0 * 365),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            items = rest.enriched.list_memories(
                sort="recency",
                include_expired=True,
                limit=None,
            )
        try:
            recent = filter_recent(items, since_hours=since_hours, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "items": recent,
            "count": len(recent),
            "since_hours": since_hours,
            "limit": limit,
        }

    @app.get("/memories/health")
    def memories_health(
        stale_days: int = Query(30, ge=1, le=3650),
        duplicate_threshold: float = Query(0.95, gt=0.0, le=1.0),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            analyzer = MemoryHealthAnalyzer(
                rest.enriched,
                provenance=rest.provenance,
                stale_days=stale_days,
                duplicate_threshold=duplicate_threshold,
            )
            report = analyzer.compute()
        return report.to_dict()

    @app.get("/memories/health/stale")
    def memories_health_stale(
        days: int = Query(30, ge=1, le=3650),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            analyzer = MemoryHealthAnalyzer(
                rest.enriched,
                provenance=rest.provenance,
                stale_days=days,
            )
            stale = analyzer.list_stale(days=days)
        return {
            "items": [s.to_dict() for s in stale],
            "count": len(stale),
            "days": days,
        }

    @app.delete("/memories/stale")
    def memories_stale_delete(
        days: int = Query(30, ge=1, le=3650),
        dry_run: bool = Query(True),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            analyzer = MemoryHealthAnalyzer(
                rest.enriched,
                provenance=rest.provenance,
                stale_days=days,
            )
            return analyzer.delete_stale(days=days, dry_run=dry_run)

    # ── Phase 16: memory versioning + consolidation history ────────────────

    @app.put("/memories/{memory_id}")
    def memories_update(
        memory_id: int,
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """Apply an edit to a memory and append a new version snapshot.

        Editable fields: ``label``, ``source``, ``importance``, ``tags``,
        ``valid_until``. Fields not present in the payload are preserved.
        """
        rest: RestState = app.state.rest
        # Build the kwargs dict so unspecified fields stay at their sentinel.
        kwargs: Dict[str, Any] = {}
        for field_name in ("label", "source", "importance", "tags", "valid_until"):
            if field_name in payload:
                kwargs[field_name] = payload[field_name]
        editor = payload.get("editor")
        with rest.lock:
            try:
                result = update_memory(
                    rest.enriched,
                    memory_id,
                    rest.versions,
                    editor=editor,
                    **kwargs,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/memories/{memory_id}/versions")
    def memories_versions(memory_id: int) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            versions = rest.versions.list_versions(memory_id)
        if not versions:
            raise HTTPException(
                status_code=404,
                detail=f"no version history for memory_id {memory_id}",
            )
        return {
            "memory_id": memory_id,
            "count": len(versions),
            "versions": [v.to_dict() for v in versions],
        }

    @app.get("/memories/{memory_id}/versions/{version}")
    def memories_version_get(memory_id: int, version: int) -> Dict[str, Any]:
        if version <= 0:
            raise HTTPException(status_code=400, detail="version must be >= 1")
        rest: RestState = app.state.rest
        with rest.lock:
            snapshot = rest.versions.get_version(memory_id, version)
        if snapshot is None:
            raise HTTPException(
                status_code=404,
                detail=f"no version {version} for memory_id {memory_id}",
            )
        return snapshot.to_dict()

    @app.get("/memories/consolidation/history")
    def consolidation_history(
        limit: int = Query(50, ge=1, le=500),
    ) -> Dict[str, Any]:
        """All sleep-phase consolidation runs in this process, newest-first."""
        rest: RestState = app.state.rest
        with rest.lock:
            reports = list(rest.sleep.reports())
            # newest-first cap
            tail = reports[-limit:][::-1]
            return {
                "count": len(tail),
                "total_runs": rest.sleep.run_count,
                "reports": [r.to_dict() for r in tail],
            }
