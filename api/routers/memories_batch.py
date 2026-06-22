"""Phase 17: typed relationships, importance rescore, and bulk (batch) ops.

Registered immediately after ``memories_extra`` so the overall ``/memories/*``
registration order is byte-for-byte identical to the original ``api/main.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query

from kohaku import (
    batch_delete_by_filter,
    batch_delete_by_ids,
    batch_export,
    batch_update,
    rescore_all,
)

from .._helpers import Ctx, RestState


def register(app: FastAPI, ctx: Ctx) -> None:
    # ── Phase 17: relationships, importance scoring, bulk ops ─────────────

    @app.post("/memories/{memory_id}/relate")
    def memories_relate(
        memory_id: int,
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        target_id = payload.get("target_id")
        relation_type = payload.get("relation_type")
        if target_id is None or relation_type is None:
            raise HTTPException(
                status_code=400,
                detail="target_id and relation_type are required",
            )
        try:
            target_id = int(target_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid target_id: {exc}"
            ) from exc
        live_ids = {e.id for e in rest.enriched.episodic.entries()}
        if memory_id not in live_ids:
            raise HTTPException(
                status_code=404, detail=f"unknown memory_id {memory_id}"
            )
        if target_id not in live_ids:
            raise HTTPException(
                status_code=404, detail=f"unknown target_id {target_id}"
            )
        try:
            rel = rest.relationships.record(
                source_id=memory_id,
                target_id=target_id,
                relation_type=str(relation_type),
                metadata=payload.get("metadata"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return rel.to_dict()

    @app.get("/memories/{memory_id}/related")
    def memories_related(
        memory_id: int,
        relation_type: Optional[str] = Query(None),
        direction: str = Query("both", description="outgoing | incoming | both"),
    ) -> Dict[str, Any]:
        if direction not in ("outgoing", "incoming", "both"):
            raise HTTPException(
                status_code=400, detail="direction must be outgoing | incoming | both"
            )
        rest: RestState = app.state.rest
        with rest.lock:
            if direction == "outgoing":
                rels = rest.relationships.list_outgoing(memory_id, relation_type)
            elif direction == "incoming":
                rels = rest.relationships.list_incoming(memory_id, relation_type)
            else:
                rels = rest.relationships.list_related(memory_id, relation_type)
        return {
            "memory_id": memory_id,
            "direction": direction,
            "relation_type": relation_type,
            "count": len(rels),
            "relationships": [r.to_dict() for r in rels],
        }

    @app.delete("/memories/{memory_id}/relate")
    def memories_relate_delete(
        memory_id: int,
        target_id: int = Query(...),
        relation_type: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            deleted = rest.relationships.delete(
                memory_id,
                target_id,
                relation_type,
            )
        return {
            "source_id": memory_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "deleted": int(deleted),
        }

    @app.get("/memories/relationships")
    def memories_relationships_index(
        relation_type: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            if relation_type is not None:
                rels = rest.relationships.list_by_type(relation_type)
            else:
                # paginate via counts only — full dump could be huge
                rels = []
                for rt in rest.relationships.counts_by_type():
                    rels.extend(rest.relationships.list_by_type(rt))
            counts = rest.relationships.counts_by_type()
        return {
            "count": len(rels),
            "counts_by_type": counts,
            "relationships": [r.to_dict() for r in rels],
        }

    @app.post("/memories/rescore")
    def memories_rescore(
        payload: Optional[Dict[str, Any]] = Body(None),
    ) -> Dict[str, Any]:
        """Recompute importance from frequency / recency / uniqueness / depth.

        Body is optional; supported keys: ``dry_run``, ``half_life_days``,
        ``freq_cap``, ``blend_alpha``, ``weights`` (dict).
        """
        payload = payload or {}
        rest: RestState = app.state.rest
        kwargs: Dict[str, Any] = {}
        for k in ("half_life_days", "freq_cap", "blend_alpha"):
            if k in payload:
                kwargs[k] = payload[k]
        if "weights" in payload:
            kwargs["weights"] = payload["weights"]
        dry_run = bool(payload.get("dry_run", False))
        with rest.lock:
            try:
                report = rescore_all(
                    rest.enriched,
                    provenance=rest.provenance,
                    dry_run=dry_run,
                    **kwargs,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.to_dict()

    @app.post("/memories/batch-update")
    def memories_batch_update(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        updates = payload.get("updates")
        if not isinstance(updates, list):
            raise HTTPException(
                status_code=400, detail="'updates' must be a list of dicts"
            )
        editor = payload.get("editor")
        rest: RestState = app.state.rest
        with rest.lock:
            try:
                report = batch_update(
                    rest.enriched,
                    rest.versions,
                    updates,
                    editor=editor,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.to_dict()

    @app.post("/memories/batch-delete")
    def memories_batch_delete(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        ids = payload.get("ids")
        filt = payload.get("filter")
        rest: RestState = app.state.rest
        if ids is not None and filt is not None:
            raise HTTPException(
                status_code=400,
                detail="provide exactly one of 'ids' or 'filter'",
            )
        if ids is None and filt is None:
            raise HTTPException(
                status_code=400,
                detail="provide exactly one of 'ids' or 'filter'",
            )
        with rest.lock:
            try:
                if ids is not None:
                    if not isinstance(ids, list):
                        raise ValueError("'ids' must be a list of integers")
                    report = batch_delete_by_ids(
                        rest.enriched,
                        ids,
                        relationships=rest.relationships,
                    )
                else:
                    if not isinstance(filt, dict):
                        raise ValueError("'filter' must be an object")
                    report = batch_delete_by_filter(
                        rest.enriched,
                        stale_days=filt.get("stale_days"),
                        older_than_days=filt.get("older_than_days"),
                        source=filt.get("source"),
                        tags_any=filt.get("tags_any"),
                        max_importance=filt.get("max_importance"),
                        relationships=rest.relationships,
                    )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.to_dict()

    @app.post("/memories/batch-export")
    def memories_batch_export(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        ids = payload.get("ids")
        fmt = str(payload.get("format", "json")).lower()
        if not isinstance(ids, list) or not ids:
            raise HTTPException(
                status_code=400, detail="'ids' must be a non-empty list"
            )
        rest: RestState = app.state.rest
        with rest.lock:
            try:
                bundle = batch_export(rest.enriched, ids, fmt=fmt)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return bundle.to_dict()
