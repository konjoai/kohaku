"""Enriched memory endpoints — store/query/list, tags, conflicts, portability.

Covers the first contiguous slice of the ``/memories/*`` surface, exactly in the
original registration order. The remaining ``/memories/*`` routes (provenance,
search/timeline/recent/health, versioning, relationships, batch ops) live in
``memories_extra`` and are registered immediately after this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query

from .._helpers import Ctx, RestState
from ..models import (
    EnrichedQueryRequest,
    EnrichedQueryResponse,
    EnrichedStoreRequest,
    EnrichedStoreResponse,
)


def register(app: FastAPI, ctx: Ctx) -> None:
    @app.post("/memories/store", response_model=EnrichedStoreResponse)
    def memories_store(req: EnrichedStoreRequest) -> EnrichedStoreResponse:
        rest: RestState = app.state.rest
        hv = ctx.encode(req.input, req.type)
        with rest.lock:
            eid = rest.enriched.store(
                hv,
                hv,
                req.label,
                source=req.source,
                importance=req.importance,
                valid_from=req.valid_from,
                valid_until=req.valid_until,
                tags=req.tags,
                forgetting_rate=req.forgetting_rate,
            )
            total = len(rest.enriched)
        return EnrichedStoreResponse(
            entry_id=eid,
            label=req.label,
            source=req.source,
            importance=req.importance,
            valid_from=(req.valid_from or datetime.now(timezone.utc)).isoformat(),
            valid_until=req.valid_until.isoformat() if req.valid_until else None,
            total_memories=total,
        )

    @app.post("/memories/query", response_model=EnrichedQueryResponse)
    def memories_query(req: EnrichedQueryRequest) -> EnrichedQueryResponse:
        rest: RestState = app.state.rest
        with rest.lock:
            if len(rest.enriched) == 0:
                return EnrichedQueryResponse(results=[], top_k=req.top_k, sort=req.sort)
            probe = ctx.encode(req.input, req.type)
            try:
                results = rest.enriched.query(
                    probe,
                    top_k=req.top_k,
                    sort=req.sort,
                    source_filter=req.source_filter,
                    include_expired=req.include_expired,
                    min_similarity=req.min_similarity,
                    reinforce_hits=req.reinforce_hits,
                    tags_any=req.tags_any or None,
                    tags_all=req.tags_all or None,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EnrichedQueryResponse(
            results=[r.to_dict() for r in results],
            top_k=req.top_k,
            sort=req.sort,
        )

    @app.get("/memories")
    def memories_list(
        sort: str = Query("salience", description="Sort: salience | recency"),
        source: Optional[str] = Query(None, description="Filter by source"),
        limit: int = Query(50, ge=1, le=1000),
        include_expired: bool = Query(False),
        tags: Optional[str] = Query(
            None, description="Comma-separated tags (any-match)"
        ),
        tags_all: Optional[str] = Query(
            None, description="Comma-separated tags (all-match)"
        ),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        if sort not in ("salience", "recency"):
            raise HTTPException(
                status_code=400, detail="sort must be 'salience' or 'recency'"
            )
        any_list = [t.strip() for t in (tags or "").split(",") if t.strip()] or None
        all_list = [t.strip() for t in (tags_all or "").split(",") if t.strip()] or None
        with rest.lock:
            items = rest.enriched.list_memories(
                sort=sort,
                source_filter=source,
                include_expired=include_expired,
                limit=limit,
                tags_any=any_list,
                tags_all=all_list,
            )
            total = len(rest.enriched)
        return {
            "items": items,
            "count": len(items),
            "total": total,
            "sort": sort,
            "source_filter": source,
            "include_expired": include_expired,
        }

    @app.post("/memories/expire")
    def memories_expire() -> Dict[str, Any]:
        """Drop all memories whose ``valid_until`` is in the past. Returns the
        list of dropped entry ids."""
        rest: RestState = app.state.rest
        with rest.lock:
            dropped = rest.enriched.expire_old()
            remaining = len(rest.enriched)
        return {
            "dropped_ids": dropped,
            "dropped_count": len(dropped),
            "remaining": remaining,
        }

    @app.get("/memories/trust-weights")
    def memories_trust_weights() -> Dict[str, float]:
        rest: RestState = app.state.rest
        # return a copy of the live table so the caller can see what's in effect
        return dict(rest.enriched.trust_weights)

    # ── Tags ───────────────────────────────────────────────────────────────

    @app.get("/memories/tags")
    def memories_tags_index() -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            counts = rest.enriched.all_tags()
        return {
            "tags": counts,
            "count": len(counts),
            "total_uses": sum(counts.values()),
        }

    @app.get("/memories/{memory_id}/tags")
    def memories_tags_get(memory_id: int) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        with rest.lock:
            tags = rest.enriched.get_tags(memory_id)
        if tags is None:
            raise HTTPException(
                status_code=404, detail=f"unknown memory_id {memory_id}"
            )
        return {"entry_id": memory_id, "tags": sorted(tags)}

    @app.post("/memories/{memory_id}/tags")
    def memories_tags_add(
        memory_id: int,
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        raw = payload.get("tags") or []
        if not isinstance(raw, list):
            raise HTTPException(
                status_code=400, detail="'tags' must be a list of strings"
            )
        rest: RestState = app.state.rest
        with rest.lock:
            updated = rest.enriched.add_tags(memory_id, [str(t) for t in raw])
        if updated is None:
            raise HTTPException(
                status_code=404, detail=f"unknown memory_id {memory_id}"
            )
        return {"entry_id": memory_id, "tags": sorted(updated)}

    @app.delete("/memories/{memory_id}/tags")
    def memories_tags_remove(
        memory_id: int,
        tag: Optional[str] = Query(None, description="Comma-separated tags"),
    ) -> Dict[str, Any]:
        if not tag:
            raise HTTPException(status_code=400, detail="'tag' query param is required")
        rest: RestState = app.state.rest
        tags = [t.strip() for t in tag.split(",") if t.strip()]
        with rest.lock:
            updated = rest.enriched.remove_tags(memory_id, tags)
        if updated is None:
            raise HTTPException(
                status_code=404, detail=f"unknown memory_id {memory_id}"
            )
        return {"entry_id": memory_id, "tags": sorted(updated)}

    # ── Conflict detection ────────────────────────────────────────────────

    @app.get("/memories/conflicts")
    def memories_conflicts(
        similarity_threshold: float = Query(0.40, gt=0.0, le=1.0),
        contradiction_threshold: float = Query(0.45, gt=0.0, le=1.0),
        max_pairs: int = Query(100, ge=0, le=1000),
    ) -> Dict[str, Any]:
        from kohaku.conflicts import detect_conflicts

        rest: RestState = app.state.rest
        with rest.lock:
            pairs = detect_conflicts(
                rest.enriched,
                similarity_threshold=similarity_threshold,
                contradiction_threshold=contradiction_threshold,
                max_pairs=max_pairs,
            )
        return {
            "pairs": [p.to_dict() for p in pairs],
            "count": len(pairs),
            "similarity_threshold": similarity_threshold,
            "contradiction_threshold": contradiction_threshold,
        }

    @app.post("/memories/conflicts/resolve")
    def memories_conflicts_resolve(
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        from kohaku.conflicts import resolve_conflict

        try:
            a_id = int(payload["a_id"])
            b_id = int(payload["b_id"])
            keep = str(payload.get("keep", "both"))
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid payload: {exc}"
            ) from exc
        rest: RestState = app.state.rest
        with rest.lock:
            try:
                outcome = resolve_conflict(
                    rest.enriched, a_id=a_id, b_id=b_id, keep=keep
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return outcome.to_dict()

    # ── Export / import (portability) ─────────────────────────────────────

    @app.get("/memories/export")
    def memories_export(format: str = Query("json")) -> Dict[str, Any]:
        from kohaku.portability import export_memories

        rest: RestState = app.state.rest
        with rest.lock:
            try:
                bundle = export_memories(rest.enriched, fmt=format)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return bundle.to_dict()

    @app.post("/memories/import")
    def memories_import(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        from kohaku.portability import import_memories

        body = payload.get("payload")
        if body is None and "memories" in payload:
            # accept a bare envelope for convenience
            import json as _json

            body = _json.dumps(payload)
        if not isinstance(body, str):
            raise HTTPException(
                status_code=400, detail="'payload' (JSON string) is required"
            )
        dedup = float(payload.get("dedup_threshold", 0.99))
        rest: RestState = app.state.rest
        with rest.lock:
            try:
                report = import_memories(rest.enriched, body, dedup_threshold=dedup)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.to_dict()
