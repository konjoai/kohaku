"""Phase 13 P2: episodic binding, multi-hop chaining, write validation, plus
the synchronous sleep-phase consolidation trigger.

Registered after the ``/memories/*`` routes, matching the original order:
``/episodes/store``, ``/episodes/query``, ``/chain``, ``/memories/validate``,
``/consolidate``.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException

from kohaku import HyperVector, chain_query, encode_text

from .._helpers import Ctx, RestState
from ..models import (
    ChainQueryRequest,
    ChainQueryResponse,
    ConsolidateRequest,
    ConsolidateResponse,
    EpisodeQueryRequest,
    EpisodeQueryResponse,
    EpisodeStoreRequest,
    EpisodeStoreResponse,
    ValidateRequest,
    ValidateResponse,
)


def register(app: FastAPI, ctx: Ctx) -> None:
    @app.post("/episodes/store", response_model=EpisodeStoreResponse)
    def episodes_store(req: EpisodeStoreRequest) -> EpisodeStoreResponse:
        """Store an episode bound from who / what / when / where role HVs.

        Each provided role vector is binarized and bound with its fixed role HV;
        the resulting bundle is stored as a single composite hypervector.
        """

        def _to_hv(vals: Optional[List[float]]) -> Optional[HyperVector]:
            return ctx.vec_input_to_hv(vals) if vals is not None else None

        who_hv = _to_hv(req.who)
        what_hv = _to_hv(req.what)
        when_hv = _to_hv(req.when)
        where_hv = _to_hv(req.where)
        if all(v is None for v in (who_hv, what_hv, when_hv, where_hv)):
            raise HTTPException(
                status_code=422, detail="At least one role must be provided"
            )
        rest: RestState = app.state.rest
        with rest.lock:
            try:
                eid = rest.episodes.store_episode(
                    req.label, who=who_hv, what=what_hv, when=when_hv, where=where_hv
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EpisodeStoreResponse(entry_id=eid, label=req.label)

    @app.post("/episodes/query", response_model=EpisodeQueryResponse)
    def episodes_query(req: EpisodeQueryRequest) -> EpisodeQueryResponse:
        """Retrieve episodes matching a partial role cue.

        Supply any subset of who / what / when / where; the query composite is
        built from those roles only, enabling partial-cue retrieval.
        """

        def _to_hv(vals: Optional[List[float]]) -> Optional[HyperVector]:
            return ctx.vec_input_to_hv(vals) if vals is not None else None

        who_hv = _to_hv(req.who)
        what_hv = _to_hv(req.what)
        when_hv = _to_hv(req.when)
        where_hv = _to_hv(req.where)
        if all(v is None for v in (who_hv, what_hv, when_hv, where_hv)):
            raise HTTPException(
                status_code=422, detail="At least one role must be provided"
            )
        rest: RestState = app.state.rest
        with rest.lock:
            try:
                results = rest.episodes.query_episode(
                    who=who_hv,
                    what=what_hv,
                    when=when_hv,
                    where=where_hv,
                    top_k=req.top_k,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EpisodeQueryResponse(
            results=[
                {
                    "entry_id": r.entry_id,
                    "label": r.label,
                    "similarity": r.similarity,
                }
                for r in results
            ]
        )

    @app.post("/chain", response_model=ChainQueryResponse)
    def chain_endpoint(req: ChainQueryRequest) -> ChainQueryResponse:
        """Multi-hop associative chain starting from a text or vector query.

        Each hop retrieves the highest-similarity unvisited entry, then follows
        that entry's key HV to the next hop.
        """
        if req.type == "text":
            start_hv = encode_text(req.start if isinstance(req.start, str) else "")
        else:
            if not isinstance(req.start, list):
                raise HTTPException(
                    status_code=422, detail="start must be a list when type='vector'"
                )
            start_hv = ctx.vec_input_to_hv(req.start)
        rest: RestState = app.state.rest
        with rest.lock:
            result = chain_query(
                rest.episodic,
                start_hv,
                hops=req.hops,
                min_similarity=req.min_similarity,
            )
        return ChainQueryResponse(
            hops=[
                {
                    "hop": h.hop,
                    "entry_id": h.entry_id,
                    "label": h.label,
                    "similarity": h.similarity,
                }
                for h in result.hops
            ],
            terminated_early=result.terminated_early,
        )

    @app.post("/memories/validate", response_model=ValidateResponse)
    def memories_validate(req: ValidateRequest) -> ValidateResponse:
        """Dry-run validation: check if a vector would be accepted by the write validator.

        Returns accepted=True/False, rejection reason, and nearest existing entry info.
        Does NOT store anything or consume a rate-limit slot.
        """
        if req.type == "text":
            key_hv = encode_text(req.input if isinstance(req.input, str) else "")
        else:
            if not isinstance(req.input, list):
                raise HTTPException(
                    status_code=422, detail="input must be a list when type='vector'"
                )
            key_hv = ctx.vec_input_to_hv(req.input)
        rest: RestState = app.state.rest
        with rest.lock:
            result = rest.validator.validate(key_hv, source=req.source)
        return ValidateResponse(
            accepted=result.accepted,
            reason=result.reason,
            nearest_similarity=result.nearest_similarity,
            nearest_label=result.nearest_label,
        )

    # ── Sleep-phase consolidation ──────────────────────────────────────────
    @app.post("/consolidate", response_model=ConsolidateResponse)
    def consolidate_endpoint(
        req: ConsolidateRequest = ConsolidateRequest(),
    ) -> ConsolidateResponse:
        """Trigger a one-shot sleep-phase consolidation pass.

        Runs synchronously over the enriched store's episodic memory:
        finds clusters with pairwise cosine >= `similarity_threshold`
        (default 0.85), merges them into semantic prototypes, returns the
        structured `SleepReport`.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            # The daemon owns its own lock too, but we hold the RestState lock
            # to serialize against /memories/store.
            if req.similarity_threshold is not None:
                rest.sleep._threshold = req.similarity_threshold
            try:
                report = rest.sleep.run_once()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ConsolidateResponse(**report.to_dict())
