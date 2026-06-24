"""Analogical memory — relational reasoning via VSA binding algebra (v15).

Registered after ``/consolidate`` and before the multi-agent ``/agents/*`` routes,
matching the original registration order: ``/analogy/extract``, ``/analogy/learn``,
``/analogy/get``, ``/analogy/transfer``, ``/analogy/records``.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from kohaku.extraction import extract_triples

from .._helpers import Ctx, RestState
from ..models import (
    AnalogyExtractRequest,
    AnalogyExtractResponse,
    AnalogyGetRequest,
    AnalogyLearnRequest,
    AnalogyLearnResponse,
    AnalogyRecordsResponse,
    AnalogyResultOut,
    AnalogyTransferRequest,
    CandidateOut,
    TripleOut,
)


def register(app: FastAPI, ctx: Ctx) -> None:
    @app.post("/analogy/extract", response_model=AnalogyExtractResponse)
    def analogy_extract(req: AnalogyExtractRequest) -> AnalogyExtractResponse:
        """Extract (subject, attribute, value) triples from free text.

        Does NOT store anything — pure parse. Returns only high-precision
        triples; the extractor emits nothing for text it cannot parse.
        """
        triples = extract_triples(req.text)
        return AnalogyExtractResponse(
            triples=[
                TripleOut(
                    subject=t.subject,
                    attribute=t.attribute,
                    value=t.value,
                    confidence=t.confidence,
                )
                for t in triples
            ],
            count=len(triples),
        )

    @app.post("/analogy/learn", response_model=AnalogyLearnResponse)
    def analogy_learn(req: AnalogyLearnRequest) -> AnalogyLearnResponse:
        """Extract triples from free text and fold them into the AnalogicalMemory.

        New attributes merge into existing subject records; later mentions
        overwrite the same attribute. Returns the triples that were stored.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            triples = rest.analogy.learn(req.text)
            count = len(rest.analogy)
        return AnalogyLearnResponse(
            triples_learned=len(triples),
            records_count=count,
            triples=[
                TripleOut(
                    subject=t.subject,
                    attribute=t.attribute,
                    value=t.value,
                    confidence=t.confidence,
                )
                for t in triples
            ],
        )

    @app.post("/analogy/get", response_model=AnalogyResultOut)
    def analogy_get(req: AnalogyGetRequest) -> AnalogyResultOut:
        """Recover the value of an attribute from a named record (unbind + cleanup).

        e.g. name="USA", attribute="currency" → AnalogyResult(value="dollar", …).
        Returns 404 when the record is unknown.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            if req.name not in rest.analogy:
                raise HTTPException(
                    status_code=404,
                    detail=f"record '{req.name}' not found",
                )
            result = rest.analogy.get(req.name, req.attribute, top_k=req.top_k)
        return AnalogyResultOut(
            value=result.value,
            confidence=result.confidence,
            margin=result.margin,
            ranked=[CandidateOut(value=v, confidence=c) for v, c in result.ranked],
        )

    @app.post("/analogy/transfer", response_model=AnalogyResultOut)
    def analogy_transfer(req: AnalogyTransferRequest) -> AnalogyResultOut:
        """Analogical transfer: "value of source is to target as…"

        e.g. source="USA", target="Mexico", value="dollar" → "peso".
        Implements the Kanerva (2010) dollar-of-Mexico operation via VSA algebra.
        Returns 404 when either source or target record is unknown.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            for name in (req.source, req.target):
                if name not in rest.analogy:
                    raise HTTPException(
                        status_code=404,
                        detail=f"record '{name}' not found",
                    )
            result = rest.analogy.analogy(
                req.source, req.target, req.value, top_k=req.top_k
            )
        return AnalogyResultOut(
            value=result.value,
            confidence=result.confidence,
            margin=result.margin,
            ranked=[CandidateOut(value=v, confidence=c) for v, c in result.ranked],
        )

    @app.get("/analogy/records", response_model=AnalogyRecordsResponse)
    def analogy_records() -> AnalogyRecordsResponse:
        """List all record names in the AnalogicalMemory."""
        rest: RestState = app.state.rest
        with rest.lock:
            records = rest.analogy.records()
        return AnalogyRecordsResponse(records=records, count=len(records))
