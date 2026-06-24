"""Pydantic request/response models for the kohaku HTTP surface.

Split out of ``api/main.py`` to keep each module within the file-size budget.
All field definitions are byte-for-byte identical to their previous inline form.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from kohaku._pure import DIMS

InputType = Literal["text", "vector"]


class EncodeRequest(BaseModel):
    input: Union[str, List[float]]
    type: InputType = "text"

    @model_validator(mode="after")
    def _check_shape(self) -> "EncodeRequest":
        if self.type == "text" and not isinstance(self.input, str):
            raise ValueError("input must be a string when type='text'")
        if self.type == "vector":
            if not isinstance(self.input, list):
                raise ValueError("input must be a list of floats when type='vector'")
            if len(self.input) != DIMS:
                raise ValueError(
                    f"vector input must have length {DIMS}, got {len(self.input)}"
                )
        return self


class EncodeResponse(BaseModel):
    vector: List[int]
    dims: int


class StoreRequest(BaseModel):
    label: str = Field(..., min_length=1)
    input: Union[str, List[float]]
    type: InputType = "text"

    @model_validator(mode="after")
    def _check_shape(self) -> "StoreRequest":
        if self.type == "text" and not isinstance(self.input, str):
            raise ValueError("input must be a string when type='text'")
        if self.type == "vector" and (
            not isinstance(self.input, list) or len(self.input) != DIMS
        ):
            raise ValueError(f"vector input must be a list of length {DIMS}")
        return self


class StoreResponse(BaseModel):
    id: int
    label: str
    dims: int
    episodic_size: int


class QueryRequest(BaseModel):
    input: Optional[Union[str, List[float]]] = None
    label: Optional[str] = None
    type: InputType = "text"
    top_k: int = Field(5, ge=1, le=100)
    half_life: Optional[float] = Field(
        default=None,
        description="If set, apply Ebbinghaus decay with this half-life (in store ticks).",
        gt=0.0,
    )
    floor: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _exactly_one_probe(self) -> "QueryRequest":
        if (self.input is None) == (self.label is None):
            raise ValueError("provide exactly one of `input` or `label`")
        if (
            self.type == "vector"
            and isinstance(self.input, list)
            and len(self.input) != DIMS
        ):
            raise ValueError(f"vector input must be a list of length {DIMS}")
        return self


class QueryHit(BaseModel):
    entry_id: int
    label: str
    similarity: float
    decayed_similarity: Optional[float] = None


class QueryResponse(BaseModel):
    results: List[QueryHit]
    top_k: int
    decay_applied: bool


class BundleRequest(BaseModel):
    inputs: List[Union[str, List[float]]] = Field(..., min_length=1)
    type: InputType = "text"

    @model_validator(mode="after")
    def _check_shape(self) -> "BundleRequest":
        if self.type == "text" and not all(isinstance(x, str) for x in self.inputs):
            raise ValueError("all inputs must be strings when type='text'")
        if self.type == "vector":
            for v in self.inputs:
                if not isinstance(v, list) or len(v) != DIMS:
                    raise ValueError(
                        f"each vector input must be a list of length {DIMS}"
                    )
        return self


class BundleResponse(BaseModel):
    vector: List[int]
    dims: int
    n_inputs: int


class StatsResponse(BaseModel):
    backend: str
    version: str
    dims: int
    episodic_size: int
    episodic_capacity: int
    semantic_concepts: int
    learning_iterations: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    backend: str


# ── kyro bridge models ────────────────────────────────────────────────────────


class BridgeDoc(BaseModel):
    text: str = Field(..., min_length=1)
    id: Optional[str] = None


class BridgeIngestRequest(BaseModel):
    documents: List[Union[str, BridgeDoc]] = Field(..., min_length=1)


class BridgeIngestResponse(BaseModel):
    entry_ids: List[int]
    total_chunks: int


class BridgeRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)
    half_life: Optional[float] = Field(default=None, gt=0.0)
    floor: float = Field(0.0, ge=0.0, le=1.0)


class BridgeChunk(BaseModel):
    entry_id: int
    doc_id: str
    text: str
    similarity: float
    decayed_similarity: Optional[float] = None
    age: int


class BridgeRetrieveResponse(BaseModel):
    results: List[BridgeChunk]
    total_chunks: int
    decay_applied: bool


# ── Enriched memory request/response models ──────────────────────────────────


class EnrichedStoreRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    input: Union[str, List[float]]
    type: InputType = "text"
    source: str = Field("user_input", min_length=1, max_length=50)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    forgetting_rate: Optional[float] = Field(
        None,
        gt=0.0,
        description="Per-memory decay rate override (> 0). Values > 1 accelerate forgetting; < 1 slow it.",
    )

    @model_validator(mode="after")
    def _check_shape(self) -> "EnrichedStoreRequest":
        if self.type == "text" and not isinstance(self.input, str):
            raise ValueError("input must be a string when type='text'")
        if self.type == "vector" and (
            not isinstance(self.input, list) or len(self.input) != DIMS
        ):
            raise ValueError(f"vector input must be a list of length {DIMS}")
        if self.valid_until is not None and self.valid_from is not None:
            if self.valid_until < self.valid_from:
                raise ValueError("valid_until must be >= valid_from")
        return self


class EnrichedStoreResponse(BaseModel):
    entry_id: int
    label: str
    source: str
    importance: float
    valid_from: str
    valid_until: Optional[str] = None
    total_memories: int


class EnrichedQueryRequest(BaseModel):
    input: Union[str, List[float]]
    type: InputType = "text"
    top_k: int = Field(5, ge=1, le=100)
    sort: Literal["similarity", "salience", "recency"] = "similarity"
    source_filter: Optional[str] = Field(None, max_length=50)
    include_expired: bool = False
    min_similarity: Optional[float] = Field(None, ge=-1.0, le=1.0)
    reinforce_hits: bool = True
    tags_any: List[str] = Field(default_factory=list)
    tags_all: List[str] = Field(default_factory=list)


class EnrichedQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    top_k: int
    sort: str


# ── Phase 13 P2 models ────────────────────────────────────────────────────────


class EpisodeStoreRequest(BaseModel):
    label: str = Field(..., min_length=1)
    who: Optional[List[float]] = None
    what: Optional[List[float]] = None
    when: Optional[List[float]] = None
    where: Optional[List[float]] = None


class EpisodeStoreResponse(BaseModel):
    entry_id: int
    label: str


class EpisodeQueryRequest(BaseModel):
    who: Optional[List[float]] = None
    what: Optional[List[float]] = None
    when: Optional[List[float]] = None
    where: Optional[List[float]] = None
    top_k: int = Field(5, ge=1, le=100)


class EpisodeQueryResponse(BaseModel):
    results: List[Dict[str, Any]]


class ChainQueryRequest(BaseModel):
    start: Union[str, List[float]]
    type: InputType = "text"
    hops: int = Field(3, ge=1, le=20)
    min_similarity: float = Field(0.0, ge=-1.0, le=1.0)


class ChainQueryResponse(BaseModel):
    hops: List[Dict[str, Any]]
    terminated_early: bool


class ValidateRequest(BaseModel):
    input: Union[str, List[float]]
    type: InputType = "text"
    source: Optional[str] = None


class ValidateResponse(BaseModel):
    accepted: bool
    reason: str
    nearest_similarity: float
    nearest_label: str


class ConsolidateRequest(BaseModel):
    similarity_threshold: Optional[float] = Field(None, ge=-1.0, le=1.0)


class ConsolidateResponse(BaseModel):
    started_at: str
    run_seconds: float
    episodes_before: int
    episodes_after: int
    episodes_consolidated: int
    prototypes_created: int
    memory_freed: int
    similarity_threshold: float


# ── Multi-agent pool models ────────────────────────────────────────────────


class AgentStoreRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=1000)
    label: str = Field(default="")


class AgentStoreResponse(BaseModel):
    agent_id: str
    stored: bool
    reason: str
    size: int


class AgentQueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    agents: Optional[List[str]] = None


class AgentQueryResult(BaseModel):
    agent_id: str
    entry_id: int
    label: str
    similarity: float


class AgentQueryResponse(BaseModel):
    results: List[AgentQueryResult]


class TenantStoreRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=1000)
    label: str = Field(default="")


class TenantStoreResponse(BaseModel):
    tenant_id: str
    stored: bool
    size: int


class TenantQueryRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class TenantQueryResult(BaseModel):
    entry_id: int
    label: str
    similarity: float


class TenantQueryResponse(BaseModel):
    tenant_id: str
    results: List[TenantQueryResult]


class NamespaceInfo(BaseModel):
    id: str
    size: int


class NamespaceListResponse(BaseModel):
    namespaces: List[NamespaceInfo]
    count: int
    total_size: int


# ── Analogical memory models ───────────────────────────────────────────────


class AnalogyExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class TripleOut(BaseModel):
    subject: str
    attribute: str
    value: str
    confidence: float


class AnalogyExtractResponse(BaseModel):
    triples: List[TripleOut]
    count: int


class AnalogyLearnRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class AnalogyLearnResponse(BaseModel):
    triples_learned: int
    records_count: int
    triples: List[TripleOut]


class AnalogyGetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    attribute: str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(default=3, ge=1, le=10)


class CandidateOut(BaseModel):
    value: str
    confidence: float


class AnalogyResultOut(BaseModel):
    value: str
    confidence: float
    margin: float
    ranked: List[CandidateOut]


class AnalogyTransferRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=200)
    target: str = Field(..., min_length=1, max_length=200)
    value: str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(default=3, ge=1, le=10)


class AnalogyRecordsResponse(BaseModel):
    records: List[str]
    count: int
