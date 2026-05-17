"""Kohaku — HDC episodic memory. Uses Rust extension when available, pure-Python otherwise."""
from __future__ import annotations

__version__ = "0.10.0"

try:
    from kohaku._kohaku_rs import HyperVector, EpisodicMemory  # compiled Rust ext
    _BACKEND = "rust"
except ImportError:
    from kohaku._pure import HyperVector, EpisodicMemory  # pure Python fallback
    _BACKEND = "python"

from kohaku._async import AsyncEpisodicMemory
from kohaku._query import RetrievalResult, query, query_threshold
from kohaku.context import ContextConfig, ContextMemoryManager
from kohaku.attention import attention_weighted_encode, encode_text
from kohaku.openai_compat import MemoryMiddleware
from kohaku.persistence import (
    save,
    load,
    save_json,
    load_json,
    save_binary,
    load_binary,
)
from kohaku.consolidation import Cluster, consolidate, consolidate_to_memory
from kohaku.decay import DecayConfig, decay_weight, query_with_decay
from kohaku.learning import ItemMemory, Prototype
from kohaku.hopfield import HopfieldAssociator, HopfieldRecall
from kohaku.memory_system import CombinedRecall, MemorySystem
from kohaku.streaming import StreamingConsolidator
from kohaku.compaction import find_duplicates, deduplicate, compact
from kohaku.tenant import TenantMemoryStore
from kohaku.kyro_bridge import HDCRetriever, RetrievedChunk
from kohaku.graph_export import (
    GraphExportConfig,
    MemoryGraphExporter,
    MemoryGraph,
    MemoryNode,
    MemoryEdge,
)
from kohaku.enriched import (
    EnrichedMemoryStore,
    EnrichedRetrievalResult,
    MemoryMetadata,
    SOURCE_TRUST_WEIGHTS,
)
from kohaku.sleep import SleepConsolidator, SleepReport
from kohaku.provenance import (
    ProvenanceGraph,
    ProvenanceNode,
    ProvenanceGraphResult,
    KNOWN_SOURCE_TYPES,
)
from kohaku.time_filter import (
    TimeFilter,
    TimelineBucket,
    apply_time_filter,
    bucket_timeline,
    filter_recent,
)
from kohaku.memory_health import (
    MemoryHealthAnalyzer,
    MemoryHealthReport,
    DuplicatePair,
    StaleMemory,
)

try:
    from kohaku.server import create_app, serve
    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False

__all__ = [
    "HyperVector",
    "EpisodicMemory",
    "AsyncEpisodicMemory",
    "RetrievalResult",
    "query",
    "query_threshold",
    "_BACKEND",
    "ContextConfig",
    "ContextMemoryManager",
    "attention_weighted_encode",
    "encode_text",
    "MemoryMiddleware",
    "save",
    "load",
    "save_json",
    "load_json",
    "save_binary",
    "load_binary",
    "Cluster",
    "consolidate",
    "consolidate_to_memory",
    "DecayConfig",
    "decay_weight",
    "query_with_decay",
    "ItemMemory",
    "Prototype",
    "HopfieldAssociator",
    "HopfieldRecall",
    "MemorySystem",
    "CombinedRecall",
    "StreamingConsolidator",
    "find_duplicates",
    "deduplicate",
    "compact",
    "TenantMemoryStore",
    "HDCRetriever",
    "RetrievedChunk",
    "GraphExportConfig",
    "MemoryGraphExporter",
    "MemoryGraph",
    "MemoryNode",
    "MemoryEdge",
    "create_app",
    "serve",
    "EnrichedMemoryStore",
    "EnrichedRetrievalResult",
    "MemoryMetadata",
    "SOURCE_TRUST_WEIGHTS",
    "SleepConsolidator",
    "SleepReport",
    "ProvenanceGraph",
    "ProvenanceNode",
    "ProvenanceGraphResult",
    "KNOWN_SOURCE_TYPES",
    "TimeFilter",
    "TimelineBucket",
    "apply_time_filter",
    "bucket_timeline",
    "filter_recent",
    "MemoryHealthAnalyzer",
    "MemoryHealthReport",
    "DuplicatePair",
    "StaleMemory",
]
