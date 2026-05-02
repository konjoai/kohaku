"""Kohaku — HDC episodic memory. Uses Rust extension when available, pure-Python otherwise."""
from __future__ import annotations

__version__ = "0.4.0"

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
]
