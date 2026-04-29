"""Kohaku — HDC episodic memory. Uses Rust extension when available, pure-Python otherwise."""
from __future__ import annotations

__version__ = "0.3.0"

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
]
