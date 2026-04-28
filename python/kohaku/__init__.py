"""Kohaku — HDC episodic memory. Uses Rust extension when available, pure-Python otherwise."""
from __future__ import annotations

__version__ = "0.2.0"

try:
    from kohaku._kohaku_rs import HyperVector, EpisodicMemory  # compiled Rust ext
    _BACKEND = "rust"
except ImportError:
    from kohaku._pure import HyperVector, EpisodicMemory  # pure Python fallback
    _BACKEND = "python"

from kohaku._async import AsyncEpisodicMemory
from kohaku._query import RetrievalResult, query, query_threshold

__all__ = [
    "HyperVector",
    "EpisodicMemory",
    "AsyncEpisodicMemory",
    "RetrievalResult",
    "query",
    "query_threshold",
    "_BACKEND",
]
