"""Tests for kohaku.context — ContextMemoryManager and ContextConfig."""
from __future__ import annotations

import pytest
from kohaku.context import ContextConfig, ContextMemoryManager, _encode_text_to_hv


# ---------------------------------------------------------------------------
# 1. store + retrieve a known entry
# ---------------------------------------------------------------------------

def test_store_and_retrieve_known_entry():
    """Storing an entry and querying with the same key returns it."""
    mgr = ContextMemoryManager()
    mgr.store("dog food", "premium kibble", label="pet")
    results = mgr.retrieve("dog food")
    assert len(results) >= 1
    labels = [r[0] for r in results]
    assert "pet" in labels


# ---------------------------------------------------------------------------
# 2. build_context_block starts with "Relevant memories:"
# ---------------------------------------------------------------------------

def test_build_context_block_prefix():
    """build_context_block returns a string starting with 'Relevant memories:'."""
    mgr = ContextMemoryManager()
    mgr.store("coffee morning", "espresso is great", label="beverage")
    block = mgr.build_context_block("coffee")
    assert isinstance(block, str)
    assert block.startswith("Relevant memories:")


# ---------------------------------------------------------------------------
# 3. capacity() == max_tokens // tokens_per_entry
# ---------------------------------------------------------------------------

def test_capacity_equals_config_ratio():
    """capacity() must equal max_tokens // tokens_per_entry from config."""
    cfg = ContextConfig(max_tokens=2000, tokens_per_entry=100)
    mgr = ContextMemoryManager(cfg)
    assert mgr.capacity() == 20


# ---------------------------------------------------------------------------
# 4. utilization() is 0.0 on empty, >0 after store
# ---------------------------------------------------------------------------

def test_utilization_empty_and_after_store():
    """utilization() is 0.0 when empty and positive after storing one entry."""
    mgr = ContextMemoryManager(ContextConfig(max_tokens=1000, tokens_per_entry=50))
    assert mgr.utilization() == 0.0
    mgr.store("hello world", "greeting", label="greet")
    assert mgr.utilization() > 0.0


# ---------------------------------------------------------------------------
# 5. semantic similarity — "dog food" retrieves more similar to "dog" than "spaceship"
# ---------------------------------------------------------------------------

def test_similar_queries_retrieve_related_entries():
    """Querying 'dog' should match 'dog food' more than 'spaceship launch'.

    HDC vectors for orthogonal concepts have near-zero and occasionally negative
    cosine similarity, so we test ranking rather than count.  We use a negative
    threshold (-1.0) to force both entries to be returned regardless of polarity.
    """
    mgr = ContextMemoryManager(ContextConfig(top_k=2, similarity_threshold=-1.0))
    mgr.store("dog food", "premium pet nutrition", label="pet")
    mgr.store("spaceship launch", "orbital trajectory", label="space")

    results = mgr.retrieve("dog", top_k=2)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}: {results}"
    # The pet entry must rank first — higher similarity to "dog"
    top_label = results[0][0]
    assert top_label == "pet", f"Expected 'pet' to rank first, got {top_label!r}"
    # Sanity: "dog food" similarity must exceed "spaceship" similarity
    pet_sim = results[0][2]
    space_sim = results[1][2]
    assert pet_sim > space_sim, (
        f"'pet' similarity ({pet_sim:.4f}) should exceed 'space' ({space_sim:.4f})"
    )


# ---------------------------------------------------------------------------
# 6. FIFO eviction — oldest entry gone after capacity+1 stores
# ---------------------------------------------------------------------------

def test_fifo_eviction():
    """After filling to capacity+1, the oldest entry should be evicted."""
    cfg = ContextConfig(max_tokens=500, tokens_per_entry=100)  # capacity = 5
    mgr = ContextMemoryManager(cfg)
    assert mgr.capacity() == 5

    # Store 5 entries
    for i in range(5):
        mgr.store(f"key_{i}", f"value_{i}", label=f"entry_{i}")
    assert len(mgr._memory) == 5

    # Store a 6th — should evict entry_0
    mgr.store("key_5", "value_5", label="entry_5")
    assert len(mgr._memory) == 5

    # entry_0 must be gone from the raw label list
    assert "entry_0" not in mgr._labels
    assert "entry_5" in mgr._labels


# ---------------------------------------------------------------------------
# 7. encode_text determinism — same text → same HyperVector
# ---------------------------------------------------------------------------

def test_encode_text_deterministic():
    """_encode_text_to_hv must return byte-identical results for the same input."""
    import numpy as np
    hv1 = _encode_text_to_hv("hello world")
    hv2 = _encode_text_to_hv("hello world")
    assert np.array_equal(hv1.data, hv2.data), "Same text must always produce the same hypervector"


# ---------------------------------------------------------------------------
# 8. ContextConfig defaults are sane
# ---------------------------------------------------------------------------

def test_context_config_defaults():
    """Default ContextConfig values must match the documented spec."""
    cfg = ContextConfig()
    assert cfg.max_tokens == 4096
    assert cfg.tokens_per_entry == 50
    assert cfg.top_k == 5
    assert 0.0 <= cfg.similarity_threshold <= 1.0
