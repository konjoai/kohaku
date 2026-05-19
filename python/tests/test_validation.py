"""Tests for kohaku.validation — write-time validation and poisoning defense."""
from __future__ import annotations

import pytest

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku.validation import RateLimit, WriteValidator


def _hv(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


# ── RateLimit construction ────────────────────────────────────────────────────

def test_rate_limit_max_stores_zero_raises():
    with pytest.raises(ValueError, match="max_stores"):
        RateLimit(max_stores=0, window_seconds=60.0)


def test_rate_limit_window_zero_raises():
    with pytest.raises(ValueError, match="window_seconds"):
        RateLimit(max_stores=10, window_seconds=0.0)


def test_rate_limit_negative_raises():
    with pytest.raises(ValueError):
        RateLimit(max_stores=-1, window_seconds=60.0)


# ── WriteValidator construction ───────────────────────────────────────────────

def test_duplicate_threshold_zero_raises():
    mem = EpisodicMemory()
    with pytest.raises(ValueError, match="duplicate_threshold"):
        WriteValidator(mem, duplicate_threshold=0.0)


def test_duplicate_threshold_above_one_raises():
    mem = EpisodicMemory()
    with pytest.raises(ValueError, match="duplicate_threshold"):
        WriteValidator(mem, duplicate_threshold=1.01)


def test_duplicate_threshold_exactly_one_is_valid():
    mem = EpisodicMemory()
    WriteValidator(mem, duplicate_threshold=1.0)  # should not raise


# ── empty memory always accepts ───────────────────────────────────────────────

def test_empty_memory_accepts_any_key():
    mem = EpisodicMemory()
    validator = WriteValidator(mem)
    result = validator.validate(_hv(1))
    assert result.accepted is True
    assert result.reason == "accepted"
    assert result.nearest_similarity == 0.0
    assert result.nearest_label == ""


# ── novelty check ─────────────────────────────────────────────────────────────

def test_identical_key_rejected_as_near_duplicate():
    mem = EpisodicMemory()
    hv = _hv(1)
    mem.store(hv, hv, "original")
    validator = WriteValidator(mem, duplicate_threshold=0.99)
    result = validator.validate(hv)
    assert result.accepted is False
    assert result.reason == "near_duplicate"
    assert result.nearest_similarity == pytest.approx(1.0, abs=1e-4)
    assert result.nearest_label == "original"


def test_orthogonal_key_accepted():
    mem = EpisodicMemory()
    hv_a = _hv(1)
    hv_b = _hv(999)
    mem.store(hv_a, hv_a, "stored")
    validator = WriteValidator(mem, duplicate_threshold=0.99)
    result = validator.validate(hv_b)
    assert result.accepted is True


def test_threshold_one_only_rejects_identical():
    """duplicate_threshold=1.0 means only cosine==1.0 is a duplicate."""
    mem = EpisodicMemory()
    hv = _hv(1)
    mem.store(hv, hv, "orig")
    validator = WriteValidator(mem, duplicate_threshold=1.0)

    # Identical → rejected
    result = validator.validate(hv)
    assert result.accepted is False

    # Slightly different (noisy copy) → accepted
    noisy_data = hv.data.copy()
    noisy_data[:50] *= -1  # flip 0.5% of bits
    noisy = HyperVector(noisy_data)
    result2 = validator.validate(noisy)
    assert result2.accepted is True


def test_nearest_similarity_populated_on_accept():
    """Even on accept, nearest_similarity reflects the closest stored entry (nonzero)."""
    mem = EpisodicMemory()
    hv_a = _hv(1)
    mem.store(hv_a, hv_a, "close")
    validator = WriteValidator(mem, duplicate_threshold=0.99)
    hv_b = _hv(2)  # different but not identical
    result = validator.validate(hv_b)
    assert result.accepted is True
    # Cosine may be negative for random HVs; just confirm it's not the default 0.0
    assert result.nearest_similarity != 0.0
    assert result.nearest_label == "close"


# ── rate limit ────────────────────────────────────────────────────────────────

def test_rate_limit_second_call_rejected():
    mem = EpisodicMemory()
    validator = WriteValidator(
        mem,
        rate_limits={"bot": RateLimit(max_stores=1, window_seconds=60.0)},
    )
    hv_a = _hv(1)
    hv_b = _hv(999)
    r1 = validator.validate(hv_a, source="bot")
    assert r1.accepted is True
    validator.record(source="bot")

    r2 = validator.validate(hv_b, source="bot")
    assert r2.accepted is False
    assert r2.reason == "rate_limit_exceeded"


def test_rate_limit_different_source_unaffected():
    mem = EpisodicMemory()
    validator = WriteValidator(
        mem,
        rate_limits={"bot": RateLimit(max_stores=1, window_seconds=60.0)},
    )
    hv = _hv(1)
    validator.validate(hv, source="bot")
    validator.record(source="bot")

    # A different source (not rate-limited) should pass freely.
    result = validator.validate(_hv(999), source="human")
    assert result.accepted is True


def test_validate_does_not_update_rate_limit():
    """validate() alone must NOT consume a rate-limit slot."""
    mem = EpisodicMemory()
    validator = WriteValidator(
        mem,
        rate_limits={"bot": RateLimit(max_stores=1, window_seconds=60.0)},
    )
    hv = _hv(1)
    # Two validate() calls without record() — both should pass.
    r1 = validator.validate(hv, source="bot")
    r2 = validator.validate(_hv(2), source="bot")
    assert r1.accepted is True
    assert r2.accepted is True


# ── validate_and_store ────────────────────────────────────────────────────────

def test_validate_and_store_accepted_stores_entry():
    mem = EpisodicMemory()
    validator = WriteValidator(mem)
    hv = _hv(1)
    result, eid = validator.validate_and_store(hv, hv, "stored")
    assert result.accepted is True
    assert eid is not None
    assert len(mem) == 1


def test_validate_and_store_rejected_does_not_store():
    mem = EpisodicMemory()
    hv = _hv(1)
    mem.store(hv, hv, "original")
    validator = WriteValidator(mem, duplicate_threshold=0.99)
    result, eid = validator.validate_and_store(hv, hv, "dup")
    assert result.accepted is False
    assert eid is None
    assert len(mem) == 1  # unchanged


def test_validate_and_store_records_rate_limit():
    """validate_and_store should call record() so the slot is consumed."""
    mem = EpisodicMemory()
    validator = WriteValidator(
        mem,
        rate_limits={"bot": RateLimit(max_stores=1, window_seconds=60.0)},
    )
    hv_a = _hv(1)
    hv_b = _hv(999)
    r1, _ = validator.validate_and_store(hv_a, hv_a, "first", source="bot")
    assert r1.accepted is True
    r2, _ = validator.validate_and_store(hv_b, hv_b, "second", source="bot")
    assert r2.accepted is False
    assert r2.reason == "rate_limit_exceeded"
