"""Tests for kohaku.time_filter — interval-overlap + timeline bucketing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kohaku.time_filter import (
    BUCKETS,
    TimeFilter,
    _parse_iso,
    apply_time_filter,
    bucket_timeline,
    filter_recent,
)


JAN_1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
FEB_1 = datetime(2026, 2, 1, tzinfo=timezone.utc)
MAR_1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
APR_1 = datetime(2026, 4, 1, tzinfo=timezone.utc)


def _mem(vf: datetime, vu=None, **kw) -> dict:
    return {
        "entry_id": kw.get("entry_id", 1),
        "label": kw.get("label", ""),
        "valid_from": vf,
        "valid_until": vu,
    }


def test_parse_iso_accepts_z_suffix() -> None:
    dt = _parse_iso("2026-05-12T00:00:00Z")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 5 and dt.tzinfo is not None


def test_parse_iso_naive_promoted_to_utc() -> None:
    dt = _parse_iso("2026-05-12T00:00:00")
    assert dt is not None
    assert dt.tzinfo == timezone.utc


def test_parse_iso_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _parse_iso("not-a-date")


def test_empty_filter_passes_through() -> None:
    rows = [_mem(JAN_1), _mem(MAR_1)]
    assert apply_time_filter(rows, TimeFilter()) == rows


def test_filter_matches_interval_overlap() -> None:
    tf = TimeFilter(valid_after=FEB_1, valid_before=MAR_1)
    # Memory created Jan 1, valid forever — overlaps Feb-Mar window
    assert tf.matches(JAN_1, None) is True
    # Memory created Jan 1, expired before window
    assert tf.matches(JAN_1, FEB_1 - timedelta(days=1)) is False
    # Memory created after window
    assert tf.matches(APR_1, None) is False


def test_filter_rejects_inverted_window() -> None:
    with pytest.raises(ValueError, match="valid_after"):
        TimeFilter(valid_after=MAR_1, valid_before=JAN_1)


def test_apply_time_filter_drops_unparseable() -> None:
    rows = [
        _mem(JAN_1, label="ok"),
        {"entry_id": 2, "label": "bad", "valid_from": "garbage", "valid_until": None},
        {"entry_id": 3, "label": "missing", "valid_until": None},
    ]
    tf = TimeFilter(valid_after=JAN_1 - timedelta(days=1), valid_before=FEB_1)
    out = apply_time_filter(rows, tf)
    assert [r["label"] for r in out] == ["ok"]


def test_apply_time_filter_accepts_iso_strings() -> None:
    tf = TimeFilter.from_iso("2026-01-15T00:00:00Z", "2026-02-15T00:00:00Z")
    rows = [_mem(JAN_1, FEB_1), _mem(MAR_1, None)]
    out = apply_time_filter(rows, tf)
    # First overlaps (Jan 1 → Feb 1 hits Jan 15 → Feb 15 window)
    assert len(out) == 1
    assert out[0]["valid_from"] == JAN_1


def test_bucket_timeline_groups_by_day() -> None:
    same_day_1 = JAN_1
    same_day_2 = JAN_1 + timedelta(hours=5)
    next_day = JAN_1 + timedelta(days=1, hours=2)
    rows = [
        _mem(same_day_1, label="a", entry_id=1),
        _mem(same_day_2, label="b", entry_id=2),
        _mem(next_day, label="c", entry_id=3),
    ]
    buckets = bucket_timeline(rows, bucket="day")
    assert len(buckets) == 2
    counts = sorted([b.count for b in buckets])
    assert counts == [1, 2]


def test_bucket_timeline_emits_empty_buckets_when_bounded() -> None:
    rows = [_mem(JAN_1, label="x")]
    buckets = bucket_timeline(
        rows,
        start="2026-01-01T00:00:00Z",
        end="2026-01-04T23:59:59Z",
        bucket="day",
    )
    # Should emit Jan 1, 2, 3, 4 buckets
    assert len(buckets) == 4
    assert buckets[0].count == 1
    assert all(b.count == 0 for b in buckets[1:])


def test_bucket_timeline_rejects_invalid_bucket() -> None:
    with pytest.raises(ValueError):
        bucket_timeline([], bucket="fortnight")


def test_bucket_timeline_preview_truncates_text() -> None:
    long = "x" * 200
    rows = [_mem(JAN_1, label=long)]
    buckets = bucket_timeline(rows, bucket="day", truncate=20)
    preview = buckets[0].memories[0]["text"]
    assert len(preview) <= 20
    assert preview.endswith("…")


def test_filter_recent_only_recent() -> None:
    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    one_hr_ago = now - timedelta(hours=1)
    two_days_ago = now - timedelta(days=2)
    rows = [
        _mem(one_hr_ago, label="fresh", entry_id=1),
        _mem(two_days_ago, label="old", entry_id=2),
    ]
    out = filter_recent(rows, since_hours=24, now=now)
    assert [r["label"] for r in out] == ["fresh"]


def test_filter_recent_respects_limit() -> None:
    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    rows = [
        _mem(now - timedelta(hours=i), label=f"m{i}", entry_id=i) for i in range(1, 11)
    ]
    out = filter_recent(rows, since_hours=24, now=now, limit=3)
    assert len(out) == 3
    # sorted most-recent-first
    assert [r["label"] for r in out] == ["m1", "m2", "m3"]


def test_filter_recent_rejects_zero_hours() -> None:
    with pytest.raises(ValueError):
        filter_recent([], since_hours=0)


def test_all_bucket_names_resolve() -> None:
    for name in BUCKETS:
        bucket_timeline([], bucket=name)
