"""Time-range memory search — interval-overlap filtering + timeline bucketing.

Used to answer "what did the agent know during this time window?" — a memory
counts as *known* during ``[a, b]`` iff its validity interval
``[valid_from, valid_until]`` overlaps the query window.

For memories with no explicit ``valid_until`` the interval is treated as
half-open into the future (``+∞``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


# Bucket sizes for the timeline endpoint, in seconds.
BUCKETS = {
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,  # 30-day approximation; consistent with the dashboard
}


# ──────────────────────────── parsing ──────────────────────────────────────


def _parse_iso(value: Union[str, datetime, None]) -> Optional[datetime]:
    """Accept ISO 8601 (``Z`` suffix permitted), :class:`datetime`, or ``None``.

    Returns a UTC-aware ``datetime`` or ``None``. Raises :class:`ValueError` on
    malformed input.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        raise ValueError(f"expected str or datetime, got {type(value).__name__}")
    raw = value.strip()
    if not raw:
        return None
    # `datetime.fromisoformat` accepts `2026-01-01T00:00:00+00:00` but not the
    # short `Z` suffix until 3.11; normalise it.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"could not parse timestamp {value!r}: {exc}") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _to_dt(value: Any) -> Optional[datetime]:
    """Coerce a memory's ``valid_from`` / ``valid_until`` field (string or
    datetime) to a UTC-aware datetime, or ``None`` if absent."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return _parse_iso(value)
        except ValueError:
            logger.warning("dropping unparseable timestamp from memory")
            return None
    return None


# ──────────────────────────── filter ───────────────────────────────────────


@dataclass(frozen=True)
class TimeFilter:
    """Window ``[valid_after, valid_before]`` against which a memory's
    ``[valid_from, valid_until]`` interval is tested for overlap.

    Either endpoint may be ``None`` to leave that side unbounded. An empty
    filter (``TimeFilter()``) matches every memory.
    """

    valid_after: Optional[datetime] = None
    valid_before: Optional[datetime] = None

    def __post_init__(self) -> None:
        a, b = self.valid_after, self.valid_before
        if a is not None and b is not None and a > b:
            raise ValueError(
                f"valid_after ({a.isoformat()}) must be <= "
                f"valid_before ({b.isoformat()})"
            )

    @classmethod
    def from_iso(
        cls,
        valid_after: Union[str, datetime, None] = None,
        valid_before: Union[str, datetime, None] = None,
    ) -> "TimeFilter":
        return cls(
            valid_after=_parse_iso(valid_after),
            valid_before=_parse_iso(valid_before),
        )

    @property
    def is_empty(self) -> bool:
        return self.valid_after is None and self.valid_before is None

    def matches(self, valid_from: Any, valid_until: Any = None) -> bool:
        """Return True iff the memory's validity interval overlaps the window.

        Semantics (interval overlap):

            memory  = [vf, vu]   (vu defaults to +∞ if absent)
            window  = [a, b]     (either endpoint may be None)

            match  iff  (b is None OR vf <= b)
                  AND  (a is None OR vu is None OR vu >= a)
        """
        if self.is_empty:
            return True
        vf = _to_dt(valid_from)
        if vf is None:
            # A memory without `valid_from` cannot be placed on the axis,
            # so it cannot satisfy a bounded window.
            return False
        vu = _to_dt(valid_until)
        a, b = self.valid_after, self.valid_before
        if b is not None and vf > b:
            return False
        if a is not None and vu is not None and vu < a:
            return False
        return True


def apply_time_filter(
    memories: Iterable[Any],
    tf: TimeFilter,
) -> List[Any]:
    """Filter an iterable of memory dicts (or objects) by the given window.

    Each item is expected to expose ``valid_from`` and ``valid_until`` — as
    attributes (dataclass / dataclass-like) or dict keys. Items without a
    ``valid_from`` are silently dropped when the window is bounded.
    """
    if tf.is_empty:
        return list(memories)
    out: List[Any] = []
    for m in memories:
        vf = _attr(m, "valid_from")
        vu = _attr(m, "valid_until")
        if tf.matches(vf, vu):
            out.append(m)
    return out


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# ──────────────────────────── timeline ─────────────────────────────────────


@dataclass(frozen=True)
class TimelineBucket:
    """One row in a timeline aggregation."""

    bucket_start: datetime
    bucket_end: datetime
    count: int
    memories: List[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_start": self.bucket_start.isoformat(),
            "bucket_end": self.bucket_end.isoformat(),
            "date": self.bucket_start.date().isoformat(),
            "count": self.count,
            "memories": self.memories,
        }


def _floor_to_bucket(dt: datetime, bucket_seconds: int) -> datetime:
    """Floor ``dt`` to the start of its bucket. UTC-aware in, UTC-aware out."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    epoch = int(dt.timestamp())
    floored = epoch - (epoch % bucket_seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def bucket_timeline(
    memories: Sequence[Any],
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    bucket: str = "day",
    preview_per_bucket: int = 5,
    text_field: str = "label",
    id_field: str = "entry_id",
    truncate: int = 80,
) -> List[TimelineBucket]:
    """Group memories into time buckets keyed by ``valid_from``.

    The returned list is sorted by bucket start ascending and includes empty
    buckets between ``start`` and ``end`` when both are provided. Empty buckets
    are still emitted so a sparse timeline renders correctly.
    """
    if bucket not in BUCKETS:
        raise ValueError(f"bucket must be one of {sorted(BUCKETS)}, got {bucket!r}")
    if preview_per_bucket < 0:
        raise ValueError("preview_per_bucket must be >= 0")
    width = BUCKETS[bucket]

    s = _parse_iso(start)
    e = _parse_iso(end)
    if s is not None and e is not None and s > e:
        raise ValueError("start must be <= end")

    # Stage 1: assign each memory to its bucket start.
    binned: dict[datetime, List[dict[str, Any]]] = {}
    for m in memories:
        vf = _to_dt(_attr(m, "valid_from"))
        if vf is None:
            continue
        if s is not None and vf < s:
            continue
        if e is not None and vf > e:
            continue
        b_start = _floor_to_bucket(vf, width)
        binned.setdefault(b_start, []).append(
            _preview_memory(m, text_field, id_field, truncate)
        )

    # Stage 2: walk start → end in fixed-width strides, emitting one bucket
    # per stride (empty or not). If start/end are unset, emit only the
    # buckets that actually contain memories.
    out: List[TimelineBucket] = []
    if s is None or e is None:
        for b_start in sorted(binned):
            b_end = b_start + timedelta(seconds=width)
            mems = binned[b_start][:preview_per_bucket]
            out.append(
                TimelineBucket(
                    bucket_start=b_start,
                    bucket_end=b_end,
                    count=len(binned[b_start]),
                    memories=mems,
                )
            )
        return out
    cur = _floor_to_bucket(s, width)
    while cur <= e:
        b_end = cur + timedelta(seconds=width)
        mems = binned.get(cur, [])
        out.append(
            TimelineBucket(
                bucket_start=cur,
                bucket_end=b_end,
                count=len(mems),
                memories=mems[:preview_per_bucket],
            )
        )
        cur = b_end
    return out


def _preview_memory(
    m: Any, text_field: str, id_field: str, truncate: int
) -> dict[str, Any]:
    text = str(_attr(m, text_field) or "")
    if truncate > 0 and len(text) > truncate:
        text = text[: truncate - 1] + "…"
    return {
        "entry_id": _attr(m, id_field),
        "text": text,
        "valid_from": _safe_iso(_attr(m, "valid_from")),
        "valid_until": _safe_iso(_attr(m, "valid_until")),
    }


def _safe_iso(value: Any) -> Optional[str]:
    dt = _to_dt(value)
    return dt.isoformat() if dt is not None else None


# ──────────────────────────── recent ───────────────────────────────────────


def filter_recent(
    memories: Iterable[Any],
    *,
    since_hours: float,
    now: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[Any]:
    """Return memories whose ``valid_from`` is within the last ``since_hours``.

    Sorted most-recent-first. ``limit`` caps the result; ``None`` returns all.
    """
    if since_hours <= 0:
        raise ValueError("since_hours must be > 0")
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=since_hours)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    rows: List[tuple[datetime, Any]] = []
    for m in memories:
        vf = _to_dt(_attr(m, "valid_from"))
        if vf is None or vf < cutoff:
            continue
        rows.append((vf, m))
    rows.sort(key=lambda r: r[0], reverse=True)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return [m for _vf, m in rows]
