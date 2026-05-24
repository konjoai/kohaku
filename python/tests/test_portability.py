"""Tests for kohaku.portability — export / import round-trips."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone

import pytest

from kohaku import EnrichedMemoryStore, encode_text
from kohaku.portability import (
    EXPORT_VERSION,
    ExportBundle,
    ImportReport,
    export_csv,
    export_json,
    export_markdown,
    export_memories,
    import_memories,
)


def _populated() -> EnrichedMemoryStore:
    s = EnrichedMemoryStore(capacity=20)
    s.store(encode_text("coffee at 8 am"),
            encode_text("coffee at 8 am"),
            label="coffee at 8 am",
            source="user_input", importance=0.8,
            tags=["routine", "beverage"])
    s.store(encode_text("forgot the umbrella"),
            encode_text("forgot the umbrella"),
            label="forgot the umbrella",
            source="agent_inference", importance=0.3,
            tags=["regret"])
    return s


def test_export_unknown_format_rejected() -> None:
    s = _populated()
    with pytest.raises(ValueError):
        export_memories(s, fmt="rtf")


def test_export_json_emits_envelope() -> None:
    s = _populated()
    bundle = export_json(s)
    assert isinstance(bundle, ExportBundle)
    payload = json.loads(bundle.payload)
    assert payload["version"] == EXPORT_VERSION
    assert payload["memory_count"] == 2
    assert payload["dims"] == s.dims
    assert isinstance(payload["memories"], list)
    assert {"label", "source", "tags"} <= payload["memories"][0].keys()


def test_export_markdown_has_header_and_sections() -> None:
    s = _populated()
    bundle = export_markdown(s)
    lines = bundle.payload.splitlines()
    assert lines[0] == "# Kohaku memory export"
    section_headers = [line for line in lines if line.startswith("## ")]
    assert len(section_headers) == 2


def test_export_csv_is_round_trip_parseable() -> None:
    s = _populated()
    bundle = export_csv(s)
    rows = list(csv.DictReader(io.StringIO(bundle.payload)))
    assert len(rows) == 2
    assert "label" in rows[0]
    # tags joined with `|`
    assert "|" in rows[0]["tags"] or "|" in rows[1]["tags"]


def test_import_round_trip_into_fresh_store() -> None:
    src = _populated()
    bundle = export_json(src)
    dst = EnrichedMemoryStore(capacity=20)
    report = import_memories(dst, bundle.payload, dedup_threshold=0.99)
    assert isinstance(report, ImportReport)
    assert report.imported == 2
    assert report.skipped_duplicates == 0
    assert report.skipped_invalid == 0
    labels = {e.label for e in dst.episodic.entries()}
    assert labels == {"coffee at 8 am", "forgot the umbrella"}


def test_import_deduplicates_against_existing_store() -> None:
    src = _populated()
    bundle = export_json(src)
    dst = _populated()  # already has both labels
    report = import_memories(dst, bundle.payload, dedup_threshold=0.99)
    assert report.imported == 0
    assert report.skipped_duplicates == 2
    assert set(report.duplicate_of.keys()) == {0, 1}


def test_import_preserves_tags_and_source() -> None:
    src = _populated()
    bundle = export_json(src)
    dst = EnrichedMemoryStore(capacity=20)
    import_memories(dst, bundle.payload, dedup_threshold=0.99)
    items = dst.list_memories()
    by_label = {it["label"]: it for it in items}
    assert by_label["coffee at 8 am"]["source"] == "user_input"
    assert "routine" in by_label["coffee at 8 am"]["tags"]
    assert by_label["forgot the umbrella"]["source"] == "agent_inference"


def test_import_rejects_garbage_json() -> None:
    dst = EnrichedMemoryStore(capacity=5)
    with pytest.raises(ValueError):
        import_memories(dst, "not valid json", dedup_threshold=0.99)


def test_import_accepts_bare_list() -> None:
    dst = EnrichedMemoryStore(capacity=5)
    payload = json.dumps([{"label": "hello world", "source": "user_input"}])
    report = import_memories(dst, payload, dedup_threshold=0.99)
    assert report.imported == 1


def test_import_skips_empty_label() -> None:
    dst = EnrichedMemoryStore(capacity=5)
    payload = json.dumps({"memories": [
        {"label": "", "source": "user_input"},
        {"label": "kept", "source": "user_input"},
    ]})
    report = import_memories(dst, payload, dedup_threshold=0.99)
    assert report.imported == 1
    assert report.skipped_invalid == 1


def test_import_parses_iso_validity_dates() -> None:
    dst = EnrichedMemoryStore(capacity=5)
    now = datetime.now(timezone.utc)
    valid = (now - timedelta(days=2)).isoformat()
    until = (now + timedelta(days=2)).isoformat()
    payload = json.dumps([{
        "label": "scheduled chat",
        "source": "user_input",
        "valid_from": valid,
        "valid_until": until,
    }])
    report = import_memories(dst, payload, dedup_threshold=0.99)
    assert report.imported == 1
    meta = dst.get_metadata(report.new_ids[0])
    assert meta is not None and meta.valid_until is not None


def test_import_skips_unparseable_dates() -> None:
    dst = EnrichedMemoryStore(capacity=5)
    payload = json.dumps([{"label": "bad date", "valid_from": "garbage"}])
    report = import_memories(dst, payload, dedup_threshold=0.99)
    assert report.skipped_invalid == 1


def test_dedup_threshold_validation() -> None:
    dst = EnrichedMemoryStore(capacity=5)
    with pytest.raises(ValueError):
        import_memories(dst, "[]", dedup_threshold=0.0)


def test_export_bundle_counts_match_payload() -> None:
    s = _populated()
    bundle = export_json(s)
    assert bundle.memory_count == 2
    # routine + beverage + regret = 3 tag uses
    assert bundle.tag_count == 3
