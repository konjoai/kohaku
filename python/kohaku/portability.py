"""Portable memory export / import — JSON, Markdown, CSV.

The existing :mod:`kohaku.persistence` ships raw `.hkb` / `.json` snapshots
of the HDC :class:`EpisodicMemory`. This module is a step higher: it
serialises the full enriched view (metadata + tags + optional provenance
edges) so memories can be backed up, audited, transferred between agents,
or rendered for human review.

Three formats, one shape:

* **JSON** — full fidelity. Round-trips cleanly via :func:`import_json`.
  Hypervectors are NOT included — they're re-derived from the label on
  import via :func:`kohaku.encode_text`, so a JSON export is a *seed
  file* rather than a verbatim mirror. This keeps payloads small and
  human-diffable; if bit-exact vectors are required, use ``save_binary``.
* **Markdown** — human-readable. One ``##`` section per memory with
  metadata as a key-value block. Useful for review or for piping into
  a wiki.
* **CSV** — spreadsheet-friendly. Tags flatten to a ``|``-joined string.

Imports are deduplicated against the live store: any incoming memory whose
encoded label has cosine similarity ≥ ``dedup_threshold`` to an existing
entry is skipped (and counted). The threshold defaults to 0.99 so only
near-identical phrasings are merged.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from kohaku.attention import encode_text
from kohaku.enriched import EnrichedMemoryStore

logger = logging.getLogger(__name__)

EXPORT_VERSION: str = "kohaku-export-1"
DEFAULT_DEDUP_THRESHOLD: float = 0.99


@dataclass(frozen=True)
class ExportBundle:
    """Result of an export call — payload + summary counts."""

    format: str
    payload: str
    memory_count: int
    tag_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "payload": self.payload,
            "memory_count": int(self.memory_count),
            "tag_count": int(self.tag_count),
        }


@dataclass(frozen=True)
class ImportReport:
    """Outcome of an import — what landed, what got deduplicated."""

    imported: int
    skipped_duplicates: int
    skipped_invalid: int
    new_ids: Tuple[int, ...]
    duplicate_of: Dict[int, int]  # incoming-index → existing entry_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "imported": int(self.imported),
            "skipped_duplicates": int(self.skipped_duplicates),
            "skipped_invalid": int(self.skipped_invalid),
            "new_ids": [int(i) for i in self.new_ids],
            "duplicate_of": {int(k): int(v) for k, v in self.duplicate_of.items()},
        }


# ──────────────────────────── export ───────────────────────────────────────


def _memory_records(store: EnrichedMemoryStore) -> List[Dict[str, Any]]:
    """Build the canonical record list — same shape across all formats."""
    records: List[Dict[str, Any]] = []
    for e in store.episodic.entries():
        meta = store.get_metadata(e.id)
        if meta is None:
            continue
        records.append(
            {
                "entry_id": e.id,
                "label": e.label,
                "source": meta.source,
                "importance": float(meta.importance),
                "reinforcement_count": int(meta.reinforcement_count),
                "valid_from": meta.valid_from.isoformat(),
                "valid_until": meta.valid_until.isoformat()
                if meta.valid_until
                else None,
                "created_at": meta.created_at.isoformat(),
                "tags": sorted(meta.tags),
            }
        )
    return records


def export_json(store: EnrichedMemoryStore, *, indent: int = 2) -> ExportBundle:
    records = _memory_records(store)
    payload = {
        "version": EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "memory_count": len(records),
        "dims": store.dims,
        "memories": records,
    }
    if store.provenance is not None:
        try:
            edges: List[List[str]] = []
            for r in records:
                parents = store.provenance._direct_parents(str(r["entry_id"]))  # type: ignore[attr-defined]
                for p in parents:
                    edges.append([p, str(r["entry_id"])])
            payload["provenance_edges"] = edges
        except (AttributeError, OSError, RuntimeError) as exc:
            logger.warning(
                "provenance export skipped (%s)",
                exc.__class__.__name__,
            )
    text = json.dumps(payload, indent=indent, sort_keys=True)
    return ExportBundle(
        format="json",
        payload=text,
        memory_count=len(records),
        tag_count=sum(len(r["tags"]) for r in records),
    )


def export_markdown(store: EnrichedMemoryStore) -> ExportBundle:
    records = _memory_records(store)
    lines: List[str] = [
        "# Kohaku memory export",
        "",
        f"_Exported {datetime.now(timezone.utc).isoformat()} · "
        f"{len(records)} memor{'y' if len(records) == 1 else 'ies'}_",
        "",
    ]
    for r in records:
        lines.append(f"## #{r['entry_id']} — {r['label']}")
        lines.append("")
        lines.append(f"- **source:** `{r['source']}`")
        lines.append(f"- **importance:** {r['importance']:.2f}")
        lines.append(f"- **reinforcement:** {r['reinforcement_count']}")
        lines.append(f"- **valid_from:** {r['valid_from']}")
        if r["valid_until"]:
            lines.append(f"- **valid_until:** {r['valid_until']}")
        if r["tags"]:
            tag_chips = ", ".join(f"`{t}`" for t in r["tags"])
            lines.append(f"- **tags:** {tag_chips}")
        lines.append("")
    payload = "\n".join(lines).rstrip() + "\n"
    return ExportBundle(
        format="markdown",
        payload=payload,
        memory_count=len(records),
        tag_count=sum(len(r["tags"]) for r in records),
    )


def export_csv(store: EnrichedMemoryStore) -> ExportBundle:
    records = _memory_records(store)
    buf = io.StringIO()
    cols = [
        "entry_id",
        "label",
        "source",
        "importance",
        "reinforcement_count",
        "valid_from",
        "valid_until",
        "created_at",
        "tags",
    ]
    writer = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    writer.writeheader()
    for r in records:
        row = dict(r)
        row["tags"] = "|".join(r["tags"])
        writer.writerow(row)
    return ExportBundle(
        format="csv",
        payload=buf.getvalue(),
        memory_count=len(records),
        tag_count=sum(len(r["tags"]) for r in records),
    )


_FORMATS = {"json": export_json, "markdown": export_markdown, "csv": export_csv}


def export_memories(store: EnrichedMemoryStore, fmt: str = "json") -> ExportBundle:
    fmt = fmt.lower()
    if fmt not in _FORMATS:
        raise ValueError(f"format must be one of {sorted(_FORMATS)}, got {fmt!r}")
    return _FORMATS[fmt](store)


# ──────────────────────────── import ───────────────────────────────────────


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value in (None, ""):
        return None
    raw = cast(str, value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"could not parse timestamp {value!r}: {exc}") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _looks_like_duplicate(
    store: EnrichedMemoryStore,
    label: str,
    *,
    threshold: float,
) -> Optional[int]:
    """Return the entry id of the closest existing memory if cos ≥ threshold."""
    if not label or threshold >= 1.0001:
        return None
    incoming = encode_text(label)
    best_id: Optional[int] = None
    best_sim = -1.0
    for e in store.episodic.entries():
        sim = float(e.key.cosine_similarity(incoming))
        if sim >= threshold and sim > best_sim:
            best_id = e.id
            best_sim = sim
    return best_id


def import_memories(
    store: EnrichedMemoryStore,
    payload: str,
    *,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> ImportReport:
    """Bulk import from a JSON export. Deduplicates near-identical labels."""
    if not 0.0 < dedup_threshold <= 1.0001:
        raise ValueError("dedup_threshold must be in (0, 1]")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload is not valid JSON: {exc}") from exc

    records = _coerce_records(data)
    new_ids: List[int] = []
    duplicate_of: Dict[int, int] = {}
    skipped_invalid = 0

    for idx, r in enumerate(records):
        label = (r.get("label") or "").strip()
        if not label:
            skipped_invalid += 1
            continue
        try:
            valid_from = _parse_dt(r.get("valid_from"))
            valid_until = _parse_dt(r.get("valid_until"))
        except ValueError:
            skipped_invalid += 1
            continue

        dup = _looks_like_duplicate(store, label, threshold=dedup_threshold)
        if dup is not None:
            duplicate_of[idx] = dup
            continue

        hv = encode_text(label)
        try:
            new_id = store.store(
                hv,
                hv,
                label,
                source=str(r.get("source") or "user_input"),
                importance=float(r.get("importance", 0.5)),
                valid_from=valid_from,
                valid_until=valid_until,
                tags=list(r.get("tags") or []),
            )
        except (ValueError, TypeError):
            skipped_invalid += 1
            continue
        new_ids.append(new_id)

    return ImportReport(
        imported=len(new_ids),
        skipped_duplicates=len(duplicate_of),
        skipped_invalid=skipped_invalid,
        new_ids=tuple(new_ids),
        duplicate_of=duplicate_of,
    )


def _coerce_records(data: Any) -> List[Dict[str, Any]]:
    """Accept either a full export envelope or a bare list of memory dicts."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        items = data.get("memories")
        if isinstance(items, list):
            return [r for r in items if isinstance(r, dict)]
    raise ValueError("import payload must be a JSON list or {memories: [...]}")


def import_iter(
    store: EnrichedMemoryStore,
    records: Iterable[Dict[str, Any]],
    *,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> ImportReport:
    """Import a Python iterable of memory dicts directly (skip JSON parse)."""
    return import_memories(
        store,
        json.dumps({"version": EXPORT_VERSION, "memories": list(records)}),
        dedup_threshold=dedup_threshold,
    )
