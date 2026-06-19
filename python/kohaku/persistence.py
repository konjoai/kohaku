"""Persistence — serialize EpisodicMemory to JSON and binary `.hkb` format.

Two formats are supported:

* **JSON** — human-readable, round-trip compatible. Each entry stores its bipolar ±1 vector
  as a list of ints. Larger on disk but easy to inspect and diff.
* **`.hkb` binary** — packed bit format. Each ±1 component becomes a single bit (+1→1,
  -1→0), giving an 8x size reduction over INT8 and ~64x over JSON. Layout:

      magic        : b"KHKU"        (4 bytes)
      version      : u16            (little-endian)
      dims         : u32
      capacity     : u32
      next_id      : u64
      timestamp    : u64
      num_entries  : u32
      [entries...]
        id          : u64
        timestamp   : u64
        label_len   : u16
        label_utf8  : label_len bytes
        key_bits    : dims/8 bytes (numpy.packbits, big-endian within byte)
        value_bits  : dims/8 bytes

The bit-packing assumes dims % 8 == 0 (default 10_000 is *not* divisible by 8, so we pad
to the next multiple of 8 and store the original `dims` in the header to truncate on load).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Union

import numpy as np

from kohaku._pure import DIMS, EpisodicMemory, HyperVector, MemoryEntry

PathLike = Union[str, Path]

_MAGIC = b"KHKU"
_VERSION = 1
_HEADER_FMT = (
    "<4sHIIQQI"  # magic, version, dims, capacity, next_id, timestamp, num_entries
)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_ENTRY_HDR_FMT = "<QQH"  # id, timestamp, label_len
_ENTRY_HDR_SIZE = struct.calcsize(_ENTRY_HDR_FMT)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def save_json(memory: EpisodicMemory, path: PathLike) -> None:
    """Serialize an EpisodicMemory to JSON at *path*."""
    payload = {
        "format": "kohaku-json",
        "version": _VERSION,
        "dims": int(len(memory.entries()[0].key)) if not memory.is_empty else DIMS,
        "capacity": memory._capacity,
        "next_id": memory._next_id,
        "timestamp": memory._timestamp,
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "label": e.label,
                "key": e.key.data.tolist(),
                "value": e.value.data.tolist(),
            }
            for e in memory.entries()
        ],
    }
    Path(path).write_text(json.dumps(payload))


def load_json(path: PathLike) -> EpisodicMemory:
    """Load an EpisodicMemory previously written by :func:`save_json`."""
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "kohaku-json":
        raise ValueError(f"Not a kohaku-json file: {path}")
    mem = EpisodicMemory(capacity=int(payload["capacity"]))
    for entry in payload["entries"]:
        key = HyperVector(np.asarray(entry["key"], dtype=np.int8))
        value = HyperVector(np.asarray(entry["value"], dtype=np.int8))
        mem._entries.append(
            MemoryEntry(
                id=int(entry["id"]),
                key=key,
                value=value,
                label=str(entry["label"]),
                timestamp=int(entry["timestamp"]),
            )
        )
    mem._next_id = int(payload["next_id"])
    mem._timestamp = int(payload["timestamp"])
    return mem


# ---------------------------------------------------------------------------
# Binary .hkb
# ---------------------------------------------------------------------------


def _pack_bipolar(vec: np.ndarray, padded_dims: int) -> bytes:
    """Pack ±1 int8 array → bits (1 for +1, 0 for -1), zero-padded to padded_dims."""
    bits = np.where(vec > 0, 1, 0).astype(np.uint8)
    if bits.shape[0] < padded_dims:
        bits = np.concatenate(
            [bits, np.zeros(padded_dims - bits.shape[0], dtype=np.uint8)]
        )
    packed = np.packbits(bits, bitorder="big")
    return packed.tobytes()


def _unpack_bipolar(buf: bytes, dims: int) -> np.ndarray:
    """Unpack bits → ±1 int8 array of length `dims`."""
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="big")
    bits = bits[:dims]
    return np.where(bits == 1, np.int8(1), np.int8(-1)).astype(np.int8)


def save_binary(memory: EpisodicMemory, path: PathLike) -> None:
    """Serialize an EpisodicMemory to the `.hkb` binary format at *path*."""
    entries = memory.entries()
    dims = int(len(entries[0].key)) if entries else DIMS
    padded_dims = (dims + 7) & ~7  # round up to multiple of 8
    bytes_per_vec = padded_dims // 8

    with Path(path).open("wb") as f:
        f.write(
            struct.pack(
                _HEADER_FMT,
                _MAGIC,
                _VERSION,
                dims,
                memory._capacity,
                memory._next_id,
                memory._timestamp,
                len(entries),
            )
        )
        for e in entries:
            label_bytes = e.label.encode("utf-8")
            if len(label_bytes) > 0xFFFF:
                raise ValueError(f"label too long ({len(label_bytes)} > 65535 bytes)")
            f.write(struct.pack(_ENTRY_HDR_FMT, e.id, e.timestamp, len(label_bytes)))
            f.write(label_bytes)
            key_buf = _pack_bipolar(e.key.data, padded_dims)
            val_buf = _pack_bipolar(e.value.data, padded_dims)
            assert len(key_buf) == bytes_per_vec and len(val_buf) == bytes_per_vec
            f.write(key_buf)
            f.write(val_buf)


def load_binary(path: PathLike) -> EpisodicMemory:
    """Load an EpisodicMemory previously written by :func:`save_binary`."""
    data = Path(path).read_bytes()
    if len(data) < _HEADER_SIZE:
        raise ValueError(f"file too small to be .hkb: {path}")
    magic, version, dims, capacity, next_id, timestamp, num_entries = struct.unpack(
        _HEADER_FMT, data[:_HEADER_SIZE]
    )
    if magic != _MAGIC:
        raise ValueError(f"Not a .hkb file (bad magic): {path}")
    if version != _VERSION:
        raise ValueError(f"Unsupported .hkb version: {version}")

    padded_dims = (dims + 7) & ~7
    bytes_per_vec = padded_dims // 8
    mem = EpisodicMemory(capacity=int(capacity))
    offset = _HEADER_SIZE

    for _ in range(num_entries):
        if offset + _ENTRY_HDR_SIZE > len(data):
            raise ValueError("truncated .hkb file (entry header)")
        eid, ets, label_len = struct.unpack(
            _ENTRY_HDR_FMT, data[offset : offset + _ENTRY_HDR_SIZE]
        )
        offset += _ENTRY_HDR_SIZE
        label = data[offset : offset + label_len].decode("utf-8")
        offset += label_len
        if offset + 2 * bytes_per_vec > len(data):
            raise ValueError("truncated .hkb file (vector payload)")
        key = HyperVector(_unpack_bipolar(data[offset : offset + bytes_per_vec], dims))
        offset += bytes_per_vec
        value = HyperVector(
            _unpack_bipolar(data[offset : offset + bytes_per_vec], dims)
        )
        offset += bytes_per_vec
        mem._entries.append(
            MemoryEntry(
                id=int(eid), key=key, value=value, label=label, timestamp=int(ets)
            )
        )

    mem._next_id = int(next_id)
    mem._timestamp = int(timestamp)
    return mem


# ---------------------------------------------------------------------------
# Convenience dispatch by file extension
# ---------------------------------------------------------------------------


def save(memory: EpisodicMemory, path: PathLike) -> None:
    """Save by extension: `.json` → JSON, `.hkb` → binary. Anything else raises."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        save_json(memory, path)
    elif suffix == ".hkb":
        save_binary(memory, path)
    else:
        raise ValueError(f"Unknown extension {suffix!r}; use .json or .hkb")


def load(path: PathLike) -> EpisodicMemory:
    """Load by extension: `.json` → JSON, `.hkb` → binary."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix == ".hkb":
        return load_binary(path)
    raise ValueError(f"Unknown extension {suffix!r}; use .json or .hkb")


# ---------------------------------------------------------------------------
# Namespaced stores (TenantMemoryStore / SharedMemoryPool)
# ---------------------------------------------------------------------------
#
# Both stores are a ``{namespace_id: EpisodicMemory}`` registry plus a little
# config. They persist as a *directory*: one packed ``.hkb`` per namespace (via
# :func:`save_binary`) plus a ``manifest.json`` that records the store config and
# maps each namespace id to its file. This mirrors :mod:`kohaku.system` and reuses
# the single-memory codec wholesale, so there is no second binary format to keep
# in round-trip parity. Namespace ids are *never* used as filenames (they may
# contain path separators or unicode) — files are index-named and the real id
# lives in the manifest.

_NS_MANIFEST = "manifest.json"


def save_namespaces(
    namespaces: dict[str, EpisodicMemory],
    directory: PathLike,
    *,
    fmt: str,
    config: dict[str, Any],
) -> None:
    """Write a ``{id: EpisodicMemory}`` registry to ``directory``.

    ``fmt`` is a format tag stored in the manifest and checked on load (so a
    tenant directory can't be loaded as a shared pool). ``config`` carries the
    store-level fields (dimension, capacity, …) to restore on load.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (ns_id, mem) in enumerate(namespaces.items()):
        fname = f"ns_{i}.hkb"
        save_binary(mem, directory / fname)
        records.append({"id": ns_id, "file": fname})
    manifest = {
        "format": fmt,
        "version": _VERSION,
        "config": config,
        "namespaces": records,
    }
    (directory / _NS_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )


def load_namespaces(
    directory: PathLike, *, fmt: str
) -> tuple[dict[str, Any], dict[str, EpisodicMemory]]:
    """Read back a registry written by :func:`save_namespaces`.

    Returns ``(config, namespaces)``. Raises ``FileNotFoundError`` if the
    manifest is missing and ``ValueError`` if its format tag doesn't match
    ``fmt``.
    """
    directory = Path(directory)
    manifest_path = directory / _NS_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {_NS_MANIFEST} in {directory}")
    manifest = json.loads(manifest_path.read_text())
    found = manifest.get("format")
    if found != fmt:
        raise ValueError(f"expected format {fmt!r}, got {found!r} in {directory}")
    namespaces: dict[str, EpisodicMemory] = {}
    for rec in manifest.get("namespaces", []):
        namespaces[str(rec["id"])] = load_binary(directory / rec["file"])
    return manifest.get("config", {}), namespaces
