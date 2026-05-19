"""Kohaku demo server — exposes the *real* kohaku library as JSON over HTTP.

Run from the repo root:

    python3 demo/server.py
        # or:
    PYTHONPATH=python python3 demo/server.py [--port 8000]

Then open http://localhost:8000/ — the demo page will detect the live API
and switch from offline-simulation mode to real-data mode automatically.

Endpoints
=========
GET  /                   → demo/index.html
GET  /api/health         → {real_kohaku, backend, dims, num_concepts, ...}
POST /api/encode         → {concept} → {vector_preview, dim, norm, latency_ms}
POST /api/add-concept    → {concept, label?} → updates graph, returns full graph
POST /api/query          → {concept, days_since, half_life} → real query_with_decay
GET  /api/graph          → full concept list + symmetric similarity matrix
POST /api/save           → write .hkb to disk → {path, size_bytes_hkb, size_bytes_json}
POST /api/load           → read .hkb from disk → restored graph
POST /api/reset          → wipe state, re-seed with defaults

All numbers returned are computed by the live kohaku library — no mocks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Tuple

# ── Make the in-repo kohaku package importable ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PY_PKG = ROOT / "python"
if str(PY_PKG) not in sys.path:
    sys.path.insert(0, str(PY_PKG))

import kohaku  # noqa: E402
from kohaku import (  # noqa: E402
    DecayConfig,
    EpisodicMemory,
    encode_text,
    load,
    query,
    save,
)
from kohaku._pure import DIMS  # noqa: E402
from kohaku.decay import decay_weight  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
INDEX_HTML = DEMO_DIR / "index.html"
STATE_FILE = DEMO_DIR / ".kohaku_demo.hkb"
STATE_JSON = DEMO_DIR / ".kohaku_demo.json"

# ── Seed concepts: short label + full phrase to encode ──────────────────────
# Phrases overlap deliberately so cosine similarity reflects real semantic
# relatedness for the demo (single-word tokens are near-orthogonal in
# kohaku.encode_text — by design).
SEED_CONCEPTS: List[Tuple[str, str, str, str]] = [
    # (id,         label,          phrase,                               color)
    ("cat",      "🐈 cat",       "the cat sat on the mat",              "#ffd394"),
    ("kitten",   "😺 kitten",    "a kitten sat on the rug",             "#ffb86c"),
    ("feline",   "🐅 feline",    "a feline rested on the mat",          "#ff9f3f"),
    ("espresso", "☕ espresso",  "espresso brewed dark and bitter",     "#f5c542"),
    ("coffee",   "🫖 coffee",    "iced coffee in the afternoon",        "#ffe28a"),
    ("river",    "🌊 river",     "the river flowing toward the ocean",  "#7dd87d"),
    ("ocean",    "🌊 ocean",     "ocean waves under the moonlight",     "#6fd2c0"),
]

PALETTE = ["#ffd394", "#ffb86c", "#ff9f3f", "#f5c542", "#ffe28a",
           "#7dd87d", "#6fd2c0", "#ff7e8a", "#ffaaaa", "#bba0ff"]


# ────────────────────────────────────────────────────────────────────────────
#  State
# ────────────────────────────────────────────────────────────────────────────

class State:
    """Thread-safe holder for the live EpisodicMemory + concept metadata."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.memory: EpisodicMemory = EpisodicMemory(capacity=512)
        self.meta: List[Dict[str, Any]] = []  # parallel to memory.entries()
        self.seed()

    def seed(self) -> None:
        with self.lock:
            self.memory = EpisodicMemory(capacity=512)
            self.meta = []
            for cid, label, phrase, color in SEED_CONCEPTS:
                self._store(cid, label, phrase, color)

    def _store(self, cid: str, label: str, phrase: str, color: str) -> Dict[str, Any]:
        # Encoded once, used as both key and value (key for retrieval).
        hv = encode_text(phrase)
        eid = self.memory.store(hv, hv, label=cid)
        info = {
            "entry_id": eid,
            "id": cid,
            "label": label,
            "phrase": phrase,
            "color": color,
        }
        self.meta.append(info)
        return info

    def add(self, concept: str, label: str | None = None) -> Dict[str, Any]:
        with self.lock:
            cid = concept.strip().lower()[:64]
            if not cid:
                raise ValueError("concept must not be empty")
            for m in self.meta:
                if m["id"] == cid:
                    return m  # idempotent
            color = PALETTE[len(self.meta) % len(PALETTE)]
            shown = label or concept
            return self._store(cid, shown, concept.strip(), color)

    def graph(self) -> Dict[str, Any]:
        """Full concept list + symmetric similarity matrix from real cosine."""
        with self.lock:
            entries = self.memory.entries()
            n = len(entries)
            sims: List[List[float]] = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    s = float(entries[i].key.cosine_similarity(entries[j].key))
                    sims[i][j] = s
                    sims[j][i] = s
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append({"i": i, "j": j, "sim": round(sims[i][j], 4)})
            nodes = []
            for idx, m in enumerate(self.meta):
                nodes.append({
                    "index": idx,
                    "id": m["id"],
                    "label": m["label"],
                    "phrase": m["phrase"],
                    "color": m["color"],
                    "entry_id": m["entry_id"],
                })
            return {"nodes": nodes, "edges": edges, "dims": DIMS}

    def query_with_decay(
        self, concept: str, days_since: float, half_life: float, top_k: int = 8
    ) -> Dict[str, Any]:
        with self.lock:
            t0 = time.time()
            hv = encode_text(concept)
            results = query(self.memory, hv, top_k=top_k)
            cfg = DecayConfig(half_life=max(0.5, half_life))
            # Apply real decay_weight from kohaku.decay (ages in "days", treating
            # ticks 1:1 with days for the demo — a tunable knob in real use).
            age = max(0, int(round(days_since)))
            w = decay_weight(age, cfg)
            matches = []
            for r in results:
                meta = next((m for m in self.meta if m["entry_id"] == r.entry_id), None)
                if not meta:
                    continue
                matches.append({
                    "entry_id": r.entry_id,
                    "id": meta["id"],
                    "label": meta["label"],
                    "raw_similarity": round(float(r.similarity), 4),
                    "decay_weight": round(w, 4),
                    "decayed_strength": round(float(r.similarity) * w, 4),
                })
            elapsed = (time.time() - t0) * 1000
            return {
                "query": concept,
                "days_since": age,
                "half_life": half_life,
                "decay_weight": round(w, 4),
                "matches": matches,
                "latency_ms": round(elapsed, 2),
                "vector_preview": [int(x) for x in hv.data[:8]],
            }

    def encode_only(self, concept: str) -> Dict[str, Any]:
        t0 = time.time()
        hv = encode_text(concept)
        elapsed = (time.time() - t0) * 1000
        norm = float((hv.data.astype("int64") ** 2).sum() ** 0.5)
        return {
            "concept": concept,
            "dim": int(len(hv)),
            "vector_preview": [int(x) for x in hv.data[:8]],
            "vector_tail": [int(x) for x in hv.data[-4:]],
            "norm": round(norm, 3),
            "n_plus": int((hv.data == 1).sum()),
            "n_minus": int((hv.data == -1).sum()),
            "latency_ms": round(elapsed, 2),
        }

    def save_to_disk(self) -> Dict[str, Any]:
        with self.lock:
            save(self.memory, STATE_FILE)
            save(self.memory, STATE_JSON)
            size_hkb = STATE_FILE.stat().st_size
            size_json = STATE_JSON.stat().st_size
            return {
                "hkb_path": str(STATE_FILE.relative_to(ROOT)),
                "json_path": str(STATE_JSON.relative_to(ROOT)),
                "size_bytes_hkb": size_hkb,
                "size_bytes_json": size_json,
                "ratio": round(size_json / max(1, size_hkb), 1),
                "num_entries": len(self.memory.entries()),
                "dims": DIMS,
            }

    def load_from_disk(self) -> Dict[str, Any]:
        with self.lock:
            if not STATE_FILE.exists():
                raise FileNotFoundError(f"No saved memory at {STATE_FILE}")
            mem = load(STATE_FILE)
            # Rebuild meta from labels (we stored cid as label).
            self.memory = mem
            self.meta = []
            for idx, e in enumerate(mem.entries()):
                cid = e.label
                seed = next((s for s in SEED_CONCEPTS if s[0] == cid), None)
                if seed:
                    label, phrase, color = seed[1], seed[2], seed[3]
                else:
                    label = cid
                    phrase = cid
                    color = PALETTE[idx % len(PALETTE)]
                self.meta.append({
                    "entry_id": e.id,
                    "id": cid,
                    "label": label,
                    "phrase": phrase,
                    "color": color,
                })
            return {
                "loaded_from": str(STATE_FILE.relative_to(ROOT)),
                "size_bytes_hkb": STATE_FILE.stat().st_size,
                "num_entries": len(self.memory.entries()),
            }

    def health(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "real_kohaku": True,
                "version": kohaku.__version__,
                "backend": kohaku._BACKEND,
                "dims": DIMS,
                "num_concepts": len(self.memory.entries()),
                "capacity": self.memory._capacity,
                "internal_clock": self.memory._timestamp,
            }


# ────────────────────────────────────────────────────────────────────────────
#  HTTP handler
# ────────────────────────────────────────────────────────────────────────────

class DemoHandler(BaseHTTPRequestHandler):
    state: State  # injected on the class before serve_forever

    # Quieter logs
    def log_message(self, fmt: str, *args: Any) -> None:
        ts = time.strftime("%H:%M:%S")
        sys.stdout.write(f"[{ts}] {self.address_string()} {fmt % args}\n")

    # ── helpers ─────────────────────────────────────────────────────────────
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": f"missing file: {path.name}"}, status=404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"invalid JSON body: {e}")

    # ── verbs ───────────────────────────────────────────────────────────────
    def do_OPTIONS(self) -> None:  # noqa: N802 (stdlib name)
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._send_file(INDEX_HTML, "text/html; charset=utf-8")
            elif path == "/api/health":
                self._send_json(self.state.health())
            elif path == "/api/graph":
                self._send_json(self.state.graph())
            elif path == "/favicon.ico":
                self.send_response(204)
                self._cors()
                self.end_headers()
            else:
                self._send_json({"error": f"unknown path {path!r}"}, status=404)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_json()
            if path == "/api/encode":
                concept = (body.get("concept") or "").strip()
                if not concept:
                    raise ValueError("'concept' is required")
                self._send_json(self.state.encode_only(concept))
            elif path == "/api/add-concept":
                concept = (body.get("concept") or "").strip()
                label = body.get("label")
                if not concept:
                    raise ValueError("'concept' is required")
                self.state.add(concept, label)
                self._send_json({"added": concept, "graph": self.state.graph()})
            elif path == "/api/query":
                concept = (body.get("concept") or "").strip()
                if not concept:
                    raise ValueError("'concept' is required")
                days = float(body.get("days_since") or 0)
                half = float(body.get("half_life") or 30.0)
                top_k = int(body.get("top_k") or 8)
                self._send_json(
                    self.state.query_with_decay(concept, days, half, top_k=top_k)
                )
            elif path == "/api/save":
                self._send_json(self.state.save_to_disk())
            elif path == "/api/load":
                self._send_json({**self.state.load_from_disk(), "graph": self.state.graph()})
            elif path == "/api/reset":
                self.state.seed()
                self._send_json({"ok": True, "graph": self.state.graph()})
            else:
                self._send_json({"error": f"unknown path {path!r}"}, status=404)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=400)


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Kohaku demo server")
    p.add_argument("--port", type=int, default=int(os.environ.get("KOHAKU_PORT", 8000)))
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    DemoHandler.state = State()

    httpd = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(
        "\n"
        f"  ▶ Kohaku demo server  ·  v{kohaku.__version__}  ·  backend={kohaku._BACKEND}\n"
        f"  ▶ Dims = {DIMS}  ·  Seeded {len(SEED_CONCEPTS)} concepts\n"
        f"  ▶ http://{args.host}:{args.port}/      ← open this in your browser\n"
        f"  ▶ http://{args.host}:{args.port}/api/health\n"
        "\n  Ctrl-C to stop.\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
