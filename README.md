# 🦀 Kōhaku

![Language](https://img.shields.io/badge/language-rust-blue) ![ML](https://img.shields.io/badge/core-python-yellow) ![License](https://img.shields.io/badge/license-busl--1.1-green) ![Status](https://img.shields.io/badge/status-active-brightgreen)

> 🧠 Episodic memory engine for LLMs — persistent, associative, and beyond context windows.

---

## 🍂 Meaning

**Kōhaku (琥珀)** — *amber, preserved in time.*

Like insects trapped in amber, memories are captured, compressed, and preserved — not lost to context limits.

---

## 🚀 What it is

Kōhaku is a **neural episodic memory system**:

* Stores experiences as HDC hypervectors
* Retrieves via associative similarity
* Works as a **drop-in memory layer for any LLM**

Not:

* ❌ RAG
* ❌ vector database

But:

> ✅ learned memory with recall

---

## ❗ The problem

LLMs forget.

* Context windows are finite
* RAG loses nuance
* Summaries lose detail

There is no true **memory system**.

---

## 🧠 What you learn

* Hyperdimensional computing (HDC)
* Associative memory / Hopfield networks
* Memory-augmented architectures
* Episodic vs semantic memory

---

## ⚙️ Architecture

* 🐍 **Python** — the full engine and API; the pure-Python path is the
  correctness baseline and works with zero native dependencies.
* 🦀 **Rust accelerator** (optional) — bit-packed XOR + popcount cosine top-k
  behind a PyO3 extension. `pip install .` (from the repo root, via maturin)
  builds `kohaku._kohaku_rs` (`kohaku._BACKEND == "rust-accel"`), parity-tested
  against NumPy in CI. Retrieval crosses the FFI boundary **zero-copy** (borrowed
  `int8` arrays, no list marshaling). The big win is `kohaku.RetrievalIndex`, a
  **resident packed index** that packs the keys once and is **~160–230× faster
  than NumPy** on repeated probes (`benchmarks/bench_backends.py`); `query` and
  `query_with_decay` use a per-memory cached index automatically. One-shot
  batches stay on NumPy (re-packing every call is ~parity with BLAS).

```bash
pip install ./python    # pure-Python baseline
pip install .           # + Rust accelerator (needs a Rust toolchain + maturin)
```

---

## 🚀 Quick Start

```bash
pip install kohaku
```

```python
from kohaku import Memory

mem = Memory()
mem.store("User prefers Italian wine")
mem.store("User is allergic to shellfish", importance=0.9, tags=["health"])

hits = mem.query("What does the user like to drink?")
for h in hits:
    print(h.text, round(h.similarity, 3))
# → User prefers Italian wine 0.63

mem.save("user.json")          # labels + metadata; HVs re-derived on load
mem2 = Memory.load("user.json")
```

`Memory` is the one-line front door: store strings, get ranked `MemoryHit`
results back (`.text`, `.similarity`, `.salience`, `.source`, `.tags`). It wraps
the full `EnrichedMemoryStore` — temporal validity, salience, source-trust,
tags — behind a string-in/string-out API. Reach for `EnrichedMemoryStore`,
`MemorySystem`, and friends directly when you need provenance graphs, version
history, or consolidation daemons.

## 🧬 Semantic recall (opt-in)

The default encoder bundles per-*token* hypervectors, so similarity is token
overlap — *"the customer enjoys merlot"* won't match *"User prefers Italian
wine"*. For meaning-based recall, plug in an `EmbeddingEncoder` that projects
a dense embedding into HDC space (SimHash — sign of a fixed random projection,
which approximately preserves cosine):

```bash
pip install "kohaku[semantic]"     # pulls sentence-transformers
```

```python
from kohaku import Memory, EmbeddingEncoder

enc = EmbeddingEncoder(model_name="all-MiniLM-L6-v2")   # or embed_fn=<your callable>
mem = Memory(encoder=enc)
mem.store("User prefers Italian wine")
mem.query("the customer enjoys a glass of merlot")[0].text
# → 'User prefers Italian wine'   (zero shared tokens, still matches)
```

`EmbeddingEncoder` takes any `embed_fn` (`str -> float array`) — sentence-
transformers, OpenAI embeddings, your own — so there's no hard dependency. A
store saved with a custom encoder must be reloaded with the same one
(`Memory.load(path, encoder=enc)`).

## ⚡ Scaling past 10⁴ memories

Exact cosine retrieval is `O(N·D)` per query. Flip on the bipolar-LSH index to
narrow each similarity query to a small candidate set before exact ranking:

```python
mem = Memory(ann=True)            # maintains a kohaku.ann.LSHIndex
# ... store thousands of memories ...
mem.query("...")                  # sub-linear: LSH candidates, exact re-rank
```

Results are unchanged except for the rare LSH miss — candidates are always
scored with exact cosine, and salience/recency sorts or empty candidate sets
fall back to a full scan. `LSHIndex` is pure NumPy (no FAISS/hnswlib) and can
be used standalone.

## 📦 Whole-system snapshots

`save_system` / `load_system` persist an entire enriched setup — episodic
hypervectors, per-memory metadata, and the provenance / version / relationship
side stores — into one directory with a manifest:

```python
from kohaku import save_system, load_system

save_system(store, "snapshot/", provenance=pg, versions=vs, relationships=rel)
bundle = load_system("snapshot/")
bundle.store, bundle.provenance, bundle.versions, bundle.relationships
```

SQLite side stores are copied via the backup API (so `:memory:` stores persist
too), and recall is exact after the round-trip.

---

## 💾 Persistence (v0.4.0)

```python
from kohaku import EpisodicMemory, save, load

mem = EpisodicMemory(capacity=1000)
# ... store entries ...
save(mem, "memories.hkb")        # packed binary, ~10x smaller than JSON
save(mem, "memories.json")       # human-readable

mem2 = load("memories.hkb")      # round-trip preserves IDs, timestamps, recall
```

## 🌱 Consolidation

```python
from kohaku import consolidate_to_memory

semantic = consolidate_to_memory(mem, similarity_threshold=0.3)
# Greedy bundle-of-bundles clustering: N noisy episodic traces → K semantic centroids.
```

## 🧠 Online learning + Hopfield + episodic↔semantic (v0.5.0)

```python
from kohaku import ItemMemory, HopfieldAssociator, MemorySystem, encode_text

# Online HDC learning — prototypes update with every example
im = ItemMemory()
for example in cat_examples:
    im.add("cat", encode_text(example))
im.train_from_feedback("cat", encode_text("a dog barked"), correct=False)
top = im.predict(encode_text("a kitten napping"), top_k=3)

# Modern Hopfield — clean noisy queries by softmax-weighted retrieval
hop = HopfieldAssociator(beta=0.05)
for proto in canonical_prototypes:
    hop.store(proto)
cleaned = hop.complete(noisy_query)

# Combined episodic + semantic store with sleep-style consolidation
ms = MemorySystem(episodic_capacity=1000)
ms.store_episode(key, value, label="meeting on monday")
ms.consolidate_to_semantic(similarity_threshold=0.3)  # promote clusters → prototypes
results = ms.recall(query, top_k=3, use_decay=True)   # tagged by source
```

## 🕰️ Temporal decay

```python
from kohaku import DecayConfig, query_with_decay

cfg = DecayConfig(half_life=100.0, floor=0.05)
results = query_with_decay(mem, query_key, top_k=5, config=cfg)
# Older memories decay exponentially: weight = max(0.5 ** (age / half_life), floor)
```

---

## 🎬 Live demo

```bash
python demo/server.py        # starts a localhost server with REAL kohaku
open http://127.0.0.1:8000
```

The page detects the API and switches from offline simulation to **live mode** — every
similarity number, decay weight, and `.hkb` file size you see is computed by the live
library. Add a phrase, click any node, drag the days slider, hit save — it's all real.

```bash
PYTHONPATH=python python3 demo/demo.py    # rich-terminal walkthrough
```

---

## 🎯 Vision

> Give models memory — not just context.
