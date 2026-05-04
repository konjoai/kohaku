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

* 🦀 Rust core — high-performance HDC engine
* 🐍 Python API — LLM integration

---

## 🚀 Quick Start

```bash
pip install kohaku
```

```python
from kohaku import Memory

mem = Memory()
mem.store("User prefers Italian wine")
mem.query("What does the user like?")
```

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
