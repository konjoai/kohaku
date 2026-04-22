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

## 🎯 Vision

> Give models memory — not just context.
