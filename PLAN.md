# Kohaku — Development Plan

## Current Version: v0.1.0

## Phase 1: Core HDC Engine (v0.1.0) ✅
- [x] Hypervector arithmetic: random, bundle, bind, permute
- [x] Cosine similarity and Hamming distance
- [x] Episodic memory store with capacity management (FIFO eviction)
- [x] Associative retrieval (top-k, threshold)
- [x] CLI: demo + bench subcommands
- [x] Python bridge script with KohakuMemory class
- [x] Integration test suite (8 core tests + bonus)

## Phase 2: Python Bindings (v0.2.0)
- [ ] PyO3 bindings for EpisodicMemory and HyperVector
- [ ] pip-installable Python package
- [ ] Async query support

## Phase 3: LLM Integration (v0.3.0)
- [ ] Context window memory manager
- [ ] Attention-guided encoding
- [ ] HuggingFace Transformers hooks
- [ ] OpenAI API compatible memory layer

## Phase 4: Persistence (v0.4.0)
- [ ] Serialize/deserialize memory to disk (JSON + binary)
- [ ] Memory consolidation (semantic clustering)
- [ ] Forgetting curves / temporal decay

## Phase 5: Learning (v0.5.0)
- [ ] Online HDC learning: update item memory from feedback
- [ ] Hopfield network associator layer
- [ ] Episodic vs semantic memory distinction
