"""Tests for the semantic encoder (B1).

All tests inject a deterministic ``embed_fn`` so the suite never needs the
optional ``sentence-transformers`` dependency.
"""
from __future__ import annotations

import numpy as np
import pytest

from kohaku import EmbeddingEncoder, Memory, project_to_hypervector
from kohaku._pure import HyperVector
from kohaku.semantic import _PROJECTION_CACHE


def _fixed_embedding(seed: int, dim: int = 32) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_projection_returns_bipolar_hypervector():
    hv = project_to_hypervector(_fixed_embedding(1), dims=2048)
    assert isinstance(hv, HyperVector)
    assert hv.data.shape == (2048,)
    assert set(np.unique(hv.data)).issubset({-1, 1})


def test_projection_is_deterministic():
    emb = _fixed_embedding(7)
    a = project_to_hypervector(emb, dims=4096)
    b = project_to_hypervector(emb, dims=4096)
    assert np.array_equal(a.data, b.data)


def test_similar_embeddings_map_close_in_hdc_space():
    base = _fixed_embedding(3, dim=64)
    near = base + 0.01 * _fixed_embedding(99, dim=64)  # tiny perturbation
    far = _fixed_embedding(123456, dim=64)  # unrelated
    hv_base = project_to_hypervector(base, dims=8192)
    hv_near = project_to_hypervector(near, dims=8192)
    hv_far = project_to_hypervector(far, dims=8192)
    sim_near = hv_base.cosine_similarity(hv_near)
    sim_far = hv_base.cosine_similarity(hv_far)
    # SimHash preserves cosine: the near embedding stays far closer.
    assert sim_near > 0.9
    assert sim_near > sim_far + 0.4


def test_projection_rejects_empty_and_nonfinite():
    with pytest.raises(ValueError):
        project_to_hypervector(np.array([], dtype=np.float32))
    with pytest.raises(ValueError):
        project_to_hypervector(np.array([1.0, np.nan, 2.0], dtype=np.float32))


def test_projection_matrix_is_cached():
    _PROJECTION_CACHE.clear()
    emb = _fixed_embedding(5, dim=16)
    project_to_hypervector(emb, dims=1024, seed=42)
    project_to_hypervector(_fixed_embedding(6, dim=16), dims=1024, seed=42)
    # Same (embedding_dim, dims, seed) → exactly one cached matrix.
    assert (16, 1024, 42) in _PROJECTION_CACHE
    assert len(_PROJECTION_CACHE) == 1


def test_encoder_with_injected_embed_fn():
    embeddings = {
        "wine": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "merlot": np.array([0.98, 0.02, 0.0], dtype=np.float32),
        "rocket": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    enc = EmbeddingEncoder(embed_fn=lambda t: embeddings[t], dims=8192)
    hv_wine = enc.encode("wine")
    hv_merlot = enc("merlot")  # __call__ path
    hv_rocket = enc.encode("rocket")
    assert hv_wine.cosine_similarity(hv_merlot) > hv_wine.cosine_similarity(hv_rocket)


def test_encoder_empty_text_rejected():
    enc = EmbeddingEncoder(embed_fn=lambda t: _fixed_embedding(1))
    with pytest.raises(ValueError):
        enc.encode("   ")


def test_encoder_missing_dependency_raises_clean_error(monkeypatch):
    # Simulate sentence-transformers being absent.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    enc = EmbeddingEncoder()  # no embed_fn → must lazily try the import
    with pytest.raises(ImportError, match="sentence-transformers"):
        enc.encode("hello")


def test_memory_with_semantic_encoder_beats_lexical_on_paraphrase():
    # Hand-built embedding space where paraphrases are close but share no tokens.
    space = {
        "User prefers Italian wine": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "The customer enjoys a glass of merlot": np.array([0.95, 0.05, 0.0], dtype=np.float32),
        "Quarterly revenue grew by twelve percent": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    enc = EmbeddingEncoder(embed_fn=lambda t: space[t], dims=8192)
    mem = Memory(encoder=enc)
    mem.store("User prefers Italian wine")
    mem.store("Quarterly revenue grew by twelve percent")
    hits = mem.query("The customer enjoys a glass of merlot", reinforce=False)
    # Despite zero shared tokens, the wine memory ranks first.
    assert hits[0].text == "User prefers Italian wine"


def test_memory_semantic_save_load_roundtrip(tmp_path):
    space = {
        "alpha fact": np.array([1.0, 0.0], dtype=np.float32),
        "beta fact": np.array([0.0, 1.0], dtype=np.float32),
    }
    enc = EmbeddingEncoder(embed_fn=lambda t: space[t], dims=4096)
    mem = Memory(encoder=enc)
    mem.store("alpha fact")
    mem.store("beta fact")
    path = str(tmp_path / "sem.json")
    mem.save(path)

    restored = Memory.load(path, encoder=enc)
    orig = mem.query("alpha fact", reinforce=False)[0]
    again = restored.query("alpha fact", reinforce=False)[0]
    assert again.text == orig.text
    assert again.similarity == pytest.approx(orig.similarity)


def test_load_warns_on_encoder_mismatch(tmp_path, caplog):
    space = {"x fact": np.array([1.0, 0.0], dtype=np.float32)}
    enc = EmbeddingEncoder(embed_fn=lambda t: space[t], dims=2048)
    mem = Memory(encoder=enc)
    mem.store("x fact")
    path = str(tmp_path / "m.json")
    mem.save(path)
    with caplog.at_level("WARNING"):
        Memory.load(path)  # no encoder supplied
    assert any("custom encoder" in r.message for r in caplog.records)
