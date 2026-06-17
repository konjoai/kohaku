//! PyO3 Python bindings for kohaku.
//! Build with: maturin develop --features python
#![cfg(feature = "python")]
use crate::accel::cosine_topk as rust_cosine_topk;
use crate::retrieval::query;
use crate::{EpisodicMemory, HyperVector, DIMS};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyclass(name = "HyperVector")]
#[derive(Clone)]
pub struct PyHyperVector {
    inner: HyperVector,
}

#[pymethods]
impl PyHyperVector {
    #[new]
    pub fn new(seed: u64) -> Self {
        Self {
            inner: HyperVector::random(DIMS, seed),
        }
    }

    pub fn cosine_similarity(&self, other: &PyHyperVector) -> f32 {
        self.inner.cosine_similarity(&other.inner)
    }

    pub fn bind(&self, other: &PyHyperVector) -> Self {
        Self {
            inner: self.inner.bind(&other.inner),
        }
    }

    #[staticmethod]
    pub fn bundle(vectors: Vec<PyHyperVector>) -> Self {
        let refs: Vec<&HyperVector> = vectors.iter().map(|v| &v.inner).collect();
        Self {
            inner: HyperVector::bundle(&refs),
        }
    }

    pub fn permute(&self, shift: usize) -> Self {
        Self {
            inner: self.inner.permute(shift),
        }
    }

    pub fn data(&self) -> Vec<i8> {
        self.inner.data.clone()
    }

    pub fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass(name = "EpisodicMemory")]
pub struct PyEpisodicMemory {
    inner: EpisodicMemory,
}

#[pymethods]
impl PyEpisodicMemory {
    #[new]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: EpisodicMemory::new(capacity),
        }
    }

    pub fn store(&mut self, key: &PyHyperVector, value: &PyHyperVector, label: String) -> u64 {
        self.inner
            .store(key.inner.clone(), value.inner.clone(), label)
    }

    pub fn query(&self, key: &PyHyperVector, top_k: usize) -> Vec<(u64, String, f32)> {
        query(&self.inner, &key.inner, top_k)
            .into_iter()
            .map(|r| (r.entry_id, r.label, r.similarity))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn clear(&mut self) {
        self.inner.clear()
    }
}

/// Bit-packed cosine top-k over a batch of bipolar key vectors.
///
/// `query` and each row of `keys` are `+1`/`-1` component lists. Returns
/// `(index, similarity)` pairs sorted by similarity descending, ties by index.
#[pyfunction]
#[pyo3(name = "cosine_topk")]
fn py_cosine_topk(query: Vec<i8>, keys: Vec<Vec<i8>>, top_k: usize) -> Vec<(usize, f32)> {
    rust_cosine_topk(&query, &keys, top_k)
}

#[pymodule]
pub fn _kohaku_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHyperVector>()?;
    m.add_class::<PyEpisodicMemory>()?;
    m.add_function(wrap_pyfunction!(py_cosine_topk, m)?)?;
    Ok(())
}
