//! PyO3 Python bindings for kohaku.
//! Build with: maturin develop --features python
#![cfg(feature = "python")]
use pyo3::prelude::*;
use crate::{HyperVector, EpisodicMemory, DIMS};
use crate::retrieval::query;

#[pyclass(name = "HyperVector")]
#[derive(Clone)]
pub struct PyHyperVector {
    inner: HyperVector,
}

#[pymethods]
impl PyHyperVector {
    #[new]
    pub fn new(seed: u64) -> Self {
        Self { inner: HyperVector::random(DIMS, seed) }
    }

    pub fn cosine_similarity(&self, other: &PyHyperVector) -> f32 {
        self.inner.cosine_similarity(&other.inner)
    }

    pub fn bind(&self, other: &PyHyperVector) -> Self {
        Self { inner: self.inner.bind(&other.inner) }
    }

    pub fn bundle(vectors: Vec<PyRef<PyHyperVector>>) -> Self {
        let refs: Vec<&HyperVector> = vectors.iter().map(|v| &v.inner).collect();
        Self { inner: HyperVector::bundle(&refs) }
    }

    pub fn permute(&self, shift: usize) -> Self {
        Self { inner: self.inner.permute(shift) }
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
        Self { inner: EpisodicMemory::new(capacity) }
    }

    pub fn store(&mut self, key: &PyHyperVector, value: &PyHyperVector, label: String) -> u64 {
        self.inner.store(key.inner.clone(), value.inner.clone(), label)
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

    pub fn clear(&mut self) {
        self.inner.clear()
    }
}

#[pymodule]
pub fn _kohaku_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHyperVector>()?;
    m.add_class::<PyEpisodicMemory>()?;
    Ok(())
}
