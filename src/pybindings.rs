//! PyO3 Python bindings for kohaku.
//! Build with: maturin develop --features python
#![cfg(feature = "python")]
use crate::accel::{cosine_topk_rows, PackedIndex};
use crate::retrieval::query;
use crate::{EpisodicMemory, HyperVector, DIMS};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
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

/// Validate that a `(n_rows, dims)` key matrix matches a query of length `qlen`.
fn check_dims(qlen: usize, cols: usize) -> PyResult<()> {
    if cols != qlen {
        return Err(PyValueError::new_err(format!(
            "query length {qlen} != key dims {cols}"
        )));
    }
    Ok(())
}

/// Bit-packed cosine top-k over a batch of bipolar key vectors (zero-copy).
///
/// `query` is a 1-D `int8` array of length `D`; `keys` is a C-contiguous
/// `(N, D)` `int8` array. Both are borrowed as Rust slices with no per-element
/// marshaling — the win that lets the popcount kernel beat the list-of-lists
/// path. Returns `(index, similarity)` pairs sorted by similarity descending,
/// ties broken by ascending index.
#[pyfunction]
#[pyo3(name = "cosine_topk")]
fn py_cosine_topk(
    query: PyReadonlyArray1<i8>,
    keys: PyReadonlyArray2<i8>,
    top_k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    let q = query.as_slice()?;
    let view = keys.as_array();
    let (n_rows, cols) = (view.nrows(), view.ncols());
    check_dims(q.len(), cols)?;
    let k = keys.as_slice()?;
    Ok(cosine_topk_rows(q, k, n_rows, cols, top_k))
}

/// Resident bit-packed index over a fixed set of bipolar key vectors.
///
/// Pack the keys once, then query repeatedly: each call marshals only the query
/// across the FFI boundary, so amortized retrieval beats re-streaming every
/// float through BLAS.
#[pyclass(name = "PackedIndex")]
pub struct PyPackedIndex {
    inner: PackedIndex,
}

#[pymethods]
impl PyPackedIndex {
    #[new]
    fn new(keys: PyReadonlyArray2<i8>) -> PyResult<Self> {
        let view = keys.as_array();
        let (n_rows, dims) = (view.nrows(), view.ncols());
        let k = keys.as_slice()?;
        Ok(Self {
            inner: PackedIndex::from_rows(k, n_rows, dims),
        })
    }

    /// Top-`k` cosine of `query` against every indexed row.
    fn topk(&self, query: PyReadonlyArray1<i8>, top_k: usize) -> PyResult<Vec<(usize, f32)>> {
        Ok(self.inner.topk(query.as_slice()?, top_k))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
pub fn _kohaku_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHyperVector>()?;
    m.add_class::<PyEpisodicMemory>()?;
    m.add_class::<PyPackedIndex>()?;
    m.add_function(wrap_pyfunction!(py_cosine_topk, m)?)?;
    Ok(())
}
