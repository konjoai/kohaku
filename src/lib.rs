//! kohaku HDC core.
//!
//! `clippy::pedantic` is armed as a blocking gate (see `.github/workflows/
//! konjo-gate.yml`). The lints below are the curated exceptions — allowed
//! repo-wide with rationale; every other pedantic lint is enforced.
// Getters/builders here are trivially consumed at call sites; `#[must_use]` on
// each would be noise, not signal.
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
// HDC numeric kernels: dims, Hamming distances and popcounts are small bounded
// integers, and f32 is the deliberate output type for cosine similarity. These
// casts are intentional and lossless in practice.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
// PyO3 extractors (`PyReadonlyArray`, etc.) are taken by value by design — the
// FFI boundary owns them; borrowing is not an option.
#![allow(clippy::needless_pass_by_value)]
// PyO3 wrappers return `PyResult` whose error is the standard Python exception
// surface; per-method `# Errors` prose would be boilerplate.
#![allow(clippy::missing_errors_doc)]
// Flags product names (NumPy, PyO3, BLAS) in prose as "missing backticks" —
// noise, not signal.
#![allow(clippy::doc_markdown)]

pub mod accel;
pub mod hypervector;
pub mod memory;
pub mod retrieval;

#[cfg(feature = "python")]
pub mod pybindings;

pub use hypervector::{HyperVector, DIMS};
pub use memory::{EpisodicMemory, MemoryEntry};
pub use retrieval::RetrievalResult;
