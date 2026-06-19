//! Bit-packed cosine top-k acceleration for bipolar hypervectors.
//!
//! For strictly bipolar (+1/-1) vectors of dimension `D`, `|v|² = D`, so
//! `cosine(a, b) = dot(a, b) / D`, and `dot = D - 2·hamming(a, b)`. Packing
//! each component to one bit (set when `+1`) turns the dot product into a
//! single XOR + popcount per 64 components — far less work and memory than a
//! float multiply-accumulate. This is the genuine win of the Rust core: the
//! pure-Python path computes the same value via NumPy as the correctness
//! baseline.

/// Pack a bipolar (`+1`/`-1`) slice into `u64` words: bit set when component `> 0`.
fn pack(v: &[i8]) -> Vec<u64> {
    let words = v.len().div_ceil(64);
    let mut bits = vec![0u64; words];
    for (i, &c) in v.iter().enumerate() {
        if c > 0 {
            bits[i / 64] |= 1u64 << (i % 64);
        }
    }
    bits
}

/// Cosine similarity of two equal-length bipolar vectors via popcount Hamming.
///
/// `cosine = 1 - 2·hamming / dims`. Trailing padding bits in the final word are
/// zero in both operands, so they never contribute to the Hamming distance.
fn cosine_from_bits(a: &[u64], b: &[u64], dims: usize) -> f32 {
    if dims == 0 {
        return 0.0;
    }
    let ham: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    1.0 - 2.0 * (ham as f32) / (dims as f32)
}

/// Rank `(index, similarity)` pairs and keep the top `top_k`.
///
/// Sort is by similarity descending, ties broken by ascending index — a stable
/// ordering shared by every accel entry point so they agree bit-for-bit with the
/// pure-Python path.
fn rank(mut scored: Vec<(usize, f32)>, top_k: usize) -> Vec<(usize, f32)> {
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(top_k);
    scored
}

/// Top-`k` cosine over a set of bipolar key vectors.
///
/// Returns `(index, similarity)` pairs sorted by similarity descending, with
/// ties broken by ascending index (stable, matching the pure-Python path).
/// Keys shorter or longer than `query` are compared over the packed words they
/// share; callers pass equal-dimension vectors.
pub fn cosine_topk(query: &[i8], keys: &[Vec<i8>], top_k: usize) -> Vec<(usize, f32)> {
    if top_k == 0 || keys.is_empty() {
        return Vec::new();
    }
    let dims = query.len();
    let q = pack(query);
    let scored: Vec<(usize, f32)> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| (i, cosine_from_bits(&q, &pack(k), dims)))
        .collect();
    rank(scored, top_k)
}

/// Top-`k` cosine over a row-major `(n_rows, dims)` bipolar buffer (one-shot).
///
/// Equivalent to building a [`PackedIndex`] and querying it once, but without
/// retaining it. This is the zero-copy batch FFI path: callers hand over a
/// contiguous NumPy `int8` view, so no per-element Python marshaling occurs.
pub fn cosine_topk_rows(
    query: &[i8],
    keys: &[i8],
    n_rows: usize,
    dims: usize,
    top_k: usize,
) -> Vec<(usize, f32)> {
    PackedIndex::from_rows(keys, n_rows, dims).topk(query, top_k)
}

/// A resident bit-packed index over a fixed set of bipolar key vectors.
///
/// Keys are packed to one bit per component *once* at construction; each query
/// then costs a single query-pack plus `n_rows · words` XOR+popcount words. This
/// is the structure that lets Rust beat NumPy on repeated retrieval: BLAS must
/// stream all `n_rows · dims` floats every call, while the packed index touches
/// `n_rows · dims / 64` words and only the query crosses the FFI boundary.
pub struct PackedIndex {
    bits: Vec<u64>, // row-major, `words` u64s per row
    n_rows: usize,
    dims: usize,
    words: usize,
}

impl PackedIndex {
    /// Build from a row-major `(n_rows, dims)` bipolar buffer.
    ///
    /// `keys` must hold exactly `n_rows * dims` components; the bit for component
    /// `c` is set when `c > 0`. Trailing padding bits in each row's final word
    /// stay zero in every row, so they never contribute to a Hamming distance.
    pub fn from_rows(keys: &[i8], n_rows: usize, dims: usize) -> Self {
        let words = dims.div_ceil(64);
        let mut bits = vec![0u64; n_rows.saturating_mul(words)];
        for r in 0..n_rows {
            let src = &keys[r * dims..r * dims + dims];
            let dst = &mut bits[r * words..r * words + words];
            for (i, &c) in src.iter().enumerate() {
                if c > 0 {
                    dst[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        Self {
            bits,
            n_rows,
            dims,
            words,
        }
    }

    /// Number of indexed rows.
    pub fn len(&self) -> usize {
        self.n_rows
    }

    /// Whether the index holds no rows.
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    /// Full symmetric cosine matrix over every pair of indexed rows.
    ///
    /// Returns a flat row-major `n_rows × n_rows` buffer (length `n_rows²`); the
    /// caller reshapes to `(n, n)`. Only the upper triangle is computed — the
    /// expensive popcount work — then mirrored, with the diagonal set to `1.0`
    /// (a row is identical to itself). This collapses an all-pairs scan from
    /// `n` separate [`PackedIndex::topk`] calls (each an `O(n log n)` sort plus
    /// an FFI crossing) into a single kernel pass of `n(n-1)/2` popcounts.
    ///
    /// When `dims == 0` every entry is `0.0`, matching [`cosine_from_bits`] on
    /// empty vectors so the two paths agree at the degenerate boundary.
    pub fn all_pairs(&self) -> Vec<f32> {
        let n = self.n_rows;
        let mut out = vec![0.0f32; n * n];
        if self.dims == 0 {
            return out;
        }
        for i in 0..n {
            out[i * n + i] = 1.0;
            let ri = &self.bits[i * self.words..i * self.words + self.words];
            for j in (i + 1)..n {
                let rj = &self.bits[j * self.words..j * self.words + self.words];
                let s = cosine_from_bits(ri, rj, self.dims);
                out[i * n + j] = s;
                out[j * n + i] = s;
            }
        }
        out
    }

    /// Top-`k` cosine of `query` against every packed row.
    pub fn topk(&self, query: &[i8], top_k: usize) -> Vec<(usize, f32)> {
        if top_k == 0 || self.n_rows == 0 || self.dims == 0 {
            return Vec::new();
        }
        let q = pack(query);
        let scored: Vec<(usize, f32)> = (0..self.n_rows)
            .map(|r| {
                let row = &self.bits[r * self.words..r * self.words + self.words];
                (r, cosine_from_bits(&q, row, self.dims))
            })
            .collect();
        rank(scored, top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }

    #[test]
    fn identical_vectors_have_cosine_one() {
        let v = vec![1i8, -1, 1, 1, -1];
        let out = cosine_topk(&v, std::slice::from_ref(&v), 1);
        approx(out[0].1, 1.0);
    }

    #[test]
    fn opposite_vectors_have_cosine_minus_one() {
        let v = vec![1i8, -1, 1, -1];
        let opp = vec![-1i8, 1, -1, 1];
        let out = cosine_topk(&v, &[opp], 1);
        approx(out[0].1, -1.0);
    }

    #[test]
    fn half_matching_is_zero() {
        let v = vec![1i8, 1, 1, 1];
        let half = vec![1i8, 1, -1, -1];
        let out = cosine_topk(&v, &[half], 1);
        approx(out[0].1, 0.0);
    }

    #[test]
    fn ranks_descending_with_index_tiebreak() {
        let q = vec![1i8, 1, 1, 1];
        let keys = vec![
            vec![1i8, 1, 1, 1],     // 0: cos 1.0
            vec![-1i8, -1, -1, -1], // 1: cos -1.0
            vec![1i8, 1, 1, 1],     // 2: cos 1.0 (tie with 0 → 0 first)
        ];
        let out = cosine_topk(&q, &keys, 3);
        assert_eq!(out[0].0, 0);
        assert_eq!(out[1].0, 2);
        assert_eq!(out[2].0, 1);
    }

    #[test]
    fn top_k_truncates_and_zero_k_empty() {
        let q = vec![1i8, -1, 1];
        let keys = vec![vec![1i8, -1, 1], vec![-1i8, 1, -1]];
        assert_eq!(cosine_topk(&q, &keys, 1).len(), 1);
        assert!(cosine_topk(&q, &keys, 0).is_empty());
    }

    #[test]
    fn handles_dims_beyond_one_word() {
        let q: Vec<i8> = (0..130).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let out = cosine_topk(&q, std::slice::from_ref(&q), 1);
        approx(out[0].1, 1.0);
    }

    #[test]
    fn rows_path_matches_vec_path() {
        let q = vec![1i8, 1, 1, 1];
        let rows = vec![
            vec![1i8, 1, 1, 1],
            vec![-1i8, -1, -1, -1],
            vec![1i8, 1, -1, -1],
            vec![1i8, 1, 1, 1],
        ];
        let flat: Vec<i8> = rows.iter().flatten().copied().collect();
        let via_vec = cosine_topk(&q, &rows, 4);
        let via_rows = cosine_topk_rows(&q, &flat, 4, 4, 4);
        assert_eq!(via_vec, via_rows);
    }

    #[test]
    fn rows_path_empty_when_zero_k_or_no_rows() {
        let q = vec![1i8, -1, 1];
        let flat = vec![1i8, -1, 1];
        assert!(cosine_topk_rows(&q, &flat, 1, 3, 0).is_empty());
        assert!(cosine_topk_rows(&q, &[], 0, 3, 5).is_empty());
    }

    #[test]
    fn packed_index_reports_len_and_emptiness() {
        let keys = vec![1i8, -1, 1, -1, 1, -1]; // 2 rows of dims 3
        let idx = PackedIndex::from_rows(&keys, 2, 3);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
        assert!(PackedIndex::from_rows(&[], 0, 3).is_empty());
    }

    #[test]
    fn all_pairs_is_symmetric_with_unit_diagonal() {
        let dims = 130; // spans 3 words
        let mut flat: Vec<i8> = Vec::new();
        for r in 0..5 {
            for i in 0..dims {
                flat.push(if (i + r) % 3 == 0 { 1 } else { -1 });
            }
        }
        let idx = PackedIndex::from_rows(&flat, 5, dims);
        let mat = idx.all_pairs();
        assert_eq!(mat.len(), 25);
        for i in 0..5 {
            approx(mat[i * 5 + i], 1.0); // diagonal: row vs itself
            for j in 0..5 {
                approx(mat[i * 5 + j], mat[j * 5 + i]); // symmetric
            }
        }
    }

    #[test]
    fn all_pairs_row_matches_topk_scores() {
        // Every off-diagonal entry must equal the cosine the per-row topk path
        // reports — the parity guarantee the Python all_scores fallback relies on.
        let dims = 96;
        let mut flat: Vec<i8> = Vec::new();
        for r in 0..6 {
            for i in 0..dims {
                flat.push(if (i * 7 + r) % 5 < 2 { 1 } else { -1 });
            }
        }
        let idx = PackedIndex::from_rows(&flat, 6, dims);
        let mat = idx.all_pairs();
        for i in 0..6 {
            let row: Vec<i8> = flat[i * dims..i * dims + dims].to_vec();
            for (j, sim) in idx.topk(&row, 6) {
                approx(mat[i * 6 + j], sim);
            }
        }
    }

    #[test]
    fn all_pairs_empty_when_no_dims() {
        let idx = PackedIndex::from_rows(&[], 0, 0);
        assert!(idx.all_pairs().is_empty());
    }

    #[test]
    fn packed_index_topk_matches_one_shot_rows() {
        let dims = 130; // spans 3 words
        let q: Vec<i8> = (0..dims).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let mut flat: Vec<i8> = Vec::new();
        for r in 0..5 {
            for i in 0..dims {
                flat.push(if (i + r) % 3 == 0 { 1 } else { -1 });
            }
        }
        let idx = PackedIndex::from_rows(&flat, 5, dims);
        assert_eq!(idx.topk(&q, 3), cosine_topk_rows(&q, &flat, 5, dims, 3));
    }
}
