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
    let mut scored: Vec<(usize, f32)> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| (i, cosine_from_bits(&q, &pack(k), dims)))
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(top_k);
    scored
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
        let out = cosine_topk(&v, &[v.clone()], 1);
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
        let out = cosine_topk(&q, &[q.clone()], 1);
        approx(out[0].1, 1.0);
    }
}
