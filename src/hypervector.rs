use std::fmt;
use serde::{Deserialize, Serialize};

/// Default dimensionality for hypervectors. 10,000-dimensional bipolar vectors
/// provide near-zero expected cosine similarity (~N(0, 1/sqrt(D)) ≈ 0.01) between
/// random pairs, enabling high-capacity associative memory.
pub const DIMS: usize = 10_000;

/// Linear Congruential Generator — Knuth multiplicative variant.
/// Deterministic, fast, sufficient entropy for sign-bit extraction.
#[inline(always)]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// A bipolar hypervector with components in {-1, +1} packed as i8.
///
/// HDC arithmetic exploits three key operations:
/// - **Bundle** (superposition): sum + sign → result is similar to all inputs
/// - **Bind** (multiplication): element-wise multiply → result dissimilar to both inputs
/// - **Permute** (rotation): circular shift → encodes sequence/positional order
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperVector {
    pub data: Vec<i8>,
}

impl HyperVector {
    /// Generate a deterministic random bipolar vector from a u64 seed.
    /// Uses LCG to produce bits; sign of the high bit determines +1/-1.
    pub fn random(dims: usize, seed: u64) -> Self {
        let mut state = seed ^ 0xDEAD_BEEF_CAFE_BABE;
        let data: Vec<i8> = (0..dims)
            .map(|_| {
                let v = lcg_next(&mut state);
                // Use the high bit as the sign: effectively uniform Bernoulli(0.5)
                if v >> 63 == 0 { 1_i8 } else { -1_i8 }
            })
            .collect();
        Self { data }
    }

    /// Bundle (superposition) of a slice of hypervectors via majority vote.
    ///
    /// Each output component is sign(Σ inputs[i][j]).
    /// Ties (even-count inputs) resolve to +1.
    /// The resulting vector is similar (positive cosine) to every input.
    ///
    /// # Panics
    /// Panics if `vectors` is empty or vectors have differing lengths.
    pub fn bundle(vectors: &[&HyperVector]) -> Self {
        assert!(!vectors.is_empty(), "bundle requires at least one vector");
        let dims = vectors[0].data.len();
        for v in vectors.iter() {
            assert_eq!(v.data.len(), dims, "all vectors must have equal dimensionality");
        }

        let mut sums: Vec<i32> = vec![0; dims];
        for hv in vectors {
            for (s, &x) in sums.iter_mut().zip(hv.data.iter()) {
                *s += x as i32;
            }
        }
        let data: Vec<i8> = sums.iter().map(|&s| if s >= 0 { 1 } else { -1 }).collect();
        Self { data }
    }

    /// Bind two hypervectors via element-wise multiplication.
    ///
    /// For bipolar vectors, multiply ≡ XNOR on sign bits.
    /// The result is dissimilar to both operands, but binding again with
    /// either operand recovers the other: `(a ⊗ b) ⊗ a ≈ b`.
    ///
    /// # Panics
    /// Panics if dimensions differ.
    pub fn bind(&self, other: &HyperVector) -> Self {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "bind requires equal dimensionality"
        );
        let data: Vec<i8> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Self { data }
    }

    /// Permute (circular left shift) by `shift` positions.
    ///
    /// Encodes positional/sequential order: `permute(hv, n)` is orthogonal
    /// to `permute(hv, m)` for n ≠ m. Used to build sequence-aware structures.
    pub fn permute(&self, shift: usize) -> Self {
        let d = self.data.len();
        if d == 0 {
            return self.clone();
        }
        let shift = shift % d;
        // Circular left rotation: destination slot i receives the element from
        // position (i + shift) % d of the original.
        let data: Vec<i8> = (0..d).map(|i| self.data[(i + shift) % d]).collect();
        Self { data }
    }

    /// Cosine similarity ∈ [-1.0, 1.0].
    ///
    /// For bipolar vectors, dot product / (|a| * |b|).
    /// Two random vectors have expected similarity ≈ 0 (std ≈ 1/sqrt(D)).
    pub fn cosine_similarity(&self, other: &HyperVector) -> f32 {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "cosine_similarity requires equal dimensionality"
        );
        let dot: i32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a as i32) * (b as i32))
            .sum();
        // For bipolar ±1 vectors, |v|² = D, so |v| = sqrt(D).
        // Cosine = dot / (sqrt(D) * sqrt(D)) = dot / D.
        let d = self.data.len() as f32;
        (dot as f32) / d
    }

    /// Hamming distance as a fraction of differing components ∈ [0.0, 1.0].
    ///
    /// For bipolar vectors, a component differs iff the product is -1.
    /// Relation to cosine: hamming ≈ (1 - cosine) / 2.
    pub fn hamming_distance(&self, other: &HyperVector) -> f32 {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "hamming_distance requires equal dimensionality"
        );
        let differ: usize = self
            .data
            .iter()
            .zip(other.data.iter())
            .filter(|(&a, &b)| a != b)
            .count();
        differ as f32 / self.data.len() as f32
    }

    /// Dimensionality of this hypervector.
    #[inline]
    pub fn dims(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Display for HyperVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HyperVector(d={}, [", self.data.len())?;
        let preview = self.data.iter().take(8);
        for (i, &v) in preview.enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:+}", v)?;
        }
        if self.data.len() > 8 {
            write!(f, ", ...")?;
        }
        write!(f, "])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_random_bipolar() {
        let hv = HyperVector::random(DIMS, 42);
        assert_eq!(hv.data.len(), DIMS);
        for &v in &hv.data {
            assert!(v == 1 || v == -1, "all components must be ±1");
        }
    }

    #[test]
    fn test_random_deterministic() {
        let a = HyperVector::random(DIMS, 99);
        let b = HyperVector::random(DIMS, 99);
        assert_eq!(a.data, b.data, "same seed must produce identical vector");
    }

    #[test]
    fn test_random_distinct_seeds() {
        let a = HyperVector::random(DIMS, 1);
        let b = HyperVector::random(DIMS, 2);
        // Probability of identical random ±1 vectors ≈ 2^-10000
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn test_self_similarity_is_one() {
        let hv = HyperVector::random(DIMS, 7);
        assert_abs_diff_eq!(hv.cosine_similarity(&hv), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_random_pair_near_orthogonal() {
        let a = HyperVector::random(DIMS, 100);
        let b = HyperVector::random(DIMS, 200);
        let sim = a.cosine_similarity(&b).abs();
        // Expected |sim| ~ 1/sqrt(10000) = 0.01; allow up to 0.05 (5-sigma)
        assert!(sim < 0.05, "random vectors should be near-orthogonal, got {sim}");
    }

    #[test]
    fn test_bundle_similar_to_inputs() {
        let a = HyperVector::random(DIMS, 1);
        let b = HyperVector::random(DIMS, 2);
        let c = HyperVector::random(DIMS, 3);
        let bundle = HyperVector::bundle(&[&a, &b, &c]);
        // Bundle must be more similar to each component than random chance
        let sim_a = bundle.cosine_similarity(&a);
        let sim_b = bundle.cosine_similarity(&b);
        let sim_c = bundle.cosine_similarity(&c);
        assert!(sim_a > 0.3, "bundle should be similar to a, got {sim_a}");
        assert!(sim_b > 0.3, "bundle should be similar to b, got {sim_b}");
        assert!(sim_c > 0.3, "bundle should be similar to c, got {sim_c}");
    }

    #[test]
    fn test_bind_dissimilar_to_inputs() {
        let a = HyperVector::random(DIMS, 10);
        let b = HyperVector::random(DIMS, 20);
        let bound = a.bind(&b);
        let sim_a = bound.cosine_similarity(&a).abs();
        let sim_b = bound.cosine_similarity(&b).abs();
        assert!(sim_a < 0.05, "bound should be dissimilar to a, got {sim_a}");
        assert!(sim_b < 0.05, "bound should be dissimilar to b, got {sim_b}");
    }

    #[test]
    fn test_bind_invertible() {
        let a = HyperVector::random(DIMS, 10);
        let b = HyperVector::random(DIMS, 20);
        let bound = a.bind(&b);
        // Unbind: (a ⊗ b) ⊗ a = b (for bipolar ±1)
        let recovered = bound.bind(&a);
        assert_abs_diff_eq!(recovered.cosine_similarity(&b), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_permute_changes_vector() {
        let hv = HyperVector::random(DIMS, 5);
        let perm = hv.permute(1);
        let sim = hv.cosine_similarity(&perm).abs();
        assert!(sim < 0.05, "permuted vector should differ from original, got {sim}");
    }

    #[test]
    fn test_permute_invertible() {
        let hv = HyperVector::random(DIMS, 5);
        let perm = hv.permute(137);
        // Rotate back: shift by D-137
        let back = perm.permute(DIMS - 137);
        assert_abs_diff_eq!(hv.cosine_similarity(&back), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hamming_self_zero() {
        let hv = HyperVector::random(DIMS, 3);
        assert_abs_diff_eq!(hv.hamming_distance(&hv), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hamming_cosine_relationship() {
        let a = HyperVector::random(DIMS, 111);
        let b = HyperVector::random(DIMS, 222);
        let h = a.hamming_distance(&b);
        let c = a.cosine_similarity(&b);
        // For bipolar ±1: hamming = (1 - cosine) / 2
        assert_abs_diff_eq!(h, (1.0 - c) / 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_display() {
        let hv = HyperVector::random(16, 42);
        let s = format!("{hv}");
        assert!(s.contains("HyperVector(d=16"));
        assert!(s.contains("..."));
    }
}
