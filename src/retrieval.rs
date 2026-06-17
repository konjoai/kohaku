use crate::{EpisodicMemory, HyperVector};

/// A single retrieval result returned from associative memory query.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// The id of the matched memory entry.
    pub entry_id: u64,
    /// Human-readable label of the matched entry.
    pub label: String,
    /// Cosine similarity of the query key to this entry's stored key.
    pub similarity: f32,
    /// Clone of the value hypervector stored under the matched key.
    pub value: HyperVector,
}

/// Query the episodic memory for the `top_k` most similar entries to `query_key`.
///
/// Computes cosine similarity between `query_key` and every stored entry's key,
/// then returns up to `top_k` results in descending similarity order.
///
/// Returns fewer than `top_k` results if the memory holds fewer entries.
pub fn query(
    memory: &EpisodicMemory,
    query_key: &HyperVector,
    top_k: usize,
) -> Vec<RetrievalResult> {
    if top_k == 0 || memory.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<RetrievalResult> = memory
        .entries()
        .iter()
        .map(|entry| RetrievalResult {
            entry_id: entry.id,
            label: entry.label.clone(),
            similarity: query_key.cosine_similarity(&entry.key),
            value: entry.value.clone(),
        })
        .collect();

    // Sort descending by similarity; break ties by entry_id ascending (stable).
    scored.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.entry_id.cmp(&b.entry_id))
    });

    scored.truncate(top_k);
    scored
}

/// Query the episodic memory for all entries whose key similarity to `query_key`
/// is at or above `threshold`.
///
/// Results are returned in descending similarity order.
pub fn query_threshold(
    memory: &EpisodicMemory,
    query_key: &HyperVector,
    threshold: f32,
) -> Vec<RetrievalResult> {
    let mut results: Vec<RetrievalResult> = memory
        .entries()
        .iter()
        .filter_map(|entry| {
            let sim = query_key.cosine_similarity(&entry.key);
            if sim >= threshold {
                Some(RetrievalResult {
                    entry_id: entry.id,
                    label: entry.label.clone(),
                    similarity: sim,
                    value: entry.value.clone(),
                })
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.entry_id.cmp(&b.entry_id))
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodicMemory, HyperVector, DIMS};

    fn populate_memory(capacity: usize, count: usize) -> (EpisodicMemory, Vec<HyperVector>) {
        let mut mem = EpisodicMemory::new(capacity);
        let mut keys = Vec::new();
        for i in 0..count {
            let key = HyperVector::random(DIMS, i as u64 * 7 + 1);
            let val = HyperVector::random(DIMS, i as u64 * 7 + 2);
            let label = format!("item-{i}");
            keys.push(key.clone());
            mem.store(key, val, label);
        }
        (mem, keys)
    }

    #[test]
    fn test_exact_key_top1() {
        let (mem, keys) = populate_memory(20, 10);
        // Query with the exact key for item-3 (index 3)
        let results = query(&mem, &keys[3], 1);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].similarity > 0.99,
            "exact key must have similarity ≈ 1.0"
        );
        assert_eq!(results[0].label, "item-3");
    }

    #[test]
    fn test_top_k_count_and_ordering() {
        let (mem, keys) = populate_memory(20, 10);
        let results = query(&mem, &keys[0], 5);
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(
                w[0].similarity >= w[1].similarity,
                "results must be sorted descending"
            );
        }
    }

    #[test]
    fn test_top_k_larger_than_memory() {
        let (mem, keys) = populate_memory(5, 5);
        let results = query(&mem, &keys[0], 100);
        assert_eq!(
            results.len(),
            5,
            "cannot return more results than memory size"
        );
    }

    #[test]
    fn test_empty_memory_returns_empty() {
        let mem = EpisodicMemory::new(10);
        let q = HyperVector::random(DIMS, 42);
        assert!(query(&mem, &q, 5).is_empty());
        assert!(query_threshold(&mem, &q, 0.5).is_empty());
    }

    #[test]
    fn test_threshold_filters_correctly() {
        let (mem, keys) = populate_memory(20, 10);
        // Query with the exact key for item-2: similarity=1.0 for itself, ~0 for others
        let results = query_threshold(&mem, &keys[2], 0.9);
        assert!(!results.is_empty());
        for r in &results {
            assert!(
                r.similarity >= 0.9,
                "all results must meet threshold, got {}",
                r.similarity
            );
        }
        // The exact match must be present
        assert!(
            results.iter().any(|r| r.label == "item-2"),
            "exact match must pass threshold"
        );
    }

    #[test]
    fn test_zero_top_k_returns_empty() {
        let (mem, keys) = populate_memory(5, 5);
        let results = query(&mem, &keys[0], 0);
        assert!(results.is_empty());
    }
}
