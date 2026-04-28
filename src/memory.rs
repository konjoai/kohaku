use crate::hypervector::HyperVector;
use serde::{Deserialize, Serialize};

/// A single episodic memory entry binding a key hypervector to a value hypervector,
/// with a human-readable label and a monotonic timestamp for FIFO eviction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique monotonic identifier assigned at store time.
    pub id: u64,
    /// Query key — the associative handle used for retrieval.
    pub key: HyperVector,
    /// Stored value — the payload retrieved when the key matches.
    pub value: HyperVector,
    /// Human-readable tag for debugging and display.
    pub label: String,
    /// Logical timestamp for ordering and FIFO eviction (increments per store).
    pub timestamp: u64,
}

/// A fixed-capacity episodic memory store.
///
/// Internally an ordered `Vec<MemoryEntry>`. When the store exceeds `capacity`,
/// the oldest entry (smallest `timestamp`, front of the vec) is evicted (FIFO).
///
/// Retrieval is O(n) linear scan — appropriate for episodic memory sizes up to
/// tens of thousands of entries. For larger-scale approximate search, a future
/// phase will add an HNSW index layer.
pub struct EpisodicMemory {
    entries: Vec<MemoryEntry>,
    next_id: u64,
    capacity: usize,
}

impl EpisodicMemory {
    /// Create a new episodic memory with the given maximum capacity.
    ///
    /// # Panics
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "EpisodicMemory capacity must be > 0");
        Self {
            entries: Vec::with_capacity(capacity.min(1024)),
            next_id: 1,
            capacity,
        }
    }

    /// Store a key-value pair with a label.
    ///
    /// If at capacity, the oldest entry is evicted before insertion (FIFO).
    /// Returns the assigned entry id.
    pub fn store(&mut self, key: HyperVector, value: HyperVector, label: String) -> u64 {
        let id = self.next_id;
        let timestamp = self.next_id;
        self.next_id += 1;

        if self.entries.len() == self.capacity {
            // Evict the front (oldest) entry — Vec is kept insertion-ordered.
            self.entries.remove(0);
        }

        self.entries.push(MemoryEntry {
            id,
            key,
            value,
            label,
            timestamp,
        });
        id
    }

    /// Number of entries currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no entries are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Read-only view of all stored entries.
    #[inline]
    pub fn entries(&self) -> &[MemoryEntry] {
        &self.entries
    }

    /// Remove all entries. Resets the id counter.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_id = 1;
    }

    /// Maximum number of entries this memory can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypervector::{HyperVector, DIMS};

    fn make_entry(seed: u64) -> (HyperVector, HyperVector) {
        let key = HyperVector::random(DIMS, seed);
        let val = HyperVector::random(DIMS, seed + 1000);
        (key, val)
    }

    #[test]
    fn test_store_and_len() {
        let mut mem = EpisodicMemory::new(10);
        assert!(mem.is_empty());
        let (k, v) = make_entry(1);
        mem.store(k, v, "first".to_string());
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn test_fifo_eviction() {
        let mut mem = EpisodicMemory::new(3);
        for i in 0..3 {
            let (k, v) = make_entry(i as u64);
            mem.store(k, v, format!("entry-{i}"));
        }
        assert_eq!(mem.len(), 3);

        // Store a 4th — evicts entry-0
        let (k, v) = make_entry(99);
        mem.store(k, v, "entry-3".to_string());
        assert_eq!(mem.len(), 3);

        // The oldest entry (label "entry-0") must be gone
        let labels: Vec<&str> = mem.entries().iter().map(|e| e.label.as_str()).collect();
        assert!(!labels.contains(&"entry-0"), "entry-0 should have been evicted");
        assert!(labels.contains(&"entry-1"));
        assert!(labels.contains(&"entry-2"));
        assert!(labels.contains(&"entry-3"));
    }

    #[test]
    fn test_clear_resets() {
        let mut mem = EpisodicMemory::new(10);
        let (k, v) = make_entry(42);
        mem.store(k, v, "x".to_string());
        mem.clear();
        assert!(mem.is_empty());
        // id counter resets — next store gets id 1
        let (k2, v2) = make_entry(43);
        let id = mem.store(k2, v2, "y".to_string());
        assert_eq!(id, 1);
    }

    #[test]
    fn test_ids_are_unique_and_monotonic() {
        let mut mem = EpisodicMemory::new(100);
        let mut ids = Vec::new();
        for i in 0..10 {
            let (k, v) = make_entry(i as u64);
            ids.push(mem.store(k, v, format!("e{i}")));
        }
        for w in ids.windows(2) {
            assert!(w[1] > w[0], "ids must be strictly increasing");
        }
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity_panics() {
        let _ = EpisodicMemory::new(0);
    }
}
