use approx::assert_abs_diff_eq;
use kohaku::retrieval::{query, query_threshold};
use kohaku::{EpisodicMemory, HyperVector, DIMS};

// ─── Test 1: Random vectors are approximately orthogonal ─────────────────────

#[test]
fn test_random_vectors_near_orthogonal() {
    // 10,000-dim bipolar vectors: expected |cosine| ~ 1/sqrt(10000) = 0.01
    // 5-sigma tolerance: 0.01 * 5 = 0.05
    let a = HyperVector::random(DIMS, 0xABCD_1234_0000_0001);
    let b = HyperVector::random(DIMS, 0xABCD_1234_0000_0002);
    let sim = a.cosine_similarity(&b).abs();
    assert!(
        sim < 0.05,
        "random D={DIMS} vectors must be near-orthogonal; got |sim|={sim:.4}"
    );
}

// ─── Test 2: Identical vectors have similarity 1.0 ───────────────────────────

#[test]
fn test_identical_vectors_similarity_one() {
    let v = HyperVector::random(DIMS, 0xDEAD_BEEF);
    let sim = v.cosine_similarity(&v);
    assert_abs_diff_eq!(sim, 1.0_f32, epsilon = 1e-6);
}

// ─── Test 3: Bundled vector is similar to its components ─────────────────────

#[test]
fn test_bundle_similar_to_components() {
    let a = HyperVector::random(DIMS, 11);
    let b = HyperVector::random(DIMS, 22);
    let c = HyperVector::random(DIMS, 33);

    let bundle = HyperVector::bundle(&[&a, &b, &c]);

    // With 3 random components, bundle should have cosine > 0.5 with each
    // (majority vote keeps ~2/3 of each component's bits)
    let sim_a = bundle.cosine_similarity(&a);
    let sim_b = bundle.cosine_similarity(&b);
    let sim_c = bundle.cosine_similarity(&c);

    assert!(
        sim_a > 0.3,
        "bundle must be similar to component a; got {sim_a:.4}"
    );
    assert!(
        sim_b > 0.3,
        "bundle must be similar to component b; got {sim_b:.4}"
    );
    assert!(
        sim_c > 0.3,
        "bundle must be similar to component c; got {sim_c:.4}"
    );
}

// ─── Test 4: Permuted vector differs from original ───────────────────────────

#[test]
fn test_permute_produces_distinct_vector() {
    let v = HyperVector::random(DIMS, 77);
    let perm1 = v.permute(1);
    let perm2 = v.permute(DIMS / 2);

    // A single-position shift produces a different vector
    let sim_1 = v.cosine_similarity(&perm1).abs();
    assert!(
        sim_1 < 0.05,
        "shift-1 permute should be near-orthogonal to original; got {sim_1:.4}"
    );

    // Half-rotation also near-orthogonal
    let sim_half = v.cosine_similarity(&perm2).abs();
    assert!(
        sim_half < 0.05,
        "half-rotation should be near-orthogonal to original; got {sim_half:.4}"
    );

    // Rotating back by D positions returns the exact original
    let back = perm1.permute(DIMS - 1);
    assert_abs_diff_eq!(v.cosine_similarity(&back), 1.0_f32, epsilon = 1e-6);
}

// ─── Test 5: Store and retrieve by similarity ────────────────────────────────

#[test]
fn test_store_and_retrieve_by_similarity() {
    let mut mem = EpisodicMemory::new(50);

    // Store 10 unrelated memories
    for i in 0..10_u64 {
        let key = HyperVector::random(DIMS, i * 31 + 1);
        let val = HyperVector::random(DIMS, i * 31 + 2);
        mem.store(key, val, format!("entry-{i}"));
    }

    // Query with the exact key for entry-5
    let target_seed: u64 = 5 * 31 + 1;
    let target_key = HyperVector::random(DIMS, target_seed);
    let results = query(&mem, &target_key, 3);

    assert!(!results.is_empty(), "should have at least one result");
    // Top result must be entry-5 with near-perfect similarity
    assert_eq!(results[0].label, "entry-5");
    assert!(
        results[0].similarity > 0.99,
        "exact-key query must return sim ≈ 1.0; got {:.4}",
        results[0].similarity
    );
}

// ─── Test 6: Top-k retrieval returns k results sorted by similarity ──────────

#[test]
fn test_top_k_sorted_descending() {
    let mut mem = EpisodicMemory::new(50);
    for i in 0..20_u64 {
        let key = HyperVector::random(DIMS, i * 19 + 3);
        let val = HyperVector::random(DIMS, i * 19 + 4);
        mem.store(key, val, format!("item-{i}"));
    }

    let query_key = HyperVector::random(DIMS, 7 * 19 + 3); // exact key for item-7
    let results = query(&mem, &query_key, 5);

    assert_eq!(results.len(), 5, "must return exactly k=5 results");

    // Sorted descending
    for w in results.windows(2) {
        assert!(
            w[0].similarity >= w[1].similarity,
            "results must be sorted by descending similarity: {} < {}",
            w[0].similarity,
            w[1].similarity
        );
    }

    // Top result is item-7
    assert_eq!(results[0].label, "item-7");
}

// ─── Test 7: Capacity eviction works correctly ───────────────────────────────

#[test]
fn test_capacity_fifo_eviction() {
    let capacity = 5;
    let mut mem = EpisodicMemory::new(capacity);

    // Fill to capacity
    for i in 0..capacity as u64 {
        let key = HyperVector::random(DIMS, i * 41 + 1);
        let val = HyperVector::random(DIMS, i * 41 + 2);
        mem.store(key, val, format!("old-{i}"));
    }
    assert_eq!(mem.len(), capacity);

    // Add one more — should evict "old-0"
    let new_key = HyperVector::random(DIMS, 9999);
    let new_val = HyperVector::random(DIMS, 9998);
    mem.store(new_key, new_val, "new".to_string());

    assert_eq!(
        mem.len(),
        capacity,
        "memory must not exceed capacity after eviction"
    );

    let labels: Vec<&str> = mem.entries().iter().map(|e| e.label.as_str()).collect();
    assert!(
        !labels.contains(&"old-0"),
        "oldest entry 'old-0' must have been evicted; labels: {labels:?}"
    );
    assert!(
        labels.contains(&"new"),
        "newest entry 'new' must be present"
    );

    // Remaining old entries must still be present
    for i in 1..capacity as u64 {
        let expected = format!("old-{i}");
        assert!(
            labels.contains(&expected.as_str()),
            "entry '{expected}' must still be in memory"
        );
    }
}

// ─── Test 8: Bound vectors retrieve correctly when unbound with same key ──────

#[test]
fn test_bind_encode_retrieve() {
    // HDC associative memory pattern:
    //   Store: memory += key ⊗ value
    //   Retrieve: key ⊗ memory ≈ value
    //
    // Here we test the simpler form: storing (key ⊗ value) as the memory key
    // and binding back with the original key to recover the value.

    let key = HyperVector::random(DIMS, 0x1111_2222_3333_4444);
    let value = HyperVector::random(DIMS, 0x5555_6666_7777_8888);

    // Encode: stored_key = key ⊗ value
    let stored_key = key.bind(&value);

    // The stored_key is dissimilar to both key and value (HDC bind property)
    let sim_to_key = stored_key.cosine_similarity(&key).abs();
    let sim_to_val = stored_key.cosine_similarity(&value).abs();
    assert!(
        sim_to_key < 0.05,
        "bound vector should be dissimilar to key; got {sim_to_key:.4}"
    );
    assert!(
        sim_to_val < 0.05,
        "bound vector should be dissimilar to value; got {sim_to_val:.4}"
    );

    // Unbind: recovered_value = stored_key ⊗ key = (key ⊗ value) ⊗ key = value
    let recovered = stored_key.bind(&key);
    let recovery_sim = recovered.cosine_similarity(&value);
    assert_abs_diff_eq!(recovery_sim, 1.0_f32, epsilon = 1e-6);

    // Now store the bound vector and retrieve using the query key
    let mut mem = EpisodicMemory::new(10);
    mem.store(stored_key.clone(), value.clone(), "bound-entry".to_string());

    // To query: bind the query key and look up
    // The user would query with `stored_key` directly; we verify the round-trip
    let results = query(&mem, &stored_key, 1);
    assert_eq!(results.len(), 1);
    assert_abs_diff_eq!(results[0].similarity, 1.0_f32, epsilon = 1e-6);

    // The retrieved value must match the original
    let retrieved_sim = results[0].value.cosine_similarity(&value);
    assert_abs_diff_eq!(retrieved_sim, 1.0_f32, epsilon = 1e-6);
}

// ─── Bonus: threshold query returns only high-similarity matches ──────────────

#[test]
fn test_threshold_query_precision() {
    let mut mem = EpisodicMemory::new(50);
    for i in 0..20_u64 {
        let key = HyperVector::random(DIMS, i * 53 + 7);
        let val = HyperVector::random(DIMS, i * 53 + 8);
        mem.store(key, val, format!("t-{i}"));
    }

    // Use exact key for t-9
    let exact_key = HyperVector::random(DIMS, 9 * 53 + 7);
    let results = query_threshold(&mem, &exact_key, 0.95);

    // Must contain exactly the exact match (all others are ~orthogonal)
    assert!(!results.is_empty(), "exact key must pass threshold 0.95");
    for r in &results {
        assert!(
            r.similarity >= 0.95,
            "all results must meet threshold; got {}",
            r.similarity
        );
    }
    assert!(
        results.iter().any(|r| r.label == "t-9"),
        "t-9 must appear in threshold results"
    );
}
