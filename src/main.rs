use clap::{Parser, Subcommand};
use std::time::Instant;

use kohaku::{EpisodicMemory, HyperVector, DIMS};
use kohaku::retrieval::{query, query_threshold};

#[derive(Parser)]
#[command(
    name = "kohaku",
    version = "0.1.0",
    about = "Neural episodic memory via Hyperdimensional Computing",
    long_about = "Kohaku stores experiences as HDC hypervectors and retrieves them via associative similarity."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a self-contained demonstration of Kohaku episodic memory
    Demo,
    /// Benchmark store and query throughput with N random vectors
    Bench {
        /// Number of vectors to store and query
        #[arg(short, long, default_value_t = 10_000)]
        count: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Demo => run_demo(),
        Commands::Bench { count } => run_bench(count),
    }
}

// ─── ASCII table helpers ────────────────────────────────────────────────────

fn print_separator(widths: &[usize]) {
    print!("+");
    for &w in widths {
        print!("{}-+", "-".repeat(w + 1));
    }
    println!();
}

fn print_row(cells: &[&str], widths: &[usize]) {
    print!("|");
    for (&w, cell) in widths.iter().zip(cells.iter()) {
        print!(" {:<w$} |", cell);
    }
    println!();
}

fn print_table(headers: &[&str], rows: &[Vec<String>]) {
    // Compute column widths
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    print_separator(&widths);
    print_row(headers, &widths);
    print_separator(&widths);
    for row in rows {
        let cells: Vec<&str> = row.iter().map(|s| s.as_str()).collect();
        print_row(&cells, &widths);
    }
    print_separator(&widths);
}

// ─── Demo ────────────────────────────────────────────────────────────────────

fn run_demo() {
    println!();
    println!("  Kohaku — Neural Episodic Memory Demo");
    println!("  ════════════════════════════════════");
    println!();

    let mut memory = EpisodicMemory::new(100);

    // Five labeled memories: each key is deterministic from a seed
    let labels = [
        ("apple",   "A round red fruit with a crisp texture"),
        ("bicycle", "A two-wheeled human-powered vehicle"),
        ("ocean",   "A vast body of salt water covering most of Earth"),
        ("library", "A building housing collections of books"),
        ("volcano", "A rupture in Earth's crust that expels lava"),
    ];

    println!("  Storing 5 memories...");
    println!();

    let mut stored_ids: Vec<u64> = Vec::new();
    let mut key_vecs: Vec<HyperVector> = Vec::new();

    for (i, (label, _desc)) in labels.iter().enumerate() {
        let seed = (i as u64 + 1) * 0xABC1_2345;
        let key = HyperVector::random(DIMS, seed);
        let val = HyperVector::random(DIMS, seed ^ 0xDEAD_CAFE);
        key_vecs.push(key.clone());
        let id = memory.store(key, val, label.to_string());
        stored_ids.push(id);
        println!("  [id={id:>2}] Stored: \"{label}\"");
    }

    println!();
    println!("  Memory size: {}/{} entries", memory.len(), 100);
    println!();

    // Query 3 scenarios -------------------------------------------------------
    // 1. Exact key match → should retrieve the exact entry at top-1
    // 2. Bundled key (apple + bicycle) → should retrieve both near top
    // 3. Threshold query with the ocean key

    // Query 1: Exact key for "ocean" (index 2)
    println!("  Query 1 — Exact key for \"ocean\" (top-3)");
    {
        let results = query(&memory, &key_vecs[2], 3);
        let rows: Vec<Vec<String>> = results
            .iter()
            .map(|r| {
                vec![
                    r.entry_id.to_string(),
                    r.label.clone(),
                    format!("{:.6}", r.similarity),
                ]
            })
            .collect();
        print_table(&["ID", "Label", "Similarity"], &rows);
    }

    println!();

    // Query 2: Bundled key apple+bicycle → both should appear with positive similarity
    println!("  Query 2 — Bundled query key (apple ⊕ bicycle), top-5");
    {
        let bundle_key = HyperVector::bundle(&[&key_vecs[0], &key_vecs[1]]);
        let results = query(&memory, &bundle_key, 5);
        let rows: Vec<Vec<String>> = results
            .iter()
            .map(|r| {
                vec![
                    r.entry_id.to_string(),
                    r.label.clone(),
                    format!("{:.6}", r.similarity),
                ]
            })
            .collect();
        print_table(&["ID", "Label", "Similarity"], &rows);
        let top_labels: Vec<&str> = results.iter().take(2).map(|r| r.label.as_str()).collect();
        println!();
        println!("  → Top-2 labels: {:?} (expected \"apple\" and \"bicycle\" in some order)", top_labels);
    }

    println!();

    // Query 3: Threshold query on "library" key at 0.9
    println!("  Query 3 — Threshold query (sim ≥ 0.90) on \"library\" key");
    {
        let results = query_threshold(&memory, &key_vecs[3], 0.90);
        if results.is_empty() {
            println!("  No entries above threshold (unexpected — check seeds)");
        } else {
            let rows: Vec<Vec<String>> = results
                .iter()
                .map(|r| {
                    vec![
                        r.entry_id.to_string(),
                        r.label.clone(),
                        format!("{:.6}", r.similarity),
                    ]
                })
                .collect();
            print_table(&["ID", "Label", "Similarity"], &rows);
        }
    }

    println!();
    println!("  Demo complete.");
    println!();
}

// ─── Bench ───────────────────────────────────────────────────────────────────

fn run_bench(count: usize) {
    println!();
    println!("  Kohaku — Throughput Benchmark");
    println!("  ═══════════════════════════════");
    println!("  Vectors: {count}  |  Dims: {DIMS}");
    println!();

    let mut memory = EpisodicMemory::new(count.max(1));

    // Pre-generate keys outside of timing window
    let keys: Vec<HyperVector> = (0..count)
        .map(|i| HyperVector::random(DIMS, i as u64 * 13 + 7))
        .collect();
    let values: Vec<HyperVector> = (0..count)
        .map(|i| HyperVector::random(DIMS, i as u64 * 17 + 3))
        .collect();

    // ── Store throughput ────────────────────────────────────────────────────
    let t0 = Instant::now();
    for i in 0..count {
        memory.store(keys[i].clone(), values[i].clone(), format!("v{i}"));
    }
    let store_ns = t0.elapsed().as_nanos();
    let store_per_sec = count as f64 / (store_ns as f64 / 1_000_000_000.0);
    let store_us_each = store_ns as f64 / count as f64 / 1_000.0;

    // ── Query throughput (top-10 for each of 100 probe vectors) ────────────
    let probe_count = 100.min(count);
    let t1 = Instant::now();
    let mut sink = 0usize; // prevent dead-code elimination
    for key in keys.iter().take(probe_count) {
        let results = query(&memory, key, 10);
        sink += results.len();
    }
    let query_ns = t1.elapsed().as_nanos();
    let query_per_sec = probe_count as f64 / (query_ns as f64 / 1_000_000_000.0);
    let query_us_each = query_ns as f64 / probe_count as f64 / 1_000.0;

    // Prevent optimiser from eliminating the query loop
    let _ = sink;

    let store_row = vec![
        "store".to_string(),
        count.to_string(),
        format!("{store_us_each:.2}"),
        format!("{store_per_sec:.0}"),
    ];
    let query_row = vec![
        "query (top-10)".to_string(),
        probe_count.to_string(),
        format!("{query_us_each:.2}"),
        format!("{query_per_sec:.0}"),
    ];

    print_table(
        &["Operation", "Count", "Avg µs/op", "ops/sec"],
        &[store_row, query_row],
    );

    println!();
    println!("  Memory footprint: {} entries × {DIMS} dims × 1 byte (i8) ≈ {:.1} MB (key+value)",
        count,
        (count * DIMS * 2) as f64 / 1_048_576.0
    );
    println!();
}
