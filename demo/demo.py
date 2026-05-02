"""Kohaku demo — exercise the live API with rich terminal output.

Run from the repo root:

    PYTHONPATH=python python3 demo/demo.py

Sections:
    1. Encode 10 text strings → hypervectors (shape, dtype, type)
    2. Bundle 3 related vectors and query the bundle
    3. Persistence — JSON vs binary .hkb round-trip
    4. Consolidation — 6 noisy bundles → 2 clusters
    5. Decay — query_with_decay() across 5 simulated time steps
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Allow running directly from the repo without `pip install`
ROOT = Path(__file__).resolve().parent.parent
PY_PKG = ROOT / "python"
if str(PY_PKG) not in sys.path:
    sys.path.insert(0, str(PY_PKG))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from kohaku import (
    DecayConfig,
    EpisodicMemory,
    HyperVector,
    consolidate,
    encode_text,
    load,
    query,
    query_with_decay,
    save,
)
from kohaku._pure import DIMS

console = Console()


def header(title: str, subtitle: str = "") -> None:
    body = Text(title, style="bold bright_cyan")
    if subtitle:
        body.append("\n")
        body.append(subtitle, style="dim")
    console.print(Panel(body, border_style="cyan", padding=(0, 2)))


def section(num: int, title: str) -> None:
    console.print()
    console.print(Rule(f"[bold yellow]{num}. {title}[/bold yellow]", style="yellow"))
    console.print()


def ascii_bar(value: float, max_value: float, width: int = 40, char: str = "█") -> str:
    if max_value <= 0:
        return ""
    n = int(round((value / max_value) * width))
    return char * max(0, min(width, n))


# ---------------------------------------------------------------------------
# 1. Encode 10 text strings
# ---------------------------------------------------------------------------

def section_encode() -> list[tuple[str, HyperVector]]:
    section(1, "HDC Encoding — text → 10 000-D bipolar hypervectors")

    phrases = [
        "the cat sat on the mat",
        "a feline rested on a rug",
        "dogs bark loudly at strangers",
        "puppies make joyful noise",
        "espresso is bold and bitter",
        "coffee tastes strong in the morning",
        "the river flows to the sea",
        "rivers run downhill always",
        "stars burn bright in the night sky",
        "konjo means beautiful in amharic",
    ]

    encoded: list[tuple[str, HyperVector]] = []
    for p in phrases:
        encoded.append((p, encode_text(p)))

    sample = encoded[0][1]
    table = Table(title="Encoded phrases", show_lines=False, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Phrase", style="white")
    table.add_column("Type", style="green")
    table.add_column("Dims", justify="right", style="yellow")
    table.add_column("dtype", style="blue")
    table.add_column("First 6 components", style="dim")
    for i, (text, hv) in enumerate(encoded, start=1):
        first6 = " ".join(f"{x:+d}" for x in hv.data[:6])
        table.add_row(
            str(i),
            text,
            type(hv).__name__,
            str(len(hv)),
            str(hv.data.dtype),
            first6 + " …",
        )
    console.print(table)

    info = (
        f"[bold]All vectors are bipolar ±1[/bold]   "
        f"min={int(sample.data.min())}  max={int(sample.data.max())}  "
        f"mean={sample.data.mean():.4f}  "
        f"|+1|={int((sample.data == 1).sum())}  "
        f"|-1|={int((sample.data == -1).sum())}"
    )
    console.print(Panel(info, border_style="green", title="Vector contract"))
    return encoded


# ---------------------------------------------------------------------------
# 2. Bundle and query
# ---------------------------------------------------------------------------

def section_bundle(encoded: list[tuple[str, HyperVector]]) -> HyperVector:
    section(2, "Bundle & Query — three felines combine into one prototype")

    cat_phrases = [
        "the cat sat on the mat",
        "a feline rested on a rug",
        "kittens nap in warm sunbeams",
    ]
    cat_hvs = [encode_text(p) for p in cat_phrases]
    bundle_vec = HyperVector.bundle_all(cat_hvs)

    console.print(
        Panel(
            "[bold]bundle_all(v1, v2, v3)[/bold] = sign( v1 + v2 + v3 ) — majority vote\n"
            "Each component takes the sign of its sum across all input vectors.",
            border_style="cyan",
            title="Operation",
        )
    )

    # Build a memory with a mix of phrases and query the bundle
    mem = EpisodicMemory(capacity=64)
    val_dummy = HyperVector.random(DIMS, seed=0)
    for text, hv in encoded:
        mem.store(hv, val_dummy, label=text)

    results = query(mem, bundle_vec, top_k=5)

    table = Table(title="Top-5 matches against the cat-bundle", header_style="bold magenta")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Phrase", style="white")
    table.add_column("Cosine sim", justify="right", style="yellow")
    table.add_column("", style="green")
    max_sim = max((abs(r.similarity) for r in results), default=1.0)
    for i, r in enumerate(results, start=1):
        bar = ascii_bar(max(0.0, r.similarity), max_sim, width=30)
        table.add_row(str(i), r.label, f"{r.similarity:+.4f}", bar)
    console.print(table)

    bundle_self = bundle_vec.cosine_similarity(bundle_vec)
    bundle_to_cat = bundle_vec.cosine_similarity(cat_hvs[0])
    console.print(
        f"[dim]bundle⋅bundle = {bundle_self:.4f}  ·  bundle⋅cat[0] = {bundle_to_cat:.4f}[/dim]"
    )
    return bundle_vec


# ---------------------------------------------------------------------------
# 3. Persistence — JSON vs binary
# ---------------------------------------------------------------------------

def section_persistence(encoded: list[tuple[str, HyperVector]]) -> None:
    section(3, "Persistence — JSON vs binary .hkb round-trip")

    mem = EpisodicMemory(capacity=64)
    val_dummy = HyperVector.random(DIMS, seed=0)
    for text, hv in encoded:
        mem.store(hv, val_dummy, label=text)

    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "memories.json"
        hkb_path = Path(tmp) / "memories.hkb"

        save(mem, json_path)
        save(mem, hkb_path)
        json_size = json_path.stat().st_size
        hkb_size = hkb_path.stat().st_size

        loaded_json = load(json_path)
        loaded_hkb = load(hkb_path)

        # Verify round-trip via recall on the first key
        target = mem.entries()[0].key
        original = query(mem, target, top_k=3)
        rj = query(loaded_json, target, top_k=3)
        rh = query(loaded_hkb, target, top_k=3)
        json_ok = [r.entry_id for r in rj] == [r.entry_id for r in original]
        hkb_ok = [r.entry_id for r in rh] == [r.entry_id for r in original]

        table = Table(title="On-disk size for 10 entries", header_style="bold magenta")
        table.add_column("Format", style="cyan")
        table.add_column("Size (bytes)", justify="right", style="yellow")
        table.add_column("Size (KB)", justify="right", style="yellow")
        table.add_column("Round-trip recall", style="green")
        table.add_column("Ratio vs JSON", justify="right", style="blue")
        table.add_row(
            ".json", f"{json_size:,}", f"{json_size/1024:.1f}",
            "✓ identical" if json_ok else "✗ differs", "1.00×",
        )
        table.add_row(
            ".hkb", f"{hkb_size:,}", f"{hkb_size/1024:.1f}",
            "✓ identical" if hkb_ok else "✗ differs",
            f"{hkb_size / json_size:.3f}×",
        )
        console.print(table)

        savings = (1 - hkb_size / json_size) * 100
        console.print(
            Panel(
                f"[bold]Binary .hkb is ~{json_size / hkb_size:.1f}× smaller[/bold] "
                f"({savings:.1f}% savings) — packs each ±1 component into 1 bit.",
                border_style="green",
                title="건조 — strip to the essence",
            )
        )


# ---------------------------------------------------------------------------
# 4. Consolidation — 6 noisy bundles → clusters
# ---------------------------------------------------------------------------

def _noisy(base: HyperVector, flip_frac: float, seed: int) -> HyperVector:
    rng = np.random.default_rng(seed)
    n_flip = int(flip_frac * len(base))
    idx = rng.choice(len(base), size=n_flip, replace=False)
    data = base.data.copy()
    data[idx] *= -1
    return HyperVector(data)


def section_consolidation() -> None:
    section(4, "Consolidation — 6 noisy bundles → semantic clusters")

    cat_proto = encode_text("the cat sat on the mat")
    coffee_proto = encode_text("espresso is bold and bitter")

    members = [
        ("cat-1", _noisy(cat_proto, 0.06, seed=10), "cat"),
        ("cat-2", _noisy(cat_proto, 0.05, seed=11), "cat"),
        ("cat-3", _noisy(cat_proto, 0.04, seed=12), "cat"),
        ("coffee-1", _noisy(coffee_proto, 0.05, seed=20), "coffee"),
        ("coffee-2", _noisy(coffee_proto, 0.06, seed=21), "coffee"),
        ("coffee-3", _noisy(coffee_proto, 0.04, seed=22), "coffee"),
    ]

    mem = EpisodicMemory(capacity=16)
    val_dummy = HyperVector.random(DIMS, seed=0)
    id_to_label = {}
    for name, hv, lab in members:
        eid = mem.store(hv, val_dummy, label=name)
        id_to_label[eid] = name

    clusters = consolidate(mem, similarity_threshold=0.3)

    table = Table(title=f"{len(clusters)} clusters from {len(members)} entries",
                  header_style="bold magenta")
    table.add_column("Cluster", style="cyan", justify="right")
    table.add_column("Seed label", style="white")
    table.add_column("Members", style="yellow")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Centroid → cat-proto", justify="right", style="blue")
    table.add_column("Centroid → coffee-proto", justify="right", style="blue")
    for i, c in enumerate(clusters, start=1):
        names = ", ".join(id_to_label[mid] for mid in c.member_ids)
        s_cat = c.centroid_key.cosine_similarity(cat_proto)
        s_cof = c.centroid_key.cosine_similarity(coffee_proto)
        table.add_row(str(i), c.label, names, str(c.size), f"{s_cat:+.3f}", f"{s_cof:+.3f}")
    console.print(table)

    # Show centroid concentration: avg single-member sim vs centroid sim to the prototype
    cat_cluster = max(clusters, key=lambda c: c.centroid_key.cosine_similarity(cat_proto))
    member_sims = []
    for name, hv, lab in members:
        if lab == "cat":
            member_sims.append(cat_proto.cosine_similarity(hv))
    avg_member = float(np.mean(member_sims))
    centroid_sim = cat_proto.cosine_similarity(cat_cluster.centroid_key)
    console.print(
        Panel(
            f"avg(member → cat-proto) = [bold yellow]{avg_member:+.4f}[/bold yellow]\n"
            f"centroid → cat-proto    = [bold green]{centroid_sim:+.4f}[/bold green]\n"
            "[dim]Bundling concentrates the signal — the centroid is closer to the\n"
            "latent prototype than any single noisy member.[/dim]",
            border_style="cyan",
            title="Bundle-of-bundles — centroid concentration",
        )
    )


# ---------------------------------------------------------------------------
# 5. Temporal decay
# ---------------------------------------------------------------------------

def section_decay() -> None:
    section(5, "Ebbinghaus decay — query_with_decay() across 5 time steps")

    # Build a memory with a known target stored at t=1 (oldest), then 4 fillers.
    mem = EpisodicMemory(capacity=32)
    target = encode_text("konjo means beautiful")
    val_dummy = HyperVector.random(DIMS, seed=0)
    target_id = mem.store(target, val_dummy, label="konjo means beautiful")
    for i in range(1, 5):
        filler = encode_text(f"unrelated phrase number {i}")
        mem.store(filler, val_dummy, label=f"filler-{i}")

    half_life = 2.0
    cfg = DecayConfig(half_life=half_life)

    # Simulate 5 time steps. At each step, simulate the target being aged further
    # by appending an extra filler and then re-querying.
    rows = []
    for step in range(5):
        if step > 0:
            mem.store(
                encode_text(f"time-step filler {step}"),
                val_dummy,
                label=f"step-{step}",
            )
        results = query_with_decay(mem, target, top_k=8, config=cfg)
        target_result = next(r for r in results if r.entry_id == target_id)
        # Compute age the same way query_with_decay does
        now = mem._timestamp - 1
        target_entry = next(e for e in mem.entries() if e.id == target_id)
        age = now - target_entry.timestamp
        rows.append((step, age, target_result.similarity))

    max_sim = max(r[2] for r in rows)
    table = Table(title=f"Decay of target memory  (half_life = {half_life} ticks)",
                  header_style="bold magenta")
    table.add_column("t", style="cyan", justify="right")
    table.add_column("age (ticks)", justify="right", style="yellow")
    table.add_column("decayed sim", justify="right", style="green")
    table.add_column("strength", style="bright_red")
    for step, age, sim in rows:
        bar = ascii_bar(max(0.0, sim), max_sim, width=40)
        table.add_row(str(step), str(age), f"{sim:+.4f}", bar)
    console.print(table)

    console.print(
        Panel(
            "[bold]weight(age) = max(0.5 ** (age / half_life), floor)[/bold]\n"
            "[dim]Sign of similarity is preserved; magnitude decays exponentially.\n"
            "Older memories fade — recent ones dominate recall.[/dim]",
            border_style="green",
            title="Forgetting curve",
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    header(
        "Kohaku — HDC Episodic Memory  ·  Demo Day v0.4.0",
        "Hyperdimensional encoding · Bundle & query · Persistence · "
        "Consolidation · Forgetting curves",
    )

    encoded = section_encode()
    section_bundle(encoded)
    section_persistence(encoded)
    section_consolidation()
    section_decay()

    console.print()
    console.print(
        Panel(
            "[bold bright_cyan]ቆንጆ — beautiful.  根性 — never surrender.  "
            "康宙 — leave it healthier than you found it.[/bold bright_cyan]\n"
            "[dim]Build, ship, repeat.[/dim]",
            border_style="bright_cyan",
            title="Konjo Mode",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
