"""Tests for the kohaku visualization API.

Six tests cover graph structure, edge thresholding, decay shape, k-means
cluster recovery, and probe behaviour. The tests instantiate a fresh app via
`create_app()` and drive it through `fastapi.testclient.TestClient` — no real
network sockets, no global state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY_PKG = ROOT / "python"
if str(PY_PKG) not in sys.path:
    sys.path.insert(0, str(PY_PKG))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import VizState, create_app  # noqa: E402
from kohaku._pure import DIMS  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """Fresh app per test — VizState() reloads the sample concepts each time."""
    return TestClient(create_app(state=VizState()))


def test_graph_structure_has_required_fields(client: TestClient) -> None:
    """`/viz/graph` returns the contract expected by the d3-force viewer."""
    r = client.get("/viz/graph")
    assert r.status_code == 200
    data = r.json()

    assert {"nodes", "edges", "dims", "threshold", "num_clusters",
            "half_life", "current_clock"} <= data.keys()
    assert data["dims"] == DIMS
    assert data["num_clusters"] == 3
    assert len(data["nodes"]) == 12

    # Every node carries the fields the viewer renders.
    required_node = {
        "id", "entry_id", "label", "cluster", "cluster_label",
        "last_accessed", "age", "decay_weight",
    }
    for node in data["nodes"]:
        assert required_node <= node.keys(), f"missing fields on node {node['id']}"
        assert isinstance(node["cluster"], int) and 0 <= node["cluster"] < 3
        assert 0.0 <= node["decay_weight"] <= 1.0
        assert node["age"] >= 0


def test_edge_threshold_is_strictly_respected(client: TestClient) -> None:
    """Every emitted edge satisfies similarity >= threshold; raising the
    threshold can only shrink the edge set."""
    r_low = client.get("/viz/graph?threshold=0.3")
    r_high = client.get("/viz/graph?threshold=0.7")
    assert r_low.status_code == 200 and r_high.status_code == 200

    low = r_low.json()
    high = r_high.json()

    for e in low["edges"]:
        assert e["similarity"] >= 0.3 - 1e-6, e
    for e in high["edges"]:
        assert e["similarity"] >= 0.7 - 1e-6, e

    assert len(high["edges"]) <= len(low["edges"])
    # high-threshold edges must be a subset of low-threshold edges
    low_keys = {(e["source"], e["target"]) for e in low["edges"]}
    for e in high["edges"]:
        assert (e["source"], e["target"]) in low_keys


def test_kmeans_recovers_ground_truth_clusters(client: TestClient) -> None:
    """Cosine k-means on the 12 sample concepts must recover the three
    ground-truth labels (animals / programming / cities) — modulo the arbitrary
    cluster index permutation k-means produces."""
    r = client.get("/viz/graph?k=3")
    data = r.json()

    # Group node ids by their assigned cluster index, and by ground truth.
    by_assigned: dict[int, set[str]] = {}
    by_truth: dict[str, set[str]] = {}
    for n in data["nodes"]:
        by_assigned.setdefault(n["cluster"], set()).add(n["id"])
        by_truth.setdefault(n["cluster_label"], set()).add(n["id"])

    assigned_groups = sorted(by_assigned.values(), key=lambda s: sorted(s))
    truth_groups = sorted(by_truth.values(), key=lambda s: sorted(s))
    assert assigned_groups == truth_groups, (
        f"k-means clusters {assigned_groups} != ground truth {truth_groups}"
    )


def test_decay_curve_shape_and_monotonicity(client: TestClient) -> None:
    """`/viz/decay` returns one curve per concept, monotonically non-increasing
    in age, anchored at weight=1.0 for age=0."""
    r = client.get("/viz/decay?half_life=10&horizon=40&steps=20")
    assert r.status_code == 200
    data = r.json()

    assert data["half_life"] == 10
    assert data["horizon"] == 40
    assert len(data["concepts"]) == 12

    for c in data["concepts"]:
        assert {"id", "label", "last_accessed", "current_age",
                "current_weight", "curve"} <= c.keys()
        curve = c["curve"]
        assert len(curve) >= 2
        # ages cover [0, horizon], starting at 0
        assert curve[0]["age"] == 0
        assert curve[0]["weight"] == pytest.approx(1.0)
        assert curve[-1]["age"] == 40
        # monotonic non-increasing weight
        weights = [pt["weight"] for pt in curve]
        for prev, nxt in zip(weights, weights[1:]):
            assert nxt <= prev + 1e-9, f"non-monotonic curve: {weights}"
        # half-life sanity: weight at age=half_life ≈ 0.5
        midpoint = next((pt for pt in curve if pt["age"] == 10), None)
        if midpoint is not None:
            assert midpoint["weight"] == pytest.approx(0.5, abs=1e-3)


def test_decay_weights_match_ages_in_graph(client: TestClient) -> None:
    """The decay weight returned on each graph node must be exactly
    0.5**(age/half_life) — proving the API is calling the real
    `kohaku.decay.decay_weight` and not hard-coding a curve."""
    half = 8.0
    r = client.get(f"/viz/graph?half_life={half}")
    data = r.json()
    for n in data["nodes"]:
        expected = 0.5 ** (n["age"] / half)
        assert n["decay_weight"] == pytest.approx(expected, abs=1e-3), (
            f"node {n['id']}: expected {expected:.4f}, got {n['decay_weight']}"
        )


def test_probe_returns_top_k_in_target_cluster(client: TestClient) -> None:
    """A query phrase that overlaps the programming-cluster vocabulary must
    rank programming concepts above animals/cities concepts."""
    r = client.post(
        "/viz/probe",
        json={"text": "the function runs as a fast modern code program", "top_k": 4},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["top_k"] == 4
    assert len(data["matches"]) == 4

    # Top-4 must all be programming-cluster ids.
    programming = {"python", "rust", "function", "debugger"}
    top_ids = [m["id"] for m in data["matches"]]
    assert set(top_ids) == programming, f"unexpected top-4: {top_ids}"

    # Similarities are sorted descending.
    sims = [m["similarity"] for m in data["matches"]]
    assert sims == sorted(sims, reverse=True)

    # Empty query is rejected with 400.
    bad = client.post("/viz/probe", json={"text": "   "})
    assert bad.status_code == 400
