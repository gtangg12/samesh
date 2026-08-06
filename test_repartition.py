"""
Tests for the alpha-expansion graph-cut repartition (issue #13).

The previous implementation aborted the entire segmentation with
`raise ValueError('Cost increased. ...')` whenever the recomputed
energy of an alpha-expansion move exceeded the previous best by any
amount, including a floating-point epsilon. The patched version
accepts only moves that decrease energy (within a small tolerance)
and reverts the rest, matching the standard alpha-expansion algorithm
and degrading gracefully on imperfect meshes.

Run with: PYTHONPATH=src python test_repartition.py
Requires the same heavy deps as samesh.models.shape_diameter_function
(igraph, pymeshlab, sklearn, omegaconf, tqdm); the suite skips with a
clear message if any are missing so the lightweight test_fixes.py
runner is not affected.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import numpy as np
    import trimesh
    from samesh.models import shape_diameter_function as sdf_mod
    from samesh.models.shape_diameter_function import (
        repartition,
        partition_cost,
    )
except Exception as exc:  # pragma: no cover - exercised only when deps missing
    print(f"SKIP test_repartition: missing dependency ({exc.__class__.__name__}: {exc})")
    sys.exit(0)


EPSILON = 1e-20


def _sphere_inputs(seed: int = 0, K: int = 3):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    F = len(mesh.faces)
    rng = np.random.RandomState(seed)
    logits = rng.randn(F, K) * 2.0
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    cost_data = -np.log(probs + EPSILON)
    cost_smoothness = -np.log(mesh.face_adjacency_angles / np.pi + EPSILON)
    partition = np.argmin(cost_data, axis=1)
    return mesh, partition, cost_data, cost_smoothness


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_repartition_smoke_runs_and_returns_valid_labels():
    """End-to-end: repartition runs on a real mesh and returns a per-face label
    array drawn only from the labels present in the initial partition."""
    mesh, partition, cost_data, cost_smoothness = _sphere_inputs(seed=0, K=3)
    initial_labels = set(np.unique(partition).tolist())
    out = repartition(
        mesh, partition.copy(), cost_data, cost_smoothness,
        smoothing_iterations=2, _lambda=1.0,
    )
    assert out.shape == partition.shape
    assert set(np.unique(out).tolist()).issubset(initial_labels)


def test_repartition_rejects_non_improving_move_instead_of_raising():
    """Force partition_cost to report a cost increase after the very first move.
    The patched code must NOT raise; it must reject the move and return a valid
    partition. (The pre-fix code raised ValueError unconditionally.)"""
    mesh, partition, cost_data, cost_smoothness = _sphere_inputs(seed=1, K=3)

    original = sdf_mod.partition_cost
    state = {"calls": 0}

    def increasing_partition_cost(*args, **kwargs):
        # 1st call: baseline cost_min before the loop (compute honestly).
        # Subsequent calls: report a value much larger than the baseline so the
        # patched guard rejects every move. The original code raised here.
        state["calls"] += 1
        if state["calls"] == 1:
            baseline = float(original(*args, **kwargs))
            state["baseline"] = baseline
            return baseline
        return state["baseline"] + 1e6  # well outside the tolerance window

    sdf_mod.partition_cost = increasing_partition_cost
    try:
        out = repartition(
            mesh, partition.copy(), cost_data, cost_smoothness,
            smoothing_iterations=1, _lambda=1.0,
        )
    finally:
        sdf_mod.partition_cost = original

    # All moves rejected => partition is unchanged, no exception raised.
    assert out.shape == partition.shape
    assert np.array_equal(out, partition), "rejected moves should leave the partition untouched"


def test_repartition_accepts_tiny_floating_point_increase():
    """Patched code must tolerate a sub-epsilon FP overshoot of cost_min
    (the exact symptom of issue #13) and continue rather than crash."""
    mesh, partition, cost_data, cost_smoothness = _sphere_inputs(seed=2, K=3)

    original = sdf_mod.partition_cost
    state = {"calls": 0, "baseline": None}

    def jitter_partition_cost(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            state["baseline"] = float(original(*args, **kwargs))
            return state["baseline"]
        # Always report cost_min + a tiny epsilon: theoretically optimal but
        # numerically above cost_min. Pre-fix code raised; patched code
        # accepts within tolerance.
        return state["baseline"] + 1e-12

    sdf_mod.partition_cost = jitter_partition_cost
    try:
        out = repartition(
            mesh, partition.copy(), cost_data, cost_smoothness,
            smoothing_iterations=1, _lambda=1.0,
        )
    finally:
        sdf_mod.partition_cost = original

    assert out.shape == partition.shape  # completed without raising


def test_repartition_clamps_negative_smoothness_costs():
    """Smoothness term -log(angle/pi + eps) dips slightly negative at angle=pi.
    Negative capacities violate the min-cut precondition; repartition must
    handle them without raising."""
    mesh, partition, cost_data, _ = _sphere_inputs(seed=3, K=3)
    E = len(mesh.face_adjacency)
    cost_smoothness = np.full(E, -1e-20)  # uniformly tiny-negative
    out = repartition(
        mesh, partition.copy(), cost_data, cost_smoothness,
        smoothing_iterations=1, _lambda=1.0,
    )
    assert out.shape == partition.shape


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_repartition_smoke_runs_and_returns_valid_labels,
    test_repartition_rejects_non_improving_move_instead_of_raising,
    test_repartition_accepts_tiny_floating_point_increase,
    test_repartition_clamps_negative_smoothness_costs,
]


def main() -> int:
    passed = 0
    failed = []
    for fn in TESTS:
        try:
            fn()
        except Exception:
            failed.append((fn.__name__, traceback.format_exc()))
            print(f"FAIL  {fn.__name__}")
        else:
            passed += 1
            print(f"PASS  {fn.__name__}")

    print(f"\n{passed}/{len(TESTS)} tests passed")
    for name, tb in failed:
        print(f"\n--- {name} ---\n{tb}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
