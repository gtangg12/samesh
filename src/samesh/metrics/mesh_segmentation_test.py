import math
import numpy as np
import trimesh

from samesh.metrics.mesh_segmentation import compute_metrics


def test_combinatorial_metrics():
    mesh = trimesh.Trimesh(
        vertices=[
            [ 1,  0,  0],
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0, -1,  0],
            [ 0,  0,  1],
            [ 0,  0, -1],
        ],
        faces=[
            [0, 2, 4],
            [0, 4, 3],
            [0, 3, 5],
            [0, 5, 2],
        ],
    )
    estimated = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32)
    reference = np.array([0, 1, 3, 2, 2, 2], dtype=np.uint32)

    metrics = compute_metrics(mesh, estimated, reference)
    assert math.isclose(metrics.hamming_distance_rm, 1 / 6)
    assert math.isclose(metrics.hamming_distance_rf, 1 / 3)
    assert math.isclose(metrics.hamming_distance, 1 / 4)
    assert math.isclose(metrics.inv_rand_index, 4 / 15)
    assert math.isclose(metrics.lce, 1 / 12)
    assert math.isclose(metrics.gce, 2 / 9)


def test_cut_discrepancy():
    octahedron = trimesh.Trimesh(
        vertices=[
            [ 1,  0,  0],
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0, -1,  0],
            [ 0,  0,  1],
            [ 0,  0, -1],
        ],
        faces=[
            [0, 2, 4],
            [0, 4, 3],
            [0, 3, 5],
            [0, 5, 2],
            [1, 4, 2],
            [1, 3, 4],
            [1, 5, 3],
            [1, 2, 5],
        ],
    )
    estimated = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)  # right half 0, left half 1
    reference = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint32)  # top half 0, bottom half 1
    metrics = compute_metrics(octahedron, estimated, reference)
    assert math.isclose(metrics.cut_discrepancy, math.sqrt(3/2))


if __name__ == "__main__":
    test_combinatorial_metrics()
    test_cut_discrepancy()
    print("All tests passed!")