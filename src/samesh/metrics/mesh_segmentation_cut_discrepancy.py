from typing import Mapping
import trimesh
import numpy as np
import igraph
from samesh.data.common import *


def compute_cut_discrepancy(mesh: trimesh.Trimesh, s1: NumpyTensor['f'], s2: NumpyTensor['f']) -> float:
    """
    """
    cut1 = _get_cut_vertex(mesh, s1)
    cut2 = _get_cut_vertex(mesh, s2)
    if len(cut1) == 0 or \
       len(cut2) == 0: # Undefined for empty cuts
        return 0
    d12 = _compute_distance_cuts(mesh, cut1, cut2)
    d21 = _compute_distance_cuts(mesh, cut2, cut1)
    cd = (d12.sum() + d21.sum()) / (len(d12) + len(d21)) # same bug as in SegEval's original code
    avg_radius = _approx_average_radius(mesh)
    return cd / avg_radius


def _compute_distance_cuts(mesh: trimesh.Trimesh, cut1: NumpyTensor['f'], cut2: NumpyTensor['f']) -> NumpyTensor | None:
    """
    Compute the mean distance from vertices in cut1 to the closest vertex in cut2.
    Distance is taken as along the skeleton of the mesh (i.e. shortest path through mesh edges). This is consistent
    with SegEval metric (https://segeval.cs.princeton.edu/).
    """
    S2_node = len(mesh.vertices)
    
    graph = igraph.Graph(directed=False)
    graph.add_vertices(len(mesh.vertices) + 1)
    graph.add_edges(
        mesh.edges, attributes={'weight': np.linalg.norm(
            mesh.vertices[mesh.edges[:, 0]] - 
            mesh.vertices[mesh.edges[:, 1]], axis=1
        )},
    )
    graph.add_edges(
        [(S2_node, vertex) for vertex in cut2], attributes={
            'weight': np.zeros(len(cut2))
        },
    )
    shortest_path = np.array(graph.shortest_paths(source=S2_node, target=cut1, weights='weight'))
    assert shortest_path.shape == (1, len(cut1))
    return shortest_path[0]


def _get_cut_vertex(mesh: trimesh.Trimesh, partition: NumpyTensor['f']) -> set[int]:
    """
    Get all vertices along cut boundaries of a segmentation
    """
    vpair2face: Mapping[tuple[int, int], list[int]] = {}
    for i, (v0, v1, v2) in enumerate(mesh.faces):
        vpair2face.setdefault(tuple(sorted((v0, v1))), []).append(i)
        vpair2face.setdefault(tuple(sorted((v1, v2))), []).append(i)
        vpair2face.setdefault(tuple(sorted((v2, v0))), []).append(i)
    cut: set[int] = set()
    for vpair, fpair in vpair2face.items():
        if len(fpair) == 1:
            continue  # this is a boundary edge
        #assert len(fpair) == 2
        if partition[fpair[0]] != partition[fpair[1]]:
            cut.add(vpair[0])
            cut.add(vpair[1])
    return cut


def _approx_average_radius(mesh: trimesh.Trimesh) -> float:
    """
    Weighted distance from an average face to the centroid of the surface
    """
    face_cents = np.mean(mesh.vertices[mesh.faces], axis=1) # (F, 3)
    face_areas = np.linalg.norm(
        np.cross(
            mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]],
            mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]],
        ), axis=1,
    )
    cent = (face_cents * face_areas[:, None]).sum(axis=0) / face_areas.sum()
    dist = np.linalg.norm(face_cents - cent[None, :], axis=1)
    return (dist * face_areas).sum() / face_areas.sum()