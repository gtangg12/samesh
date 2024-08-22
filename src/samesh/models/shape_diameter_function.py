import glob
import json
import os
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pymeshlab
import trimesh
import networkx as nx
import igraph
from numpy.random import RandomState
from trimesh.base import Trimesh, Scene
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from omegaconf import OmegaConf

from samesh.data.common import NumpyTensor
from samesh.data.loaders import scene2mesh, read_mesh
from samesh.utils.mesh import duplicate_verts


EPSILON = 1e-20
SCALE = 1e6


def partition_cost(
    mesh           : Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> float:
    """
    """
    cost = 0
    for f in range(len(partition)):
        cost += cost_data[f, partition[f]]
    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            cost += cost_smoothness[i]
    return cost


def construct_expansion_graph(
    label          : int,
    mesh           : Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> nx.Graph:
    """
    """
    G = nx.Graph() # undirected graph
    A = 'alpha'
    B = 'alpha_complement'

    node2index = {}
    G.add_node(A)
    G.add_node(B)
    node2index[A] = 0
    node2index[B] = 1
    for i in range(len(mesh.faces)):
        G.add_node(i)
        node2index[i] = 2 + i

    aux_count = 0
    for i, edge in enumerate(mesh.face_adjacency): # auxillary nodes
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            a = (f1, f2)
            if a in node2index: # duplicate edge
                continue
            G.add_node(a)
            node2index[a] = len(mesh.faces) + 2 + aux_count
            aux_count += 1

    for f in range(len(mesh.faces)):
        G.add_edge(A, f, capacity=cost_data[f, label])
        G.add_edge(B, f, capacity=float('inf') if partition[f] == label else cost_data[f, partition[f]])

    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        a = (f1, f2)
        if partition[f1] == partition[f2]:
            if partition[f1] != label:
                G.add_edge(f1, f2, capacity=cost_smoothness[i])
        else:
            G.add_edge(a, B, capacity=cost_smoothness[i])
            if partition[f1] != label:
                G.add_edge(f1, a, capacity=cost_smoothness[i])
            if partition[f2] != label:
                G.add_edge(a, f2, capacity=cost_smoothness[i])
    
    return G, node2index


def repartition(
    mesh: trimesh.Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e'],
    smoothing_iterations: int,
    _lambda=1.0,
):
    A = 'alpha'
    B = 'alpha_complement'
    labels = np.unique(partition)

    cost_smoothness = cost_smoothness * _lambda

    # networkx broken for float capacities
    #cost_data       = np.round(cost_data       * SCALE).astype(int)
    #cost_smoothness = np.round(cost_smoothness * SCALE).astype(int)

    cost_min = partition_cost(mesh, partition, cost_data, cost_smoothness)

    for i in range(smoothing_iterations):

        #print('Repartition iteration ', i)
        
        for label in tqdm(labels):
            G, node2index = construct_expansion_graph(label, mesh, partition, cost_data, cost_smoothness)
            index2node = {v: k for k, v in node2index.items()}

            '''
            _, (S, T) = nx.minimum_cut(G, A, B)
            assert A in S and B in T
            S = np.array([v for v in S if isinstance(v, int)]).astype(int)
            T = np.array([v for v in T if isinstance(v, int)]).astype(int)
            '''

            G = igraph.Graph.from_networkx(G)
            outputs = G.st_mincut(source=node2index[A], target=node2index[B], capacity='capacity')
            S = outputs.partition[0]
            T = outputs.partition[1]
            assert node2index[A] in S and node2index[B] in T
            S = np.array([index2node[v] for v in S if isinstance(index2node[v], int)]).astype(int)
            T = np.array([index2node[v] for v in T if isinstance(index2node[v], int)]).astype(int)

            assert (partition[S] == label).sum() == 0 # T consists of those assigned 'alpha' and S 'alpha_complement' (see paper)
            partition[T] = label

            cost = partition_cost(mesh, partition, cost_data, cost_smoothness)
            if cost > cost_min:
                raise ValueError('Cost increased. This should not happen because the graph cut is optimal.')
            cost_min = cost
    
    return partition


def prep_mesh_shape_diameter_function(source: Trimesh | Scene) -> Trimesh:
    """
    """
    if isinstance(source, trimesh.Scene):
        source = scene2mesh(source)
    source.merge_vertices(merge_tex=True, merge_norm=True)
    return source


def colormap_shape_diameter_function(mesh: Trimesh, sdf_values: NumpyTensor['f']) -> Trimesh:
    """
    """
    assert len(mesh.faces) == len(sdf_values)
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    mesh.visual.face_colors = trimesh.visual.interpolate(sdf_values, color_map='jet')
    return mesh


def colormap_partition(mesh: Trimesh, partition: NumpyTensor['f']) -> Trimesh:
    """
    """
    assert len(mesh.faces) == len(partition)
    palette = RandomState(0).randint(0, 255, (np.max(partition) + 1, 3)) # must init every time to get same colors
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    mesh.visual.face_colors = palette[partition]
    return mesh


def shape_diameter_function(mesh: Trimesh, norm=True, alpha=4, rays=64, cone_amplitude=120) -> NumpyTensor['f']:
    """
    """
    mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(mesh)
    meshset.compute_scalar_by_shape_diameter_function_per_vertex(rays=rays, cone_amplitude=cone_amplitude)

    sdf_values = meshset.current_mesh().face_scalar_array()
    sdf_values[np.isnan(sdf_values)] = 0
    if norm:
        # normalize and smooth shape diameter function values
        min = sdf_values.min()
        max = sdf_values.max()
        sdf_values = (sdf_values - min) / (max - min)
        sdf_values = np.log(sdf_values * alpha + 1) / np.log(alpha + 1)
    return sdf_values


def partition_faces(mesh: Trimesh, num_components: int, _lambda: float, smooth=True, smoothing_iterations=1, **kwargs) -> NumpyTensor['f']:
    """
    """
    sdf_values = shape_diameter_function(mesh, norm=True).reshape(-1, 1)

    # fit 1D GMM to shape diameter function values
    gmm = GaussianMixture(num_components)
    gmm.fit(sdf_values)
    probs = gmm.predict_proba(sdf_values)
    if not smooth:
        return np.argmax(probs, axis=1)

    # data and smoothness terms
    cost_data       = -np.log(probs + EPSILON)
    cost_smoothness = -np.log(mesh.face_adjacency_angles / np.pi + EPSILON)
    cost_smoothness = _lambda * cost_smoothness

    # generate initial partition and refine with alpha expansion graph cut
    partition = np.argmin(cost_data, axis=1)
    partition = repartition(mesh, partition, cost_data, cost_smoothness, smoothing_iterations=smoothing_iterations)
    return partition


def partition2label(mesh: Trimesh, partition: NumpyTensor['f']) -> NumpyTensor['f']:
    """
    """
    edges = trimesh.graph.face_adjacency(mesh=mesh)
    graph = defaultdict(set)
    for face1, face2 in edges:
        graph[face1].add(face2)
        graph[face2].add(face1)
    labels = set(list(np.unique(partition)))
    
    components = []
    visited = set()

    def dfs(source: int):
        stack = [source]
        components.append({source})
        visited.add(source)
        
        while stack:
            node = stack.pop()
            for adj in graph[node]:
                if adj not in visited and partition[adj] == partition[node]:
                    stack.append(adj)
                    components[-1].add(adj)
                    visited.add(adj)

    for face in range(len(mesh.faces)):
        if face not in visited:
            dfs(face)

    partition_output = np.zeros_like(partition)
    label_total = 0
    for component in components:
        for face in component:
            partition_output[face] = label_total
        label_total += 1
    return partition_output


def segment_mesh_sdf(filename: Path | str, config: OmegaConf, extension='glb') -> Trimesh:
    """
    """
    print('Segmenting mesh with Shape Diameter Funciont: ', filename)
    filename = Path(filename)
    config = copy.deepcopy(config)
    config.output = Path(config.output) / filename.stem

    mesh = read_mesh(filename, norm=True)
    mesh = prep_mesh_shape_diameter_function(mesh)
    partition              = partition_faces(mesh, config.num_components, config.repartition_lambda, config.repartition_iterations)
    partition_disconnected = partition2label(mesh, partition)
    faces2label = {int(i): int(partition_disconnected[i]) for i in range(len(partition_disconnected))}

    os.makedirs(config.output, exist_ok=True)
    mesh_colored = colormap_partition(mesh, partition_disconnected)
    mesh_colored.export        (f'{config.output}/{filename.stem}_segmented.{extension}')
    json.dump(faces2label, open(f'{config.output}/{filename.stem}_face2label.json', 'w'))
    return mesh_colored
    

if __name__ == '__main__':
    import glob
    from natsort import natsorted

    def read_filenames(pattern: str):
        """
        """
        filenames = glob.glob(pattern)
        filenames = map(Path, filenames)
        filenames = natsorted(list(set(filenames)))
        print('Segmenting ', len(filenames), ' meshes')
        return filenames

    '''
    config = OmegaConf.load('/home/ubuntu/meshseg/configs/mesh_segmentation_shape_diameter_function.yaml')
    filenames = read_filenames('/home/ubuntu/data/backflip-benchmark-remeshed-processed/*.glb')

    for filename in filenames:
        segment_mesh_sdf(filename config)
    '''
    
    config = OmegaConf.load('/home/ubuntu/meshseg/configs/mesh_segmentation_shape_diameter_function_coseg.yaml')
    categories = ['candelabra', 'chairs', 'fourleg', 'goblets', 'guitars', 'irons', 'lamps', 'vases']
    for cat in categories:
        filenames = read_filenames(f'/home/ubuntu/data/coseg/{cat}/*.off')
        for filename in filenames:
            config = copy.deepcopy(config)
            config.output = Path(config.output) / cat
            segment_mesh_sdf(filename, config)