from __future__ import annotations

import glob
import json
import dataclasses
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import trimesh
from tqdm import tqdm

from samesh.data.common import NumpyTensor
from samesh.metrics.mesh_segmentation_cut_discrepancy import compute_cut_discrepancy


@dataclass
class Metrics:
    """
    Metrics as described in the MeshsegBenchmark paper (https://segeval.cs.princeton.edu/).
    """
    # Cut Discrepancy according to SegEval implementation
    cut_discrepancy: float

    # Hamming distance and Directional Hamming distances
    hamming_distance_rm    : float
    hamming_distance_rf: float
    hamming_distance                 : float

    # Rand Index
    inv_rand_index: float

    # Local and global consistency errors
    lce: float
    gce: float

    @staticmethod
    def average(metrics: Sequence[Metrics]) -> Metrics:
        """
        Given a sequence of metrics, compute the average of each metric and return a new Metrics object
        """
        metrics_dicts = [dataclasses.asdict(m) for m in metrics]
        metrics_means = {k: np.mean(
            [d[k] for d in metrics_dicts if d[k] is not None] # handle case where cut discrepancy is undefined
        ) for k in metrics_dicts[0].keys()}
        return Metrics(**metrics_means)
    
    def check_bounds(self) -> str | None:
        for k, v in dataclasses.asdict(self).items():
            if k == 'cut_discrepancy':
                if v is None: # handle case where cut discrepancy is undefined
                    continue
                if not 0 <= v:
                    return f'{k} {v} not in [0, inf)'
            else:
                if not 0 <= v <= 1:
                    return f'{k} {v} not in [0, 1]'


@dataclass
class SegmentSizes:
    """
    """
    total_faces: int
    estimated: NumpyTensor['num_estimated']
    reference: NumpyTensor['num_reference']
    intersect: NumpyTensor['num_estimated, num_reference']
    
    def check_bounds(self) -> str | None:
        if not np.all(self.intersect.sum(axis=1) == self.estimated):
            return 'intersect sizes do not sum to estimated sizes'
        if not np.all(self.intersect.sum(axis=0) == self.reference):
            return 'intersect sizes do not sum to ground truth sizes'
        if self.intersect.sum() != self.total_faces:
            return f'intersect sizes sum to {self.intersect.sum()} instead of {self.total_faces}'

        if np.any(self.estimated < 0):
            return 'Estimated sizes contain negative values'
        if np.any(self.reference < 0):
            return 'Ground truth sizes contain negative values'
        if np.any(self.intersect < 0):
            return 'intersect sizes contain negative values'


def compute_metrics(mesh: trimesh.Trimesh | None, estimated: NumpyTensor['f'], reference: NumpyTensor['f']) -> Metrics:
    """
    """
    metrics = {}
    segment_sizes = _compute_segment_sizes(estimated, reference)
    rm = (
        _compute_directional_hamming_distance(segment_sizes.reference, segment_sizes.intersect.T)
        / segment_sizes.total_faces
    )
    rf = (
        _compute_directional_hamming_distance(segment_sizes.estimated, segment_sizes.intersect)
        / segment_sizes.total_faces
    )
    hamming_distance = (rm + rf) / 2
    metrics.update({
        'hamming_distance_rm': rm,
        'hamming_distance_rf': rf,
        'hamming_distance': hamming_distance,
    })
    metrics['inv_rand_index'] = 1 - _compute_rand_index(segment_sizes)
    metrics['gce'], metrics['lce'] = _compute_consistency_error(segment_sizes, estimated, reference)
    metrics['cut_discrepancy'] = compute_cut_discrepancy(mesh, estimated, reference)
    return Metrics(**metrics)


def _compute_segment_sizes(estimated: NumpyTensor['f'], reference: NumpyTensor['f']) -> SegmentSizes:
    """
    """
    def bincount_check(arr):
        sizes = np.bincount(arr)
        assert len(sizes) == np.amax(arr) + 1
        return sizes, len(sizes)
    
    estimated_sizes, P_estimated = bincount_check(estimated)
    reference_sizes, P_reference = bincount_check(reference)
    intersect_sizes = np.bincount(
        estimated * P_reference + reference, minlength=P_estimated * P_reference
    ).reshape((P_estimated, P_reference))
    res = SegmentSizes(
        total_faces=len(estimated),
        estimated=estimated_sizes, 
        reference=reference_sizes, 
        intersect=intersect_sizes,
    )
    assert (err := res.check_bounds()) is None, err
    return res


def _compute_directional_hamming_distance(s2_sizes: NumpyTensor, intersect_sizes: NumpyTensor) -> float:
    """
    """
    return sum(s2_sizes) - sum(intersect_sizes.max(axis=1))


def _compute_rand_index(sizes: SegmentSizes) -> float:
    """
    """
    def choose_2(n):
        return n * (n - 1) / 2
    
    N2  = choose_2(sizes.total_faces)
    s1  = choose_2(sizes.estimated).sum()
    s2  = choose_2(sizes.reference).sum()
    s12 = choose_2(sizes.intersect).sum()
    return (N2 - s1 - s2 + 2 * s12) / N2


def _compute_consistency_error(sizes: SegmentSizes, estimated: NumpyTensor['f'], reference: NumpyTensor['f']) -> tuple[float, float]:
    """
    """
    R1 = sizes.estimated[estimated]
    R2 = sizes.reference[reference]
    E12 = (R1 - sizes.intersect[estimated, reference]) / R1
    E21 = (R2 - sizes.intersect[estimated, reference]) / R2
    assert E12.shape == estimated.shape
    assert E21.shape == reference.shape
    gce = min(E21.sum(), E12.sum())    / sizes.total_faces
    lce = np.sum(np.minimum(E12, E21)) / sizes.total_faces
    return gce, lce


def seg_from_face2label(filename: Path | str) -> np.ndarray:
    """
    """
    face2label = json.loads(Path(filename).read_text())
    face2label = {int(k): int(v) for k, v in face2label.items()}
    face2label_items = sorted(face2label.items())
    assert face2label_items[ 0][0] == 0
    assert face2label_items[-1][0] == len(face2label) - 1
    return np.array([label for _, label in face2label_items], dtype=np.uint32)


def benchmark_dataset_princeton_one(
    path_meshes                 : Path | str,
    path_segmentations          : Path | str,
    path_segmentations_reference: Path | str,
    filename: str, category=None, load_json=False,
) -> Mapping[int, Metrics]:
    """
    """
    metrics = {}
    print(f'Processing {filename} in category {category}')

    mesh = trimesh.load(f'{path_meshes}/{filename}.off')

    if load_json:
        segmentation = seg_from_face2label(f'{path_segmentations}/{filename}_face2label.json')
    else:
        with open(f'{path_segmentations}/{filename}.seg', 'r') as f:
            segmentation = np.array([int(x) for x in f.readlines()], dtype=np.uint32)
        
    bench_dir = Path(f'{path_segmentations_reference}/{filename}')
    # Compute average metrics over all human segmentations
    for bench_path in bench_dir.iterdir():
        with open(bench_path, 'r') as f:
            bench = np.array([int(x) for x in f.readlines()], dtype=np.uint32)
            metric = compute_metrics(mesh, segmentation, bench)
            assert (err := metric.check_bounds()) is None, (metric, filename, err)
            metrics.setdefault(category, []).append(metric)
    return metrics


def benchmark_dataset_coseg_one(
    path_meshes                 : Path | str,
    path_segmentations          : Path | str,
    path_segmentations_reference: Path | str,
    filename: str, category=None
) -> Mapping[int, Metrics]:
    """
    """
    metrics = {}
    print(f'Processing {filename} in category {category}')

    mesh = trimesh.load(f'{path_meshes}/{filename}.off')

    segmentation = seg_from_face2label(f'{path_segmentations}/{filename}/{filename}_face2label.json')

    with open(f'{path_segmentations_reference}/{filename}.seg', 'r') as f:
        bench = np.array([int(x) for x in f.readlines()], dtype=np.uint32)
        metric = compute_metrics(mesh, segmentation, bench)
        assert (err := metric.check_bounds()) is None, (metric, filename, err)
        metrics.setdefault(category, []).append(metric)
    return metrics
    

def benchmark_dataset_princeton(
    path_meshes                 : Path | str,
    path_segmentations          : Path | str,
    path_segmentations_reference: Path | str,
    load_json=False,
) -> Mapping[int, Metrics]:
    """
    """
    extract_category = lambda i: (i - 1) // 20 + 1

    pool = mp.Pool(mp.cpu_count())
    chunks = [
        (path_meshes, path_segmentations, path_segmentations_reference, i, extract_category(i), load_json)
        for i in range(1, 401) if extract_category(i) not in [13] #, 4, 8, 14, 17]
    ]
    metrics_list = pool.starmap(benchmark_dataset_princeton_one, chunks)
    metrics = {}
    for metrics_one in metrics_list:
        if metrics_one is None:
            continue
        for k, v in metrics_one.items():
            metrics.setdefault(k, []).extend(v)
    return {
        'averages': Metrics.average([m for v in metrics.values() for m in v]),
        'averages_by_category': {k: Metrics.average(v) for k, v in metrics.items()}
    }


def benchmark_dataset_coseg(
    path_meshes                 : Path | str,
    path_segmentations          : Path | str,
    path_segmentations_reference: Path | str,
) -> Mapping[int, Metrics]:
    """
    """
    chunks = []
    categories = ['candelabra', 'chairs', 'fourleg', 'goblets', 'guitars', 'irons', 'lamps', 'vases']
    for cat in categories:
        cat_path_meshes                  = f'{path_meshes}/{cat}'
        cat_path_segmentations           = f'{path_segmentations}/{cat}'
        cat_path_segmentations_reference = f'{path_segmentations_reference}/{cat}_gt'
        filenames = glob.glob(f'{cat_path_meshes}/*.off')
        chunks.extend([
            (cat_path_meshes, cat_path_segmentations, cat_path_segmentations_reference, i, cat)
            for i in [int(Path(f).stem) for f in filenames]
        ])
    pool = mp.Pool(mp.cpu_count())
    metrics_list = pool.starmap(benchmark_dataset_coseg_one, chunks)
    metrics = {}
    for metrics_one in metrics_list:
        for k, v in metrics_one.items():
            metrics.setdefault(k, []).extend(v)
    return {
        'averages': Metrics.average([m for v in metrics.values() for m in v]),
        'averages_by_category': {k: Metrics.average(v) for k, v in metrics.items()}
    }


if __name__ == "__main__":
    '''
    metrics1 = benchmark_dataset_princeton(
        path_meshes='/home/ubuntu/data/MeshsegBenchmark-1.0/data/off',
        path_segmentations='/home/ubuntu/data/MeshsegBenchmark-1.0/data/seg/ShapeDiam',
        path_segmentations_reference='/home/ubuntu/data/MeshsegBenchmark-1.0/data/seg/Benchmark',
    )
    metrics2 = benchmark_dataset_princeton(
        path_meshes='/home/ubuntu/data/MeshsegBenchmark-1.0/data/off',
        path_segmentations ='/home/ubuntu/meshseg/tests/mesh_segmentation_output_princeton-dynamic-0.35-6-0.5-fast-1-15',
        path_segmentations_reference='/home/ubuntu/data/MeshsegBenchmark-1.0/data/seg/Benchmark',
        load_json=True
    )
    print(metrics1)
    print(metrics2)
    '''

    metrics1 = benchmark_dataset_coseg(
        path_meshes='/home/ubuntu/data/coseg',
        path_segmentations='/home/ubuntu/meshseg/tests/mesh_segmentation_output_coseg_shape_diameter_function',
        path_segmentations_reference='/home/ubuntu/data/coseg',
    )
    metrics2 = benchmark_dataset_coseg(
        path_meshes='/home/ubuntu/data/coseg',
        path_segmentations='/home/ubuntu/meshseg/tests/mesh_segmentation_output_coseg-dynamic-0.050-6-0.5',
        path_segmentations_reference='/home/ubuntu/data/coseg',
    )
    print(metrics1)
    print(metrics2)