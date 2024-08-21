import argparse
import glob
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import trimesh


'''
NOTE Common mesh format:
    - glb format
    - poses embedded in mesh graph
    - textures, if exist, are expressed as uv coordinates
'''

def convert_backflip(filename: str, filename_out: str):
    """
    """
    source = trimesh.load(filename)
    if isinstance(source, trimesh.Scene):
        data = list(source.graph.transforms.edge_data.values())[0]
        name = data['geometry']
        pose = data['matrix']
        data['matrix'] = np.eye(4)
        source.graph[name] = pose
    source.export(filename_out)


def convert_meshseg_benchmark(filename: str, filename_out: str):
    """
    """
    trimesh.load(filename).export(filename_out)


CONVERTERS = {
    'backflip'         : convert_backflip,
    'meshseg_benchmark': convert_meshseg_benchmark
}


def convert_formats(idir: Path, odir: Path, load_extension: str, save_extension='glb', origin='backflip'):
    """
    """
    os.makedirs(odir, exist_ok=True)

    filenames = glob.glob(str(idir) + f'/*.{load_extension}')
    for filename in tqdm(filenames):
        filename_out = odir / Path(filename).name.replace(f'.{load_extension}', f'.{save_extension}')
        if origin not in CONVERTERS:
            raise ValueError(f'Conversion from data source {origin} not supported')
        CONVERTERS[origin](filename, filename_out)
    
    print(f'Converted {len(filenames)} meshes from {origin} to {save_extension}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert mesh formats to common format'
    )
    parser.add_argument(
        '-id', '--idir', type=str, help='Path to the directory containing the meshes'
    )
    parser.add_argument(
        '-od', '--odir', type=str, help='Path to the directory containing the processed meshes'
    )
    parser.add_argument(
        '-le', '--load_extension', type=str, help='Extension of the meshes to load'
    )
    parser.add_argument(
        '-se', '--save_extension', type=str, help='Extension of the meshes to save'
    )
    parser.add_argument(
        '-o', '--origin', type=str, default='backflip', help='Origin of the meshes'
    )
    args = parser.parse_args()

    convert_formats(Path(args.idir), Path(args.odir), args.load_extension, args.save_extension, args.origin)
