import argparse
import glob
import os
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from samesh.data.loaders import read_mesh
from samesh.renderer.renderer_animations import mesh2gif


def convert_mesh2gif_worker(filename: Path | str, args: argparse.Namespace):
    """
    """
    filename_out = Path(args.odir) / Path(filename).name.replace(f'.{args.load_extension}', f'.gif')
    source = read_mesh(filename, process=False)
    mesh2gif(source, filename_out, fps=args.fps, length=args.length, key=args.key, blend=0.5, background=0)


def convert_mesh2gif(args: argparse.Namespace):
    """
    """
    os.makedirs(args.odir, exist_ok=True)

    filenames = glob.glob(f'{args.idir}/*/*_segmented_recolored.{args.load_extension}')
    chunks = [
        (filename, args) for filename in filenames
    ]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(convert_mesh2gif_worker, chunks)
    print(f'Converted {len(filenames)} meshes to gifs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert mesh to rendererd gifs'
    )
    parser.add_argument(
        '-id', '--idir', type=str, help='Path to the directory containing the meshes'
    )
    parser.add_argument(
        '-od', '--odir', type=str, help='Path to the directory containing the rendered gifs'
    )
    parser.add_argument(
        '-le', '--load_extension', type=str, default='glb', help='Extension of the meshes to load'
    )
    parser.add_argument(
        '--fps', type=int, default=30, help='Frames per second'
    )
    parser.add_argument(
        '--length', type=int, default=120, help='Number of frames'
    )
    parser.add_argument(
        '--key', type=str, default='face_colors', help='Key to render'
    )
    args = parser.parse_args()

    convert_mesh2gif(args)

'''
python -m scripts.convert_mesh2gif -id /home/ubuntu/meshseg/tests/mesh_segmentation_output_shape_diameter_function-5-15 -od /home/ubuntu/meshseg/tests/mesh_segmentation_output_shape_diameter_function-5-15-gifs
python -m scripts.convert_mesh2gif -id /home/ubuntu/meshseg/tests/mesh_segmentation_output-dynamic-0.125-6-0.5-sdf/ -od /home/ubuntu/meshseg/tests/mesh_segmentation_output-dynamic-0.125-6-0.5-sdf-gifs
python -m scripts.convert_mesh2gif -id /home/ubuntu/meshseg/tests/mesh_segmentation_output-dynamic-0.125-6-0.5-norm/ -od /home/ubuntu/meshseg/tests/mesh_segmentation_output-dynamic-0.125-6-0.5-norm-gifs
python -m scripts.convert_mesh2gif -id /home/ubuntu/meshseg/tests/mesh_segmentation_output-dynamic-0.125-6-0.5-combined/ -od /home/ubuntu/meshseg/tests/mesh_segmentation_output-dynamic-0.125-6-0.5-combined-gifs
python -m scripts.convert_mesh2gif -id /home/ubuntu/meshseg/tests/mesh_segmentation_output-dynamic-0.125-6-0.5-matte/ -od /home/ubuntu/meshseg/tests/mesh_segmentation_output-dynamic-0.125-6-0.5-matte-gifs
'''