import argparse
import glob
import os
from pathlib import Path
import multiprocessing as mp

import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm


def convert_gif2mp4(ifilename: Path | str, ofilename: Path | str, duration=8):
    """
    """
    print(f'Converting {ifilename} to {ofilename}.mp4')

    with Image.open(ifilename) as gif:
        frames = []
        try:
            while True:
                frame = gif.convert('RGB')
                frames.append(np.array(frame))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

    imageio.mimsave(ofilename, frames, format='ffmpeg', fps=len(frames) // duration)


def convert_filenames(idir: Path | str, odir: Path | str):
    """
    """
    os.makedirs(odir, exist_ok=True)

    chunks = []
    filenames = glob.glob(f'{idir}/*.gif')
    for filename in tqdm(filenames):
        ofilename = os.path.join(odir, os.path.basename(filename).replace('.gif', '.mp4'))
        chunks.append((filename, ofilename))

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(convert_gif2mp4, chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert gif to mp4 format'
    )
    parser.add_argument(
        '-id', '--idir', type=str, help='Path to the directory containing the meshes'
    )
    parser.add_argument(
        '-od', '--odir', type=str, help='Path to the directory containing the rendered gifs'
    )
    args = parser.parse_args()
    convert_filenames(args.idir, args.odir)