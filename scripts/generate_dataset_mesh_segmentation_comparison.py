import argparse
import glob
import os
import json
import re
from pathlib import Path
import multiprocessing as mp

import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm

from samesh.renderer.renderer_animations import images2gif


def combine(filenames: list[str], output: Path | str, fps=30):
    """
    """
    image_sequences = []
    for filename in tqdm(filenames):
        sequence = []
        for frame in ImageSequence.Iterator(Image.open(filename)):
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            sequence.append(np.array(frame))
        image_sequences.append(sequence)

    combined = []
    for i in range(len(image_sequences[0])):
        combined_frame = np.concatenate([x[i] for x in image_sequences], axis=1)
        combined.append(Image.fromarray(combined_frame))
    images2gif(combined, path=output, duration=len(combined) / fps)


def combine_renders(idirs: list[Path | str], odir: Path | str, shuffle=True):
    """
    """
    filenames_accum = []
    for idir in idirs:
        filenames = sorted(list(glob.glob(f'{idir}/*.gif')))
        filenames_accum.append(filenames)
    filenames_combined = [
        [filenames[i] for filenames in filenames_accum]
        for i in range(len(filenames_accum[0]))
    ]
    
    pattern = re.compile('.*(shape_diameter_function|sdf|matte|norm|combined).*')
    chunks = []
    os.makedirs(odir, exist_ok=True)
    metadata = []
    for filenames in filenames_combined:
        if shuffle:
            indices = np.random.permutation(len(filenames))
            split2index = {}
            for i, index in enumerate(indices):
                split = pattern.match(filenames[index]).group(1)
                split2index[split] = i
            metadata.append(split2index)
            filenames = [filenames[index] for index in indices]
        else:
            metadata.append({
                filename: i for i, filename in enumerate(filenames)
            })
        output_filename = odir / f'{Path(filenames[0]).stem}.gif'
        chunks.append((filenames, output_filename))
    
    with open(odir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(combine, chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert mesh formats to common format'
    )
    parser.add_argument(
        '-id', '--idir', required=True, nargs='+',
        help='List of dirs containing rendererd videos of mesh segmentations'
    )
    parser.add_argument(
        '-od', '--odir', type=str, required=True, 
        help='Output directory of combined mesh segmentation comparison'
    )
    args = parser.parse_args()

    combine_renders(list(map(Path, args.idir)), Path(args.odir), shuffle=True)