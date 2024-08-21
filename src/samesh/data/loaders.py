from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from trimesh.base import Trimesh, ColorVisuals, Scene

from samesh.data.common import NumpyTensor
from samesh.utils.mesh import transform, norm_mesh, norm_scene, handle_pose

from trimesh.scene import transforms


COLORS = {
    'material_diffuse' : np.array([102, 102, 102, 255], dtype=np.uint8),
    'material_ambient' : np.array([ 64,  64,  64, 255], dtype=np.uint8),
    'material_specular': np.array([197, 197, 197, 255], dtype=np.uint8),
}


def remove_texture(source: Trimesh | Scene, material='material_diffuse', visual_kind='face'):
    """
    Remove texture from mesh or scene.
    """
    def assign(visual, color):
        """
        Helper function to assign color to visual given visual kind.
        """
        if visual_kind == 'face':
            visual.face_colors = color
        elif visual_kind == 'vertex':
            visual.vertex_colors = color
        else:
            raise ValueError(f'Invalid visual kind {visual_kind}.')

    if isinstance(source, trimesh.Scene):
        for _, geom in source.geometry.items():
            geom.visual = ColorVisuals()
            assign(geom.visual, COLORS[material])
    else:
        source.visual = ColorVisuals()
        assign(source.visual, COLORS[material])
    return source


def scene2scene_no_transform(scene: Scene) -> Scene:
    """

    NOTE:: in place operation that consumes scene.
    """
    for name, geom in scene.geometry.items():
        if name in scene.graph:
            pose, _ = scene.graph[name]
            pose = handle_pose(pose)
            geom.vertices = transform(pose, geom.vertices)
        scene.graph[name] = np.eye(4)
    return scene


def scene2mesh(scene: Scene, process=True) -> Trimesh:
    """
    """
    if len(scene.geometry) == 0:
        mesh = None  # empty scene
    else:
        data = []
        for name, geom in scene.geometry.items():
            if name in scene.graph:
                pose, _ = scene.graph[name]
                pose = handle_pose(pose)
                vertices = transform(pose, geom.vertices)
            else:
                vertices = geom.vertices
            # process=True removes duplicate vertices (needed for correct face graph), affecting face indices but not faces.shape
            data.append(Trimesh(vertices=vertices, faces=geom.faces, visual=geom.visual, process=process))
        
        mesh = trimesh.util.concatenate(data)
        mesh = Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=mesh.visual, process=process)
    return mesh


def read_mesh(filename: Path, norm=False, process=True) -> Trimesh | None:
    """
    Read/convert a possible scene to mesh. 
    
    If conversion occurs, the returned mesh has only vertex and face data i.e. no texture information.

    NOTE: sometimes process=True does unexpected actions, such as cause face color misalignment with faces
    """    
    source = trimesh.load(filename)

    if isinstance(source, trimesh.Scene):
        mesh = scene2mesh(source, process=process)
    else:
        assert(isinstance(source, trimesh.Trimesh))
        mesh = source
    if norm:
        mesh = norm_mesh(mesh)
    return mesh


def read_scene(filename: Path, norm=False) -> Scene | None:
    """
    """
    source = trimesh.load(filename)
    source = scene2scene_no_transform(source)
    if norm:
        source = norm_scene(source)
    return source


if __name__ == '__main__':
    mesh = read_mesh('/home/ubuntu/meshseg/tests/examples/0ba4ae3aa97b4298866a2903de4fd1e7.glb')
    print(mesh.faces.shape)
    print(mesh.vertices.shape)