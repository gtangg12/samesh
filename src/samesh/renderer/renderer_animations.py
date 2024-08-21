import numpy as np
from PIL import Image
from trimesh import Trimesh, Scene
from omegaconf import OmegaConf

from samesh.data.loaders import scene2mesh, read_mesh
from samesh.renderer.renderer import Renderer, render_multiview
from samesh.utils.mesh import duplicate_verts


def images2gif(images: list[Image.Image], path, duration=100, loop=0):
    """
    """
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=loop)


def mesh2gif(
    source: Scene | Trimesh, path: str, fps: int, length: int, size=1024, key='matte', colormap=None, **kwargs
):
    """
    """
    if isinstance(source, Scene):
        source = scene2mesh(source)
    
    renderer = Renderer(OmegaConf.create({'target_dim': (size, size)}))
    renderer.set_object(source)
    renderer.set_camera()

    duration = length / fps
    print(f'Rendering {length} frames at {fps} fps for {duration} s')
    renders = render_multiview(
        renderer, 
        camera_generation_method='swirl', 
        renderer_args=kwargs.pop('renderer_args', {}),
        sampling_args=kwargs.pop('sampling_args', {'n': length, 'radius': 3}),
        lighting_args=kwargs.pop('lighting_args', {}),
    )
    blend = kwargs.pop('blend', 0)
    if key == 'face_colors':
        images = []
        from samesh.renderer.renderer import colormap_faces
        for faces, matte in zip(renders['faces'], renders['matte']):
            image = source.visual.face_colors[faces]
            image = (1 - blend) * image[:, :, :3] + blend * np.array(matte) # blend
            image[faces == -1] = kwargs.pop('background', 0)
            image = Image.fromarray(image.astype(np.uint8))
            images.append(image)

    elif key == 'vertex_colors':
        raise NotImplementedError
    
    else:
        colormap = colormap or (lambda x: x)
        images = [colormap(image) for image in renders[key]]

    images2gif(images, path, duration=duration)


if __name__ == '__main__':
    source = read_mesh('/home/ubuntu/meshseg/tests/mesh_segmentation_output-0.075-3-0.5/basin/basin_segmented.glb', process=False)
    mesh2gif(source, path='segmented.gif', fps=30, length=120, key='face_colors', blend=0.5)