import os

import numpy as np
import pyrender
from trimesh import Trimesh

from samesh.data.common import NumpyTensor

SHADERS_PATH = os.path.join(os.path.dirname(__file__), 'shaders')


class NormalShaderCache:
    """
    """
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        """
        """
        self.program = self.program or pyrender.shader_program.ShaderProgram(
            f'{SHADERS_PATH}/normal.vert', 
            f'{SHADERS_PATH}/normal.frag',
            defines=defines
        )
        return self.program


class BarycentricShaderCache:
    """
    """
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        """
        """
        self.program = self.program or pyrender.shader_program.ShaderProgram(
            f'{SHADERS_PATH}/barycentric.vert', 
            f'{SHADERS_PATH}/barycentric.frag',
            defines=defines
        )
        return self.program


class FaceidShaderCache:
    """
    """
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        """
        """
        self.program = self.program or pyrender.shader_program.ShaderProgram(
            f'{SHADERS_PATH}/faceid.vert', 
            f'{SHADERS_PATH}/faceid.frag',
            defines=defines
        )
        return self.program