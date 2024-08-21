import numpy as np
import torch


def golden_ratio():
    return (1 + np.sqrt(5)) / 2


def tetrahedron():
    return np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
    ])


def octohedron():
    return np.array([
        [ 1,  0,  0],
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0],
        [ 0, -1,  0],
    ])


def cube():
    return np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1, -1],
        [ 1, -1, -1],
    ])


def icosahedron():
    phi = golden_ratio()
    return np.array([
        [-1,  phi,  0],
        [-1, -phi,  0],
        [ 1,  phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ]) / np.sqrt(1 + phi ** 2)


def dodecahedron():
    phi = golden_ratio()
    a, b = 1 / phi, 1 / (phi * phi)
    return np.array([
        [-a, -a,  b], [ a, -a,  b], [ a,  a,  b], [-a,  a,  b],
        [-a, -a, -b], [ a, -a, -b], [ a,  a, -b], [-a,  a, -b],
        [ b, -a, -a], [ b,  a, -a], [ b,  a,  a], [ b, -a,  a],
        [-b, -a, -a], [-b,  a, -a], [-b,  a,  a], [-b, -a,  a],
        [-a,  b, -a], [ a,  b, -a], [ a,  b,  a], [-a,  b,  a],
    ]) / np.sqrt(a ** 2 + b ** 2)


def standard(n=8, elevation=15):
    """
    """
    pphi =  elevation * np.pi / 180
    nphi = -elevation * np.pi / 180
    coords = []
    for phi in [pphi, nphi]:
        for theta in np.linspace(0, 2 * np.pi, n, endpoint=False):
            coords.append([
                np.cos(theta) * np.cos(phi),
                np.sin(phi),
                np.sin(theta) * np.cos(phi),
            ])
    coords.append([0,  0,  1])
    coords.append([0,  0, -1])
    return np.array(coords)


def swirl(n=120, cycles=1, elevation_range=(-45, 60)):
    """
    """
    pphi = elevation_range[0] * np.pi / 180
    nphi = elevation_range[1] * np.pi / 180
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = []
    for i, phi in enumerate(np.linspace(pphi, nphi, n)):
        coords.append([
            np.cos(cycles * thetas[i]) * np.cos(phi),
            np.sin(phi),
            np.sin(cycles * thetas[i]) * np.cos(phi),
        ])
    return np.array(coords)