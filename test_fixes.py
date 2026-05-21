"""
Tests for SAMesh fixes.

Run with: PYTHONPATH=src python test_fixes.py
Requires: trimesh, numpy, torch, torchtyping, omegaconf
"""
from __future__ import annotations

import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from trimesh.base import Trimesh

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from samesh.utils.mesh import duplicate_verts  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_triangle_mesh(n: int = 3) -> Trimesh:
    """Build a tiny mesh with `n` independent triangles."""
    verts = np.zeros((3 * n, 3), dtype=np.float64)
    faces = np.zeros((n, 3), dtype=np.int64)
    for i in range(n):
        verts[3 * i + 0] = [i, 0, 0]
        verts[3 * i + 1] = [i + 1, 0, 0]
        verts[3 * i + 2] = [i, 1, 0]
        faces[i] = [3 * i + 0, 3 * i + 1, 3 * i + 2]
    return Trimesh(vertices=verts, faces=faces, process=False)


def _color_visuals_mesh() -> Trimesh:
    mesh = _unit_triangle_mesh(2)
    mesh.visual.face_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8
    )
    return mesh


def _texture_visuals_mesh() -> Trimesh:
    mesh = _unit_triangle_mesh(2)
    uv = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float64)
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    pil = trimesh.visual.material.SimpleMaterial(image=image)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=pil)
    return mesh


def _broken_visuals_mesh() -> Trimesh:
    """Mesh whose .visual raises when face_colors is accessed."""
    mesh = _unit_triangle_mesh(2)

    class _BadVisual:
        @property
        def face_colors(self):
            raise AttributeError("no face_colors here")

    mesh.visual = _BadVisual()
    return mesh


def _quad_strip_with_isolated(n_quads: int = 5, n_isolated: int = 5) -> Trimesh:
    """Triangulated quad strip + isolated triangles with no shared vertices."""
    n_cols = n_quads + 1
    strip_verts = []
    for col in range(n_cols):
        strip_verts.append([col, 0.0, 0.0])
        strip_verts.append([col, 1.0, 0.0])
    strip_verts = np.array(strip_verts, dtype=np.float64)

    strip_faces = []
    for q in range(n_quads):
        bl, tl = 2 * q, 2 * q + 1
        br, tr = 2 * (q + 1), 2 * (q + 1) + 1
        strip_faces.append([bl, br, tl])
        strip_faces.append([tl, br, tr])
    strip_faces = np.array(strip_faces, dtype=np.int64)

    iso_verts = []
    iso_faces = []
    base = len(strip_verts)
    for i in range(n_isolated):
        x = 100 + 10 * i
        iso_verts.append([x, 0, 0])
        iso_verts.append([x + 1, 0, 0])
        iso_verts.append([x, 1, 0])
        iso_faces.append([base + 3 * i + 0, base + 3 * i + 1, base + 3 * i + 2])
    iso_verts = np.array(iso_verts, dtype=np.float64)
    iso_faces = np.array(iso_faces, dtype=np.int64)

    verts = np.concatenate([strip_verts, iso_verts], axis=0)
    faces = np.concatenate([strip_faces, iso_faces], axis=0)
    return Trimesh(vertices=verts, faces=faces, process=False)


def _label_components(mesh_graph, num_faces, face2label):
    """Mirror of SamModelMesh.label_components for testing without heavy deps."""
    components = []
    visited = set()

    def dfs(source):
        stack = [source]
        components.append({source})
        visited.add(source)
        while stack:
            node = stack.pop()
            for adj in mesh_graph[node]:
                if (
                    adj not in visited
                    and adj in face2label
                    and face2label[adj] == face2label[node]
                ):
                    stack.append(adj)
                    components[-1].add(adj)
                    visited.add(adj)

    for face in range(num_faces):
        if face not in visited and face in face2label:
            dfs(face)
    return components


def _run_split(mesh: Trimesh, face2label: dict, min_component_size: int) -> dict:
    """Mirror of SamModelMesh.split (patched) for testing without heavy deps."""
    edges = trimesh.graph.face_adjacency(mesh=mesh)
    mesh_graph = defaultdict(set)
    for a, b in edges:
        mesh_graph[int(a)].add(int(b))
        mesh_graph[int(b)].add(int(a))

    face2label = dict(face2label)
    components = _label_components(mesh_graph, len(mesh.faces), face2label)
    labels_seen = set()
    labels_curr = max(face2label.values()) + 1
    for comp in components:
        face = next(iter(comp))
        label = face2label[face]
        if (label == 0 or label in labels_seen) and len(comp) >= min_component_size:
            face2label.update({f: labels_curr for f in comp})
            labels_curr += 1
        labels_seen.add(label)
    return face2label


# ---------------------------------------------------------------------------
# Face ID encode/decode helpers (mirrors renderer.render_faces)
# ---------------------------------------------------------------------------

def encode_face_id(face_id: int) -> tuple[int, int, int]:
    return (
        (face_id >> 16) & 0xFF,
        (face_id >> 8) & 0xFF,
        face_id & 0xFF,
    )


def decode_face_ids(rgb: np.ndarray, num_faces: int) -> np.ndarray:
    rgb = rgb.astype(np.int32)
    faces = rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]
    faces[faces >= num_faces] = -1
    return faces


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_duplicate_verts_color_visuals():
    mesh = _color_visuals_mesh()
    out = duplicate_verts(mesh)
    assert out.faces.shape == (2, 3)
    assert out.vertices.shape == (6, 3)


def test_duplicate_verts_texture_visuals():
    mesh = _texture_visuals_mesh()
    out = duplicate_verts(mesh)
    assert out.faces.shape == (2, 3)
    assert out.vertices.shape == (6, 3)


def test_duplicate_verts_broken_visuals():
    mesh = _broken_visuals_mesh()
    out = duplicate_verts(mesh)
    assert out.faces.shape == (2, 3)
    assert out.vertices.shape == (6, 3)


def test_face_id_roundtrip():
    ids = [0, 1, 255, 256, 65535, 65536, 16_777_213, 16_777_214]
    num_faces = 16_777_215
    rgb = np.array([encode_face_id(i) for i in ids], dtype=np.uint8).reshape(-1, 1, 3)
    decoded = decode_face_ids(rgb, num_faces=num_faces).reshape(-1)
    assert decoded.tolist() == ids, decoded.tolist()


def test_white_background_decodes_to_minus_one():
    num_faces = 34_000
    rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
    decoded = decode_face_ids(rgb, num_faces=num_faces)
    assert decoded[0, 0] == -1


def test_antialiased_pixel_clamped():
    num_faces = 34_000
    face_rgb = np.array(encode_face_id(0), dtype=np.float64)
    bg_rgb = np.array([255, 255, 255], dtype=np.float64)
    blended = np.round(0.5 * face_rgb + 0.5 * bg_rgb).astype(np.uint8)
    raw = blended.astype(np.int32)
    decoded_raw = int(raw[0]) * 65536 + int(raw[1]) * 256 + int(raw[2])
    assert decoded_raw >= num_faces, decoded_raw
    rgb = blended.reshape(1, 1, 3)
    decoded = decode_face_ids(rgb, num_faces=num_faces)
    assert decoded[0, 0] == -1, decoded[0, 0]


def test_pyopengl_platform_not_forced_on_non_linux():
    src = (SRC_ROOT / "samesh" / "renderer" / "renderer.py").read_text()
    head = "\n".join(src.splitlines()[:10])
    assert "if _platform.system() == 'Linux'" in head, head
    assert "os.environ['PYOPENGL_PLATFORM'] = 'egl'" not in head, head


def test_split_without_threshold_creates_new_labels():
    mesh = _quad_strip_with_isolated(n_quads=5, n_isolated=5)
    face2label = {i: 1 for i in range(len(mesh.faces))}
    out = _run_split(mesh, face2label, min_component_size=1)
    n_labels = len(set(out.values()))
    assert n_labels == 6, f"expected 6 labels, got {n_labels}"


def test_split_with_threshold_prevents_isolated_splits():
    mesh = _quad_strip_with_isolated(n_quads=5, n_isolated=5)
    face2label = {i: 1 for i in range(len(mesh.faces))}
    out = _run_split(mesh, face2label, min_component_size=5)
    n_labels = len(set(out.values()))
    assert n_labels == 1, f"expected 1 label, got {n_labels}"


def test_sam_mesh_split_source_uses_min_component_size():
    src = (SRC_ROOT / "samesh" / "models" / "sam_mesh.py").read_text()
    assert "split_min_component_size" in src, src
    assert "len(comp) >= min_component_size" in src, src


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

TESTS = [
    test_duplicate_verts_color_visuals,
    test_duplicate_verts_texture_visuals,
    test_duplicate_verts_broken_visuals,
    test_face_id_roundtrip,
    test_white_background_decodes_to_minus_one,
    test_antialiased_pixel_clamped,
    test_pyopengl_platform_not_forced_on_non_linux,
    test_split_without_threshold_creates_new_labels,
    test_split_with_threshold_prevents_isolated_splits,
    test_sam_mesh_split_source_uses_min_component_size,
]


def main() -> int:
    passed = 0
    failed = []
    for fn in TESTS:
        try:
            fn()
        except Exception:
            failed.append((fn.__name__, traceback.format_exc()))
            print(f"FAIL  {fn.__name__}")
        else:
            passed += 1
            print(f"PASS  {fn.__name__}")

    print(f"\n{passed}/{len(TESTS)} tests passed")
    for name, tb in failed:
        print(f"\n--- {name} ---\n{tb}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
