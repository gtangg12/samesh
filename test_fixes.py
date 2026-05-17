"""
Tests for SAMesh fixes.

Run with: PYTHONPATH=src python test_fixes.py
Requires: trimesh, numpy, torch, torchtyping, omegaconf
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

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
