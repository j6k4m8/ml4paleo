"""
Block meshing: exact surfaces in (x, y, z) voxel corners, outward normals,
and watertight joins across blocks.
"""

import json
import struct

import numpy as np
import pytest

from ml4paleo.meshing.blocks import (
    join,
    mesh_block,
    mesh_blocks,
    read_box,
    to_glb,
    to_obj,
    to_stl,
)

pytest.importorskip("zmesh")


def mesh_volume(volume, block, values, downsample=1, method="any", max_error=0.0):
    pieces: dict[int, list] = {}
    for box in mesh_blocks(volume.shape, block):
        rb = read_box(box, volume.shape, downsample)
        region = volume[rb[0] : rb[3], rb[1] : rb[4], rb[2] : rb[5]]
        for value, piece in mesh_block(
            region, box, volume.shape, values, downsample, method, max_error
        ).items():
            pieces.setdefault(value, []).append(piece)
    return {value: join(found, (1.0, 1.0, 1.0)) for value, found in pieces.items()}


def signed_volume(vertices, faces):
    triangles = vertices[faces].astype(np.float64)
    return (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )


def edges_shared_twice(faces):
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return (counts == 2).all()


def test_a_box_meshes_to_its_voxel_faces_in_xyz():
    volume = np.zeros((10, 12, 30), dtype=np.uint8)  # (z, y, x)
    volume[1:5, 2:6, 2:18] = 2
    vertices, faces = mesh_volume(volume, 64, [2])[2]
    assert vertices.min(axis=0).tolist() == [2, 2, 1]  # x, y, z
    assert vertices.max(axis=0).tolist() == [18, 6, 5]
    # Marching cubes cuts the box's corners a little; its volume stays close.
    assert signed_volume(vertices, faces) > 0.85 * 16 * 4 * 4


def test_blocks_join_into_one_closed_surface():
    z, y, x = np.indices((40, 44, 48))
    ball = ((z - 20) ** 2 + (y - 22) ** 2 + (x - 24) ** 2 <= 15**2).astype(np.uint8) * 3
    whole_v, whole_f = mesh_volume(ball, 64, [3])[3]
    v, f = mesh_volume(ball, 16, [3])[3]
    assert edges_shared_twice(f)  # watertight: no holes at block seams
    assert signed_volume(v, f) == pytest.approx(
        signed_volume(whole_v, whole_f), rel=1e-6
    )


def test_simplified_blocks_still_join():
    rng = np.random.default_rng(0)
    z, y, x = np.indices((40, 44, 48))
    ball = (z - 20) ** 2 + (y - 22) ** 2 + (x - 24) ** 2 <= 15**2
    rough = np.where(ball ^ (rng.random(ball.shape) < 0.02), 2, 1).astype(np.uint8)
    full_v, full_f = mesh_volume(rough, 16, [2])[2]
    v, f = mesh_volume(rough, 16, [2], max_error=2)[2]
    assert edges_shared_twice(f)
    assert len(f) < 0.6 * len(full_f)
    assert signed_volume(v, f) == pytest.approx(signed_volume(full_v, full_f), rel=0.02)


def test_objects_touching_the_volume_edges_are_closed():
    volume = np.full((8, 8, 8), 2, dtype=np.uint8)
    v, f = mesh_volume(volume, 4, [2])[2]
    assert edges_shared_twice(f)
    assert v.min(axis=0).tolist() == [0, 0, 0] and v.max(axis=0).tolist() == [8, 8, 8]


def test_downsampling_keeps_thin_parts_with_any():
    volume = np.zeros((16, 16, 16), dtype=np.uint8)
    volume[8, 2:14, 2:14] = 2  # one voxel thick
    assert 2 in mesh_volume(volume, 8, [2], downsample=2, method="any")
    assert 2 not in mesh_volume(volume, 8, [2], downsample=2, method="majority")
    v, f = mesh_volume(volume, 8, [2], downsample=2, method="any")[2]
    assert edges_shared_twice(f)


def test_file_formats():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    stl = to_stl(vertices, faces)
    assert len(stl) == 84 + 50 and struct.unpack("<I", stl[80:84])[0] == 1
    assert to_obj(vertices, faces).decode().splitlines()[-1] == "f 1 2 3"
    glb = to_glb(vertices, faces)
    magic, version, length = struct.unpack("<4sII", glb[:12])
    assert (magic, version, length) == (b"glTF", 2, len(glb))
    size = struct.unpack("<I", glb[12:16])[0]
    gltf = json.loads(glb[20 : 20 + size])
    assert gltf["accessors"][0]["count"] == 3
