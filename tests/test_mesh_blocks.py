"""
Block meshing: exact surfaces in (x, y, z) voxel corners, outward normals,
and watertight joins across blocks and the sub-boxes that porous blocks are
meshed in, streamed into STL, OBJ, and GLB files.
"""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml4paleo.meshing import blocks
from ml4paleo.meshing.blocks import (
    Join,
    Piece,
    PieceWriter,
    TooDetailed,
    mesh_block,
    mesh_blocks,
    read_box,
    read_pieces,
)

pytest.importorskip("zmesh")


def read_glb(raw: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    (size,) = struct.unpack_from("<I", raw, 12)
    gltf = json.loads(raw[20 : 20 + size])
    count = gltf["accessors"][0]["count"]
    binary = raw[28 + size :]
    vertices = np.frombuffer(binary[: 12 * count], dtype="<f4").reshape(-1, 3)
    faces = np.frombuffer(binary[12 * count :], dtype="<u4").reshape(-1, 3)
    return vertices, faces.astype(np.int64), gltf


def join(per_block, shape, block, voxel_size=(1.0, 1.0, 1.0)):
    """Join each block's pieces, blocks in order: (vertices, faces), or None."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        Join(Path(tmp), shape, block, voxel_size) as joined,
    ):
        for index, pieces in enumerate(per_block):
            for piece in pieces:
                joined.add(piece)
            joined.block_done(index)
        assert joined.pending == 0  # every seam vertex met its neighbors
        joined.finish()
        if not joined.triangles:
            return None
        glb = joined.write_glb(Path(tmp) / "mesh.glb").read_bytes()
    vertices, faces, _ = read_glb(glb)
    return vertices, faces


def mesh_volume(
    volume, block, values, downsample=1, method="any", max_error=0.0, max_faces=None
):
    per_class: dict[int, list] = {value: [] for value in values}
    for box in mesh_blocks(volume.shape, block):
        rb = read_box(box, volume.shape, downsample)
        region = volume[rb[0] : rb[3], rb[1] : rb[4], rb[2] : rb[5]]
        found: dict[int, list] = {value: [] for value in values}
        for value, piece in mesh_block(
            region, box, volume.shape, values, downsample, method, max_error, max_faces
        ):
            found[value].append(piece)
        for value in values:
            per_class[value].append(found[value])
    meshes = {
        value: join(per_block, volume.shape, block)
        for value, per_block in per_class.items()
    }
    return {value: mesh for value, mesh in meshes.items() if mesh is not None}


def mesh_whole(volume, values, downsample=1, method="any"):
    return mesh_volume(volume, max(volume.shape), values, downsample, method)


def signed_volume(vertices, faces):
    triangles = vertices[faces].astype(np.float64)
    return (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )


def edges_shared_twice(faces):
    """Closed, and consistently wound: each edge once each way."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(directed, axis=0, return_counts=True)
    _, undirected = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return (counts == 1).all() and (undirected == 2).all()


def triangles(vertices, faces):
    """The corners and triangles, whatever order the vertices came in."""
    corners, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    # Start each triangle at its lowest corner, keeping its winding.
    turn = (faces.argmin(axis=1)[:, None] + np.arange(3)) % 3
    faces = np.take_along_axis(faces, turn, axis=1)
    return corners, faces[np.lexsort(faces.T[::-1])]


def blobs(shape, seed=0, sigma=1.5):
    from scipy import ndimage

    noise = ndimage.gaussian_filter(np.random.default_rng(seed).random(shape), sigma)
    volume = np.ones(shape, dtype=np.uint8)
    volume[noise > np.quantile(noise, 0.45)] = 2
    volume[noise > np.quantile(noise, 0.7)] = 3
    return volume


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


@pytest.mark.parametrize("shape", [(17, 23, 31), (20, 20, 20), (33, 18, 40)])
@pytest.mark.parametrize("downsample", [1, 2, 4, 8])
@pytest.mark.parametrize("method", ["any", "majority"])
def test_blocks_of_odd_volumes_mesh_like_the_whole(shape, downsample, method):
    # The last blocks are short, or only part of a coarse voxel deep.
    volume = blobs(shape)
    blocked = mesh_volume(volume, 16, [2, 3], downsample, method)
    whole = mesh_whole(volume, [2, 3], downsample, method)
    assert blocked.keys() == whole.keys()
    for value, (v, f) in blocked.items():
        assert edges_shared_twice(f)
        got, expected = triangles(v, f), triangles(*whole[value])
        assert all(np.array_equal(a, b) for a, b in zip(got, expected, strict=True))


def test_porous_blocks_mesh_in_sub_boxes_that_join(monkeypatch):
    volume = blobs((70, 50, 90), seed=1, sigma=2.0)
    largest = []
    mesh = blocks._mesh

    def measured(padded, *args):
        largest.append(blocks._faces(padded))
        return mesh(padded, *args)

    monkeypatch.setattr(blocks, "_mesh", measured)
    split = mesh_volume(volume, 64, [2, 3], max_faces=5000)
    assert len(largest) > 4 * len(mesh_blocks(volume.shape, 64))
    assert max(largest) <= 5000
    whole = mesh_whole(volume, [2, 3])
    for value, (v, f) in split.items():
        assert edges_shared_twice(f)
        got, expected = triangles(v, f), triangles(*whole[value])
        assert all(np.array_equal(a, b) for a, b in zip(got, expected, strict=True))


def test_too_much_surface_for_the_smallest_sub_boxes_is_refused():
    noise = np.random.default_rng(0).random((32, 32, 32)) < 0.5
    with pytest.raises(TooDetailed):
        mesh_volume(noise.astype(np.uint8) * 2, 32, [2], max_faces=1000)


def test_simplified_blocks_still_join():
    rng = np.random.default_rng(0)
    z, y, x = np.indices((40, 44, 48))
    ball = (z - 20) ** 2 + (y - 22) ** 2 + (x - 24) ** 2 <= 15**2
    rough = np.where(ball ^ (rng.random(ball.shape) < 0.02), 2, 1).astype(np.uint8)
    full_v, full_f = mesh_volume(rough, 16, [2])[2]
    for max_faces in (None, 2000):
        v, f = mesh_volume(rough, 32, [2], max_error=2, max_faces=max_faces)[2]
        assert edges_shared_twice(f)
        assert len(f) < 0.6 * len(full_f)
        assert signed_volume(v, f) == pytest.approx(
            signed_volume(full_v, full_f), rel=0.02
        )


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


def test_pieces_round_trip_through_a_file(tmp_path):
    volume = blobs((20, 20, 20))
    box = (0, 0, 0, 16, 16, 16)
    rb = read_box(box, volume.shape, 1)
    region = volume[rb[0] : rb[3], rb[1] : rb[4], rb[2] : rb[5]]
    pieces = [piece for _, piece in mesh_block(region, box, volume.shape, [2, 3])]
    with PieceWriter(tmp_path / "pieces.npz") as writer:
        for piece in pieces:
            writer.add(piece)
    with (tmp_path / "pieces.npz").open("rb") as file:
        read = list(read_pieces(file))
    assert len(read) == len(pieces) == 2
    for got, sent in zip(read, pieces, strict=True):
        assert all(np.array_equal(a, b) for a, b in zip(got, sent, strict=True))
        assert got.seam.any()  # the block meets its neighbors on its high sides


def test_welding_drops_triangles_it_collapses(tmp_path):
    vertices = np.array(
        [[0, 0, 8.5], [0, 0, 8.5], [1, 0, 8], [0, 1, 8]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    seam = np.array([True, True, False, False])
    with Join(tmp_path, (16, 16, 16), 8, (1.0, 1.0, 1.0)) as joined:
        joined.add(Piece(vertices, faces, seam))
        joined.finish()
    assert (joined.vertices, joined.triangles) == (3, 1)


def test_file_formats(tmp_path):
    vertices = np.array([[0, 0, 0], [1234.5679, 0, 0], [0, 1, 0]], dtype=np.float32)
    piece = Piece(vertices, np.array([[0, 1, 2]], dtype=np.int32), np.zeros(3, bool))
    with Join(tmp_path, (10, 10, 2000), 256, (1.0, 1.0, 1.0)) as joined:
        joined.add(piece)
        stl = joined.finish().read_bytes()
        obj = joined.write_obj(tmp_path / "mesh.obj").read_text().splitlines()
        glb = joined.write_glb(tmp_path / "mesh.glb").read_bytes()
    assert len(stl) == 84 + 50 and struct.unpack("<I", stl[80:84])[0] == 1
    assert obj[1] == "v 1234.56787 0 0"  # float32 exactly, not 1234.57
    assert obj[-1] == "f 1 2 3"
    magic, version, length = struct.unpack("<4sII", glb[:12])
    assert (magic, version, length) == (b"glTF", 2, len(glb))
    got, faces, gltf = read_glb(glb)
    assert np.array_equal(got, vertices) and faces.tolist() == [[0, 1, 2]]
    assert gltf["accessors"][0]["max"] == [float(np.float32(1234.5679)), 1, 0]
