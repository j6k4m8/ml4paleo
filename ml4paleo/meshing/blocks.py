"""
Meshes of a segmentation, made block by block and joined per class.

Each block is meshed with one extra voxel (at mesh resolution) on its high
sides, so neighboring blocks' surfaces meet without gaps or overlaps, and
with a plane of nothing beyond the volume's own faces, so surfaces close
there. Vertices come out in (x, y, z) order, in level-0 voxel units (voxel
corners: voxel i spans [i, i + 1)), with triangles wound so normals point
out; `join` welds the blocks' pieces into one mesh per class and scales it
to physical units.

Downsampling (#22, #24) meshes at 1/d resolution, keeping a coarse voxel if
any of its fine voxels is the class ("any", which keeps thin structures) or
if most are ("majority", smoother).
"""

import itertools
import json
import struct
from collections.abc import Sequence
from typing import Literal

import numpy as np

Box = tuple[int, int, int, int, int, int]
Method = Literal["any", "majority"]


def mesh_blocks(shape_zyx: Sequence[int], block: int) -> list[Box]:
    starts = [range(0, int(n), block) for n in shape_zyx]
    return [
        (
            z,
            y,
            x,
            min(z + block, shape_zyx[0]),
            min(y + block, shape_zyx[1]),
            min(x + block, shape_zyx[2]),
        )
        for z, y, x in itertools.product(*starts)
    ]


def read_box(box: Box, shape_zyx: Sequence[int], downsample: int) -> Box:
    """The region to read for a block: the block plus `downsample` voxels on its high sides."""
    return (
        box[0],
        box[1],
        box[2],
        min(box[3] + downsample, shape_zyx[0]),
        min(box[4] + downsample, shape_zyx[1]),
        min(box[5] + downsample, shape_zyx[2]),
    )


def _downsample(mask: np.ndarray, d: int, method: Method) -> np.ndarray:
    if d == 1:
        return mask
    padded_shape = [-(-n // d) * d for n in mask.shape]
    padded = np.zeros(padded_shape, dtype=np.uint8)
    padded[tuple(slice(0, n) for n in mask.shape)] = mask
    blocks = padded.reshape(
        padded_shape[0] // d, d, padded_shape[1] // d, d, padded_shape[2] // d, d
    )
    counts = blocks.sum(axis=(1, 3, 5), dtype=np.int32)
    if method == "any":
        return counts > 0
    return counts * 2 > d**3


def mesh_block(
    classes: np.ndarray,
    box: Box,
    shape_zyx: Sequence[int],
    values: Sequence[int],
    downsample: int = 1,
    method: Method = "any",
    max_error: float = 0.0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Mesh one block. `classes` holds the region `read_box` names. Returns,
    per class value present, (vertices, faces): float32 (x, y, z) level-0
    voxel coordinates and int32 triangles.

    With `max_error` above 0, surfaces are simplified as far as they can be
    without moving more than that many (meshed, so coarse) voxels. Vertices
    on the block's faces stay put, so blocks still join.
    """
    from zmesh import Mesher  # pyright: ignore[reportAttributeAccessIssue]

    d = downsample
    lo = [box[a] // d for a in range(3)]
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for value in values:
        mask = _downsample(classes == value, d, method)
        if not mask.any():
            continue
        # A plane of nothing beyond the volume's faces closes surfaces there.
        before = [1 if box[a] == 0 else 0 for a in range(3)]
        after = [1 if box[a + 3] + d >= shape_zyx[a] else 0 for a in range(3)]
        # Without an extra voxel inside the volume (its last block), keep
        # only the block's own coarse voxels plus the closing plane.
        own = [
            -(-(box[a + 3] - box[a]) // d) + (0 if after[a] else 1) for a in range(3)
        ]
        mask = mask[tuple(slice(0, n) for n in own)]
        padded = np.pad(mask.astype(np.uint8), list(zip(before, after, strict=True)))
        mesher = Mesher((1, 1, 1))
        mesher.mesh(padded, close=False)
        if 1 not in mesher.ids():
            continue
        mesh = mesher.get(
            1,
            normals=False,
            reduction_factor=100 if max_error > 0 else 0,
            max_error=max_error or None,
        )
        mesher.clear()
        if len(mesh.faces) == 0:
            continue
        # zmesh centers voxel i at i, so its faces sit half a voxel before
        # the corners we count from.
        zyx = mesh.vertices.astype(np.float64) + 0.5 - np.array(before) + np.array(lo)
        vertices = (zyx[:, ::-1] * d).astype(np.float32)
        # Reversing the axes mirrors the mesh; reverse the winding to match.
        out[int(value)] = (vertices, mesh.faces[:, ::-1].astype(np.int32))
    return out


def join(
    pieces: Sequence[tuple[np.ndarray, np.ndarray]], voxel_size_xyz: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """One mesh from blocks' pieces: shared vertices welded, scaled to physical units."""
    if not pieces:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    vertices = np.concatenate([v for v, _ in pieces])
    offsets = np.cumsum([0] + [len(v) for v, _ in pieces[:-1]])
    faces = np.concatenate([f + o for (_, f), o in zip(pieces, offsets, strict=True)])
    # Pieces meet on shared planes, with identical vertices there.
    unique, inverse = np.unique(np.round(vertices, 4), axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    scaled = (unique * np.asarray(voxel_size_xyz, dtype=np.float64)).astype(np.float32)
    return scaled, faces[keep].astype(np.int32)


def to_stl(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """A binary STL."""
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    record = np.dtype(
        [("normal", "<f4", 3), ("points", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    data = np.zeros(len(faces), dtype=record)
    data["normal"] = normals
    data["points"] = triangles
    header = b"ml4paleo mesh".ljust(80, b" ")
    return header + struct.pack("<I", len(faces)) + data.tobytes()


def to_obj(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    lines = [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return ("\n".join(lines) + "\n").encode()


def to_glb(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """A binary glTF 2.0 file with one mesh."""
    positions = np.ascontiguousarray(vertices, dtype="<f4").tobytes()
    indices = np.ascontiguousarray(faces, dtype="<u4").tobytes()
    gltf = {
        "asset": {"version": "2.0", "generator": "ml4paleo"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(len(vertices)),
                "type": "VEC3",
                "min": [float(v) for v in vertices.min(axis=0)]
                if len(vertices)
                else [0, 0, 0],
                "max": [float(v) for v in vertices.max(axis=0)]
                if len(vertices)
                else [0, 0, 0],
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": int(faces.size),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(positions),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(indices),
                "target": 34963,
            },
        ],
        "buffers": [{"byteLength": len(positions) + len(indices)}],
    }
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary = positions + indices
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(text), b"JSON")
        + text
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )
