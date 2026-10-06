"""
Meshes of a segmentation, made block by block and joined per class.

Each block is meshed with one extra coarse voxel (a voxel at mesh
resolution) on its high sides, so neighboring blocks' surfaces meet without
gaps or overlaps, and with a plane of nothing beyond the volume's own faces,
so surfaces close there. Vertices come out in (x, y, z) order, in level-0
voxel units (voxel corners: voxel i spans [i, i + 1)), with triangles wound
so normals point out.

zmesh needs memory and time in proportion to a surface's size, and holds
the GIL throughout, so a porous block can take gigabytes and minutes in one
call. A block with more surface than one call should take is meshed in
sub-boxes instead, halved on a power-of-two grid until each is small enough;
sub-boxes meet each other the way blocks do. Each piece marks its vertices
on the planes it shares with other pieces (its seams), and `Join` welds
those while it streams a class's pieces into STL, OBJ, and GLB files, so no
whole class is ever held in memory.

Downsampling (#22, #24) meshes at 1/d resolution, keeping a coarse voxel if
any of its fine voxels is the class ("any", which keeps thin structures) or
if most are ("majority", smoother). A coarse voxel cut by the scan's far
faces counts only its fine voxels inside the scan, and `Join` ends its
surfaces at those faces.
"""

import itertools
import json
import shutil
import struct
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, Literal, NamedTuple

import numpy as np

Box = tuple[int, int, int, int, int, int]
Method = Literal["any", "majority"]

# What zmesh takes per voxel face of surface (a class voxel next to one that
# isn't), with room to spare: peaks of 470 to 1050 bytes were measured on
# porous, noisy, and smooth volumes, and 1 µs a face (12 µs simplifying).
BYTES_PER_FACE = 1200
SECONDS_PER_FACE = 1.5e-6
SECONDS_PER_FACE_SIMPLIFIED = 15e-6
# How long one zmesh call may hold the GIL; the job's heartbeats wait for it.
TARGET_SECONDS = 20
# Sub-boxes stop halving at this many coarse voxels a side.
SMALLEST = 16
# Triangles handled at once while joining, and OBJ lines formatted at once:
# about 10 MB of arrays or strings.
CHUNK = 1 << 16
LINES = 1 << 16
# glTF's unit is the meter.
METERS_PER_UNIT = {
    "meter": 1.0,
    "centimeter": 0.01,
    "millimeter": 0.001,
    "micrometer": 1e-6,
    "nanometer": 1e-9,
}

_STL_HEADER = b"ml4paleo mesh".ljust(80, b" ")
_STL_RECORD = np.dtype(
    [("normal", "<f4", 3), ("points", "<f4", (3, 3)), ("attribute", "<u2")]
)


class TooDetailed(ValueError):
    """
    A surface too large to mesh or store with what the job has.
    """


class Piece(NamedTuple):
    """
    Part of one class's surface, from one block or sub-box.
    """

    # float32 (x, y, z), level-0 voxel corners
    vertices: np.ndarray
    # int32 triangles, wound so normals point out
    faces: np.ndarray
    # Which vertices lie on a plane shared with another piece.
    seam: np.ndarray


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


def face_limit(memory_budget_bytes: int, simplify: bool) -> int:
    """
    The most surface, in voxel faces, to give zmesh at once: what fits in
    half a job's memory and takes about `TARGET_SECONDS`.
    """
    seconds = SECONDS_PER_FACE_SIMPLIFIED if simplify else SECONDS_PER_FACE
    return int(min(memory_budget_bytes / 2 / BYTES_PER_FACE, TARGET_SECONDS / seconds))


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
    # The mask ends only where the scan does, so a coarse voxel cut short
    # holds fewer fine voxels than d³.
    inside = [np.minimum(d, n - np.arange(0, n, d)) for n in mask.shape]
    return counts * 2 > np.einsum("i,j,k->ijk", *inside)


def _faces(mask: np.ndarray) -> int:
    """Voxel faces between the class and the rest, inside `mask`."""
    return sum(int(np.count_nonzero(np.diff(mask, axis=a))) for a in range(3))


def mesh_block(
    classes: np.ndarray,
    box: Box,
    shape_zyx: Sequence[int],
    values: Sequence[int],
    downsample: int = 1,
    method: Method = "any",
    max_error: float = 0.0,
    max_faces: int | None = None,
) -> Iterator[tuple[int, Piece]]:
    """
    Mesh one block. `classes` holds the region `read_box` names. Yields
    each class value present with its pieces: one for the block, or one per
    sub-box when its surface has more than `max_faces` voxel faces. Raises
    `TooDetailed` if even the smallest sub-boxes have more.

    With `max_error` above 0, surfaces are simplified as far as they can be
    without moving more than that many (meshed, so coarse) voxels. Vertices
    on seams stay put, so pieces still join.
    """
    d = downsample
    assert all(box[a] % d == 0 for a in range(3)), "Blocks start on coarse voxels"
    start = [box[a] // d for a in range(3)]
    own = [-(-(box[a + 3] - box[a]) // d) for a in range(3)]
    end = [-(-int(n) // d) for n in shape_zyx]
    for value in values:
        mask = _downsample(classes == value, d, method)
        if not mask.any():
            continue
        for lo, hi, padded in _sub_boxes(mask, start, own, end, max_faces):
            piece = _mesh(padded, lo, hi, end, d, max_error)
            if piece is not None:
                yield int(value), piece


def _sub_boxes(
    mask: np.ndarray,
    start: Sequence[int],
    own: Sequence[int],
    end: Sequence[int],
    max_faces: int | None,
) -> Iterator[tuple[list[int], list[int], np.ndarray]]:
    """
    The block's coarse voxels (`own` of them from `start`) in sub-boxes on a
    power-of-two grid, each halved until it has at most `max_faces` voxel
    faces of surface. Yields each one's first and end coarse voxels and its
    mask ready to mesh: with the next coarse voxel on its high sides, as
    blocks have, and a plane of nothing beyond the volume's faces.
    """
    todo = [((0, 0, 0), 1 << (max(own) - 1).bit_length())]
    while todo:
        corner, size = todo.pop()
        lo = [start[a] + corner[a] for a in range(3)]
        hi = [start[a] + min(corner[a] + size, own[a]) for a in range(3)]
        if any(lo[a] >= hi[a] for a in range(3)):
            continue
        before = [1 if lo[a] == 0 else 0 for a in range(3)]
        after = [1 if hi[a] >= end[a] else 0 for a in range(3)]
        region = mask[
            tuple(
                slice(lo[a] - start[a], hi[a] - start[a] + 1 - after[a])
                for a in range(3)
            )
        ]
        padded = np.pad(region.astype(np.uint8), list(zip(before, after, strict=True)))
        faces = _faces(padded)
        if faces == 0:
            continue
        if max_faces is None or faces <= max_faces:
            yield lo, hi, padded
        elif size <= SMALLEST:
            raise TooDetailed(
                f"{faces} voxel faces of surface in {size}³ coarse voxels, more "
                f"than the {max_faces} that fit"
            )
        else:
            half = size // 2
            todo += [
                (tuple(c + o for c, o in zip(corner, offset, strict=True)), half)
                for offset in itertools.product((0, half), repeat=3)
            ]


def _mesh(
    padded: np.ndarray,
    lo: Sequence[int],
    hi: Sequence[int],
    end: Sequence[int],
    d: int,
    max_error: float,
) -> Piece | None:
    from zmesh import Mesher  # pyright: ignore[reportAttributeAccessIssue]

    mesher = Mesher((1, 1, 1))
    mesher.mesh(padded, close=False)
    if 1 not in mesher.ids():
        return None
    mesh = mesher.get(
        1,
        normals=False,
        reduction_factor=100 if max_error > 0 else 0,
        max_error=max_error or None,
    )
    mesher.clear()
    if len(mesh.faces) == 0:
        return None
    before = [1 if lo[a] == 0 else 0 for a in range(3)]
    # zmesh centers voxel i at i, so its faces sit half a voxel before the
    # corners we count from.
    zyx = mesh.vertices.astype(np.float64) + 0.5 - np.array(before) + np.array(lo)
    # Pieces meet on the planes through the centers of their first coarse
    # voxels and of the next ones past their ends.
    seam = np.zeros(len(zyx), dtype=bool)
    for a in range(3):
        if lo[a] > 0:
            seam |= zyx[:, a] == lo[a] + 0.5
        if hi[a] < end[a]:
            seam |= zyx[:, a] == hi[a] + 0.5
    vertices = (zyx[:, ::-1] * d).astype(np.float32)
    # Reversing the axes mirrors the mesh; reverse the winding to match.
    return Piece(vertices, mesh.faces[:, ::-1].astype(np.int32), seam)


class PieceWriter:
    """
    Pieces written one at a time into one .npz file (`v<i>`, `f<i>`, and
    `s<i>` for piece i), so a block never holds them all.
    """

    def __init__(self, path: Path):
        self._zip = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True)
        self.pieces = 0

    def __enter__(self) -> "PieceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self._zip.close()

    def add(self, piece: Piece) -> None:
        for name, array in zip("vfs", piece, strict=True):
            with self._zip.open(f"{name}{self.pieces}.npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(array))
        self.pieces += 1


def read_pieces(file: IO[bytes]) -> Iterator[Piece]:
    """The pieces a `PieceWriter` wrote, one at a time."""
    with np.load(file) as saved:
        for i in range(len(saved.files) // 3):
            yield Piece(saved[f"v{i}"], saved[f"f{i}"], saved[f"s{i}"])


class Join:
    """
    One class's mesh, welded from its pieces as they come. The binary STL
    is written as it goes, and the vertices and triangles go to files that
    `write_obj` and `write_glb` read back, so memory holds one piece and the
    seam vertices still waiting for a neighbor, never the whole mesh.

    Add pieces block by block, in the order `mesh_blocks` gives, and call
    `block_done` after each block: seam vertices no later block can share
    are forgotten then. Vertices are scaled to physical units, after the
    surface inside coarse voxels cut by the scan's far faces is squeezed
    into their part inside the scan, so it ends at the faces without
    flattening or folding. Triangles that welding collapses are dropped.
    """

    def __init__(
        self,
        directory: Path,
        shape_zyx: Sequence[int],
        block: int,
        voxel_size_xyz: Sequence[float],
        downsample: int = 1,
    ):
        self.directory = Path(directory)
        self.stl_path = self.directory / "mesh.stl"
        self._stl = self.stl_path.open("w+b")
        self._stl.write(_STL_HEADER + struct.pack("<I", 0))
        self._positions = (self.directory / "positions.bin").open("w+b")
        self._indices = (self.directory / "indices.bin").open("w+b")
        extent = np.array(shape_zyx[::-1], dtype=np.float64)
        # Where each axis's last coarse voxel starts, and how much of it is
        # inside the scan (all of it unless the scan ends partway through).
        self._last_start = (np.ceil(extent / downsample) - 1) * downsample
        self._inside = (extent - self._last_start) / downsample
        self._scale = np.asarray(voxel_size_xyz, dtype=np.float64)
        self._block = block
        self._grid = [-(-int(n) // block) for n in shape_zyx]
        # Per block, the seam vertices it is the last to reach: their bytes
        # as float32 (x, y, z), and their index in the mesh.
        self._seams: dict[int, dict[bytes, int]] = {}
        self.pending = 0
        self.vertices = 0
        self.triangles = 0
        self.low = np.full(3, np.inf)
        self.high = np.full(3, -np.inf)

    def __enter__(self) -> "Join":
        return self

    def __exit__(self, *exc) -> None:
        for file in (self._stl, self._positions, self._indices):
            file.close()

    def _last_blocks(self, xyz: np.ndarray) -> list[int]:
        """The last block, in order, that each vertex is in."""
        zyx = np.minimum(xyz[:, ::-1] // self._block, np.array(self._grid) - 1)
        z, y, x = zyx.astype(np.int64).T
        return ((z * self._grid[1] + y) * self._grid[2] + x).tolist()

    def add(self, piece: Piece) -> None:
        vertices, faces, seam = piece
        index = np.empty(len(vertices), dtype=np.int64)
        plain = np.flatnonzero(~seam)
        index[plain] = self.vertices + np.arange(len(plain))
        count = self.vertices + len(plain)
        shared = np.flatnonzero(seam)
        keys = np.ascontiguousarray(vertices[shared], dtype="<f4").tobytes()
        added = []
        for n, (i, last) in enumerate(
            zip(shared.tolist(), self._last_blocks(vertices[shared]), strict=True)
        ):
            known = self._seams.setdefault(last, {})
            key = keys[12 * n : 12 * n + 12]
            if (j := known.get(key)) is None:
                j = known[key] = count
                count += 1
                added.append(i)
            index[i] = j
        self.pending += len(added)
        physical = np.where(
            vertices > self._last_start,
            self._last_start + (vertices - self._last_start) * self._inside,
            vertices,
        )
        physical *= self._scale
        physical = physical.astype("<f4")
        fresh = physical[np.concatenate([plain, np.array(added, dtype=np.int64)])]
        if len(fresh):
            self._positions.write(fresh)
            self.low = np.minimum(self.low, fresh.min(axis=0))
            self.high = np.maximum(self.high, fresh.max(axis=0))
        self.vertices = count
        for first in range(0, len(faces), CHUNK):
            local = faces[first : first + CHUNK]
            joined = index[local]
            keep = (
                (joined[:, 0] != joined[:, 1])
                & (joined[:, 1] != joined[:, 2])
                & (joined[:, 0] != joined[:, 2])
            )
            self._indices.write(joined[keep].astype("<u4"))
            self._stl.write(_stl_records(physical[local[keep]]))
            self.triangles += int(keep.sum())

    def block_done(self, block: int) -> None:
        self.pending -= len(self._seams.pop(block, {}))

    def finish(self) -> Path:
        """Finish the STL and return its path."""
        self._stl.seek(80)
        self._stl.write(struct.pack("<I", self.triangles))
        for file in (self._stl, self._positions, self._indices):
            file.flush()
        return self.stl_path

    def _rows(self, file: IO[bytes], dtype: str) -> Iterator[np.ndarray]:
        file.seek(0)
        while chunk := file.read(LINES * 12):
            yield np.frombuffer(chunk, dtype=dtype).reshape(-1, 3)

    def write_obj(self, path: Path) -> Path:
        with path.open("w", encoding="ascii", newline="\n") as out:
            for rows in self._rows(self._positions, "<f4"):
                out.write(
                    "".join(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in rows.tolist())
                )
            for rows in self._rows(self._indices, "<u4"):
                out.write(
                    "".join(
                        f"f {a} {b} {c}\n"
                        for a, b, c in (rows.astype(np.int64) + 1).tolist()
                    )
                )
        return path

    def write_glb(self, path: Path, unit: str | None) -> Path:
        """
        A binary glTF 2.0 file with one mesh. The vertices stay in `unit`;
        the node scales them to meters, as glTF expects, when it is a length
        (meshes in voxels are left as they are).
        """
        positions, indices = 12 * self.vertices, 12 * self.triangles
        node: dict = {"mesh": 0}
        if unit in METERS_PER_UNIT and METERS_PER_UNIT[unit] != 1:
            node["scale"] = [METERS_PER_UNIT[unit]] * 3
        gltf = {
            "asset": {"version": "2.0", "generator": "ml4paleo"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [node],
            "meshes": [
                {
                    "primitives": [
                        {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
                    ]
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": self.vertices,
                    "type": "VEC3",
                    "min": [float(v) for v in self.low] if self.vertices else [0, 0, 0],
                    "max": [float(v) for v in self.high]
                    if self.vertices
                    else [0, 0, 0],
                },
                {
                    "bufferView": 1,
                    "componentType": 5125,
                    "count": 3 * self.triangles,
                    "type": "SCALAR",
                },
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": positions,
                    "target": 34962,
                },
                {
                    "buffer": 0,
                    "byteOffset": positions,
                    "byteLength": indices,
                    "target": 34963,
                },
            ],
            "buffers": [{"byteLength": positions + indices}],
        }
        text = json.dumps(gltf, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        # Both buffers are whole float32 and uint32 values, so 4-byte aligned.
        total = 12 + 8 + len(text) + 8 + positions + indices
        if total >= 2**32:
            raise TooDetailed(
                f"A GLB file holds at most 4 GiB; this one is {total} bytes"
            )
        with path.open("wb") as out:
            out.write(struct.pack("<4sII", b"glTF", 2, total))
            out.write(struct.pack("<I4s", len(text), b"JSON") + text)
            out.write(struct.pack("<I4s", positions + indices, b"BIN\0"))
            for file in (self._positions, self._indices):
                file.seek(0)
                shutil.copyfileobj(file, out, 1024 * 1024)
        return path


def _stl_records(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(len(triangles), dtype=_STL_RECORD)
    records["normal"] = normals
    records["points"] = triangles
    return records
