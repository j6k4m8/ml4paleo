"""Ephemeral, bounded zmesh previews. Never annotations, artifacts, or training inputs.

zmesh holds the GIL: use a separate process, not the API's event loop or a thread.
Wire input: uint32 LE shape_zyx[3], chunk_count, then per chunk origin_zyx[3],
shape_zyx[3], and raw uint8 C-order labels. Output: float32 LE records containing
class value followed by three xyz vertices (in coarse voxel-corner coordinates).
"""

import asyncio
import hashlib
import multiprocessing
import struct
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ml4paleo.meshing.blocks import TooDetailed, mesh_block

MAX_INPUT = 4 * 1024 * 1024 + 16 * 24 + 16
MAX_TRIANGLES = 120_000
MAX_CHUNK_TRIANGLES = MAX_TRIANGLES // 16
MAX_SURFACE = 60_000
CACHE_BYTES = 16 * 1024 * 1024


def build_preview(
    raw: bytes, target: tuple[int, int, int] | None = None, downsample: int = 1
) -> bytes:
    if downsample not in (1, 2, 4, 8, 16):
        raise ValueError("Invalid 3D preview detail.")
    if len(raw) < 16 or len(raw) > MAX_INPUT:
        raise ValueError("Invalid 3D preview size.")
    *shape, count = struct.unpack_from("<4I", raw)
    if not 1 <= count <= 16 or any(n < 1 or n > 2**24 for n in shape):
        raise ValueError("Invalid 3D preview dimensions.")
    chunks: dict[tuple[int, int, int], np.ndarray] = {}
    offset = 16
    for _ in range(count):
        if offset + 24 > len(raw):
            raise ValueError("Incomplete 3D preview.")
        z, y, x, nz, ny, nx = struct.unpack_from("<6I", raw, offset)
        offset += 24
        origin, size = (z, y, x), (nz, ny, nx)
        if (
            any(
                o % 64 or n < 1 or n > 64 or o + n > s
                for o, n, s in zip(origin, size, shape, strict=True)
            )
            or origin in chunks
        ):
            raise ValueError("Invalid 3D preview chunk.")
        length = nz * ny * nx
        if offset + length > len(raw):
            raise ValueError("Incomplete 3D preview labels.")
        data = np.frombuffer(raw, dtype=np.uint8, count=length, offset=offset).reshape(
            size
        )
        chunks[origin] = np.where((data > 1) & (data < 255), data, 0).astype(np.uint8)
        offset += length
    if offset != len(raw):
        raise ValueError("Unexpected 3D preview bytes.")
    if target is not None and target not in chunks:
        raise ValueError("Missing 3D preview chunk.")

    pieces = []
    triangles = 0
    for origin, data in chunks.items():
        if target is not None and origin != target:
            continue
        # Positive one-cell halo (including edge/corner neighbors) gives adjacent
        # marching-cubes pieces identical seam vertices, with no internal caps.
        end = tuple(o + n for o, n in zip(origin, data.shape, strict=True))
        region_shape = tuple(
            min(n + downsample, s - o)
            for n, s, o in zip(data.shape, shape, origin, strict=True)
        )
        region = np.zeros(region_shape, dtype=np.uint8)
        for neighbor, values in chunks.items():
            lo = tuple(max(o, p) for o, p in zip(origin, neighbor, strict=True))
            hi = tuple(
                min(o + n, p + m)
                for o, n, p, m in zip(
                    origin, region_shape, neighbor, values.shape, strict=True
                )
            )
            if any(a >= b for a, b in zip(lo, hi, strict=True)):
                continue
            region[
                tuple(
                    slice(a - o, b - o) for a, b, o in zip(lo, hi, origin, strict=True)
                )
            ] = values[
                tuple(
                    slice(a - p, b - p)
                    for a, b, p in zip(lo, hi, neighbor, strict=True)
                )
            ]
        # mesh_block subdivides complex regions BEFORE native meshing, so no
        # zmesh call exceeds the surface budget. Decimation keeps seam vertices.
        values = [int(v) for v in np.unique(region) if v > 1]
        for value, piece in mesh_block(
            region,
            (*origin, *end),
            shape,
            values,
            downsample=downsample,
            method="majority",
            max_error=0.5,
            max_faces=MAX_SURFACE,
        ):
            triangles += len(piece.faces)
            if triangles > (
                MAX_CHUNK_TRIANGLES if target is not None else MAX_TRIANGLES
            ):
                raise TooDetailed("Use a coarser 3D preview.")
            packed = np.empty((len(piece.faces), 10), dtype="<f4")
            packed[:, 0] = value
            vertices = piece.vertices
            if downsample > 1:
                # Squeeze partial edge cells, as the exported mesh joiner does.
                extent = np.asarray(shape[::-1])
                last = (np.ceil(extent / downsample) - 1) * downsample
                inside = (extent - last) / downsample
                vertices = np.where(
                    vertices > last, last + (vertices - last) * inside, vertices
                )
            vertices = np.clip(vertices, 0, shape[::-1])
            packed[:, 1:] = vertices[piece.faces].reshape(-1, 9)
            pieces.append(packed.tobytes())
    return b"".join(pieces)


class PreviewBusy(Exception):
    pass


class MeshPreviewPool:
    def __init__(self):
        self.pool: ProcessPoolExecutor | None = None
        self.pending = 0
        self.cache: OrderedDict[bytes, bytes] = OrderedDict()
        self.cache_bytes = 0

    async def build(
        self, project: str, raw: bytes, target=None, downsample: int = 1
    ) -> bytes:
        key = hashlib.sha256(
            f"{project}:{target}:{downsample}:".encode() + raw
        ).digest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.pending >= 2:
            raise PreviewBusy
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        self.pending += 1
        future = asyncio.wrap_future(
            self.pool.submit(build_preview, raw, target, downsample)
        )
        # A disconnected browser must not free capacity while native work still runs.
        future.add_done_callback(lambda _: setattr(self, "pending", self.pending - 1))
        result = await asyncio.shield(future)
        if len(result) <= CACHE_BYTES:
            previous = self.cache.pop(key, b"")
            self.cache_bytes += len(result) - len(previous)
            self.cache[key] = result
            while self.cache_bytes > CACHE_BYTES or len(self.cache) > 64:
                self.cache_bytes -= len(self.cache.popitem(last=False)[1])
        return result

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)
        self.cache.clear()
        self.cache_bytes = 0
