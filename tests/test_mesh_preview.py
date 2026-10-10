"""Interactive zmesh previews: bounded wire input, real native surfaces, isolated execution."""

import asyncio
import struct

import numpy as np
import pytest
from ml4paleo_server.mesh_preview import (
    MAX_CHUNK_TRIANGLES,
    MAX_INPUT,
    MeshPreviewPool,
    PreviewBusy,
    build_preview,
)


def wire(data, block=64):
    shape = data.shape
    chunks = []
    for z in range(0, shape[0], block):
        for y in range(0, shape[1], block):
            for x in range(0, shape[2], block):
                part = data[z : z + block, y : y + block, x : x + block]
                chunks.append(struct.pack("<6I", z, y, x, *part.shape) + part.tobytes())
    return struct.pack("<4I", *shape, len(chunks)) + b"".join(chunks)


def test_native_surface_bounds_classes_and_empty_state():
    data = np.zeros((12, 16, 20), dtype=np.uint8)
    data[2:10, 3:13, 4:16] = 2
    data[4:8, 6:10, 8:12] = 3
    mesh = np.frombuffer(build_preview(wire(data)), dtype="<f4").reshape(-1, 10)
    assert set(mesh[:, 0]) == {2, 3}
    xyz = mesh[:, 1:].reshape(-1, 3)
    assert np.allclose(xyz.min(axis=0), [4, 3, 2])
    assert np.allclose(xyz.max(axis=0), [16, 13, 10])
    assert np.isfinite(mesh).all()
    for value in (0, 1, 255):
        assert build_preview(wire(np.full((2, 3, 4), value, dtype=np.uint8))) == b""


def test_native_marching_cubes_shape_and_chunk_seams():
    data = np.full((8, 8, 128), 2, dtype=np.uint8)
    mesh = np.frombuffer(build_preview(wire(data)), dtype="<f4").reshape(-1, 10)
    triangles = mesh[:, 1:].reshape(-1, 3, 3)
    assert len(triangles) > 0
    # zmesh bevels the corners, rather than returning axis-aligned voxel cubes.
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    assert np.any(np.count_nonzero(np.abs(normals) > 1e-6, axis=1) > 1)
    # The seam is x=64.5; no cap lies entirely on that internal plane.
    assert not np.any(np.all(np.isclose(triangles[:, :, 0], 64.5), axis=1))
    assert np.allclose(triangles.reshape(-1, 3).max(axis=0), [128, 8, 8])


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"x" * (MAX_INPUT + 1),
        struct.pack("<4I", 1, 1, 1, 17),
        struct.pack("<4I", 0, 1, 1, 1),
    ],
)
def test_refuses_invalid_headers(raw):
    with pytest.raises(ValueError):
        build_preview(raw)


def test_refuses_truncation_extra_bytes_duplicate_or_unaligned_chunks():
    raw = wire(np.full((2, 3, 4), 2, dtype=np.uint8))
    for bad in (
        raw[:-1],
        raw + b"x",
        struct.pack("<4I", 2, 3, 4, 2) + raw[16:] * 2,
        struct.pack("<4I", 2, 3, 5, 1)
        + struct.pack("<6I", 0, 0, 1, 2, 3, 4)
        + raw[40:],
    ):
        with pytest.raises(ValueError):
            build_preview(bad)


def test_dense_chunks_subdivide_and_can_use_a_coarser_native_mesh(monkeypatch):
    from ml4paleo_server.mesh_preview import MAX_SURFACE

    from ml4paleo.meshing import blocks

    native = blocks._mesh
    surfaces = []

    def bounded(padded, *args):
        surfaces.append(blocks._faces(padded))
        assert surfaces[-1] <= MAX_SURFACE
        return native(padded, *args)

    monkeypatch.setattr(blocks, "_mesh", bounded)
    data = np.random.default_rng(0).integers(0, 2, (64, 64, 64), dtype=np.uint8) * 2
    with pytest.raises(blocks.TooDetailed):
        build_preview(wire(data), (0, 0, 0))
    result = build_preview(wire(data), (0, 0, 0), downsample=8)
    assert 0 < len(result) // 40 <= MAX_CHUNK_TRIANGLES
    assert surfaces


@pytest.mark.parametrize("downsample", [1, 2, 4, 8, 16])
def test_individual_chunks_meet_at_every_preview_detail(downsample):
    data = np.full((9, 11, 129), 2, dtype=np.uint8)
    raw = wire(data)
    parts = [
        np.frombuffer(build_preview(raw, (0, 0, x), downsample), dtype="<f4").reshape(
            -1, 10
        )
        for x in (0, 64, 128)
    ]
    triangles = np.concatenate(parts)[:, 1:].reshape(-1, 3, 3)
    assert np.allclose(triangles.reshape(-1, 3).min(axis=0), [0, 0, 0])
    assert np.allclose(triangles.reshape(-1, 3).max(axis=0), [129, 11, 9])
    # Shared vertices agree exactly, even with a partial final coarse cell.
    for left, right in zip(parts[:-1], parts[1:], strict=True):
        a, b = left[:, 1:].reshape(-1, 3), right[:, 1:].reshape(-1, 3)
        seam = a[:, 0].max()
        assert seam == b[:, 0].min()
        assert {tuple(p) for p in a[a[:, 0] == seam]} == {
            tuple(p) for p in b[b[:, 0] == seam]
        }
        assert not np.any(np.all(np.isclose(triangles[:, :, 0], seam), axis=1))


def test_pool_runs_native_work_in_child_and_caches_results():
    async def check():
        pool = MeshPreviewPool()
        try:
            raw = wire(np.full((8, 8, 8), 2, dtype=np.uint8))
            result = await pool.build("project", raw)
            assert result
            assert pool.pending == 0
            pool.pending = 2
            assert await pool.build("project", raw) == result
            with pytest.raises(PreviewBusy):
                await pool.build("another-project", raw)
        finally:
            pool.close()

    asyncio.run(check())
