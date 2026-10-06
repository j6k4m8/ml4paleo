"""
Export archives: zip files written in parts that join into one valid
archive, the same bytes every time, and slices in the orientation of an
uploaded stack.
"""

import io
import zipfile

import numpy as np
import pytest
import tifffile
from PIL import Image

from ml4paleo.export import (
    Parts,
    add_entry,
    encode_slice,
    open_archive,
    slab_depth,
    slice_names,
    slices,
)


def archive(entries, part_bytes) -> tuple[bytes, list[int]]:
    stored: dict[int, bytes] = {}
    out = Parts(stored.__setitem__, part_bytes=part_bytes)
    with open_archive(out) as zipped:
        for name, data, compress in entries:
            # Written in pieces, as streamed objects are.
            pieces = [data[i : i + 1000] for i in range(0, len(data), 1000)]
            add_entry(zipped, name, pieces, len(data), compress=compress)
    sizes = out.finish()
    assert sorted(stored) == list(range(len(sizes)))
    assert [len(stored[i]) for i in range(len(sizes))] == sizes
    return b"".join(stored[i] for i in range(len(sizes))), sizes


def test_parts_join_into_one_archive():
    rng = np.random.default_rng(0)
    entries = [
        ("volume.zarr/zarr.json", b'{"zarr_format": 3}', False),
        ("volume.zarr/c/0/0/0", rng.bytes(10_000), False),
        ("meshes/bone.obj", b"v 0 0 0\n" * 2000, True),
    ]
    data, sizes = archive(entries, part_bytes=4096)
    assert len(sizes) >= 3 and all(size == 4096 for size in sizes[:-1])
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        assert zipped.testzip() is None
        assert zipped.namelist() == [name for name, _, _ in entries]
        for name, content, _ in entries:
            assert zipped.read(name) == content
        assert zipped.getinfo("meshes/bone.obj").compress_size < 2000 * 8


def test_archives_are_the_same_every_time():
    entries = [("a/z00000.tif", b"x" * 5000, False), ("a/z00001.tif", b"y", False)]
    assert archive(entries, 1024) == archive(entries, 1024)


def test_an_empty_archive_is_one_part():
    data, sizes = archive([], 1024)
    assert len(sizes) == 1
    assert zipfile.ZipFile(io.BytesIO(data)).namelist() == []


def test_slices_keep_the_stack_orientation():
    # (z, y, x), not square, so a transposed slice can't pass.
    volume = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7) * 300
    for fmt, read in (
        ("png", lambda raw: np.asarray(Image.open(io.BytesIO(raw)))),
        ("tiff", lambda raw: tifffile.imread(io.BytesIO(raw))),
    ):
        planes = list(slices(volume, depth=2))
        assert [(c, z) for c, z, _ in planes] == [(0, 0), (0, 1), (0, 2)]
        for _, z, plane in planes:
            back = read(encode_slice(plane, fmt))  # type: ignore[arg-type]
            assert back.shape == (5, 7)
            np.testing.assert_array_equal(back, volume[z])


def test_tiff_holds_any_image_type_and_png_doesnt():
    for dtype in (np.uint8, np.uint16, np.int16, np.float32):
        plane = (np.arange(20).reshape(4, 5) - 7).astype(dtype)
        back = tifffile.imread(io.BytesIO(encode_slice(plane, "tiff")))
        assert back.dtype == plane.dtype
        np.testing.assert_array_equal(back, plane)
    with pytest.raises(ValueError, match="TIFF"):
        encode_slice(np.zeros((2, 2), dtype=np.float32), "png")


def test_channels_get_folders_and_slabs_fit_the_budget():
    name = slice_names((2, 1200, 10, 10), "tiff", "skull-image-tiff")
    assert name(1, 7) == "skull-image-tiff/c1/z00007.tif"
    assert slice_names((1, 3, 1, 1), "png", "x")(0, 2) == "x/z00002.png"
    assert slice_names((1, 200_000, 1, 1), "png", "x")(0, 2) == "x/z000002.png"
    # Two 1000² uint16 channels are 4 MB per z: 64 fit in 1 GiB, 2 in 20 MB.
    assert slab_depth((2, 100, 1000, 1000), 2, 64, 1024**3) == 64
    assert slab_depth((2, 100, 1000, 1000), 2, 64, 20 * 1024**2) == 2
    assert slab_depth((2, 100, 1000, 1000), 2, 64, 1) == 1
    image = np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    planes = list(slices(image, depth=2))
    assert [(c, z) for c, z, _ in planes] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
    ]
    np.testing.assert_array_equal(planes[3][2], image[1, 1])
