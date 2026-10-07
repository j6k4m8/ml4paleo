"""
Label files: TIFF stacks and zips of TIFF or PNG slices, read a slice at a
time, checked against the image's size, counted, and turned into label
chunks through a lookup from the file's values to label values.
"""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from ml4paleo.ingest import IngestError
from ml4paleo.labelimport import (
    MAX_VALUES,
    ImagePlanes,
    Lookup,
    ZipPlanes,
    chunk_blocks,
    chunk_delta,
    count_values,
    file_kind,
)
from ml4paleo.labels import LABEL_CHUNK_ZYX, Source
from ml4paleo.labels.deltas import apply_delta


def _tiff(path, slices, compression="tiff_lzw"):
    images = [Image.fromarray(s) for s in slices]
    images[0].save(
        path,
        format="TIFF",
        save_all=True,
        append_images=images[1:],
        compression=compression,
    )
    return path


def _image(pixels: np.ndarray, format: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=format)
    return buffer.getvalue()


def _zip(entries: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer


def _labels(shape_zyx, seed=0, values=(0, 0, 0, 1, 7, 300)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array(values, dtype=np.uint16), size=shape_zyx)


def _assemble(planes, lookup, shape_zyx, **kwargs) -> np.ndarray:
    """Every block `chunk_blocks` gives, put back in place."""
    out = np.zeros(shape_zyx, dtype=np.uint8)
    for key, origin, block in chunk_blocks(planes, lookup, shape_zyx, **kwargs):
        assert origin == tuple(k * s for k, s in zip(key, LABEL_CHUNK_ZYX, strict=True))
        assert block.any()
        assert all(n <= s for n, s in zip(block.shape, LABEL_CHUNK_ZYX, strict=True))
        z, y, x = origin
        out[z : z + block.shape[0], y : y + block.shape[1], x : x + block.shape[2]] = (
            block
        )
    return out


def test_a_tiff_stack_is_counted_and_mapped_to_labels(tmp_path):
    labels = _labels((70, 66, 130))
    path = _tiff(tmp_path / "labels.tif", list(labels))
    with open(path, "rb") as fileobj:
        assert file_kind(fileobj) == "tiff"
    with ImagePlanes(path) as planes:
        assert len(planes) == 70
        assert planes.size_yx == (66, 130)
        seen = []
        counts = count_values(planes, (70, 66, 130), progress=seen.append)
        values, voxels = np.unique(labels, return_counts=True)
        assert counts == dict(zip(values.tolist(), voxels.tolist(), strict=True))
        assert seen[-1] == 1.0 and seen == sorted(seen)

        # 7 becomes background, 300 a class, and 1 (left out) stays unlabeled.
        lookup = Lookup([(7, 1), (300, 4)])
        expected = np.select([labels == 7, labels == 300], [1, 4], 0)
        np.testing.assert_array_equal(
            _assemble(planes, lookup, (70, 66, 130)), expected
        )


def test_a_zip_of_slices_is_read_in_name_order(tmp_path):
    labels = _labels((12, 9, 7), values=(0, 2, 3))
    entries = {}
    for z, plane in enumerate(labels.astype(np.uint8)):
        # PNG and TIFF slices mix; slice_2 sorts before slice_10.
        format = "PNG" if z % 2 else "TIFF"
        entries[f"stack/slice_{z}.{format.lower()}"] = _image(plane, format)
    entries["__MACOSX/stack/._slice_0.png"] = b"resource fork"
    archive = _zip(entries)
    assert file_kind(archive) == "zip"
    planes = ZipPlanes(archive)
    assert [planes.name(z) for z in (1, 2, 10)] == [
        "stack/slice_1.png",
        "stack/slice_2.tiff",
        "stack/slice_10.tiff",
    ]
    assert count_values(planes, (12, 9, 7)) == {
        int(v): int((labels == v).sum()) for v in (0, 2, 3)
    }
    np.testing.assert_array_equal(
        _assemble(planes, Lookup([(2, 5), (3, 6)]), (12, 9, 7)),
        np.select([labels == 2, labels == 3], [5, 6], 0),
    )


def test_a_single_png_is_one_slice():
    plane = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    source = io.BytesIO(_image(plane, "PNG"))
    assert file_kind(source) == "png"
    planes = ImagePlanes(source)
    assert count_values(planes, (1, 2, 2)) == {0: 2, 1: 1, 2: 1}


def test_labels_must_be_the_size_of_the_image(tmp_path):
    path = _tiff(tmp_path / "labels.tif", list(_labels((3, 5, 4))))
    with ImagePlanes(path) as planes:
        with pytest.raises(IngestError) as refused:
            count_values(planes, (4, 5, 4))
    assert str(refused.value).startswith(
        "The labels are 4 × 5 × 3 voxels, but the image is 4 × 5 × 4 (x × y × z)."
    )


@pytest.mark.parametrize(
    ("pixels", "message"),
    [
        (np.array([[0, 0.5]], dtype=np.float32), "values that aren't"),
        (np.array([[0, np.nan]], dtype=np.float32), "values that aren't"),
        (np.zeros((1, 2, 3), dtype=np.uint8), "has RGB pixels"),
    ],
)
def test_labels_must_be_whole_numbers_in_one_channel(pixels, message):
    planes = ZipPlanes(_zip({"a.tif": _image(pixels, "TIFF")}))
    with pytest.raises(IngestError, match=message):
        count_values(planes)


def test_floats_that_hold_whole_numbers_are_fine():
    pixels = np.array([[0, 2], [-1, 2]], dtype=np.float32)
    planes = ZipPlanes(_zip({"a.tif": _image(pixels, "TIFF")}))
    assert count_values(planes) == {-1: 1, 0: 1, 2: 2}
    lookup = Lookup([(-1, 1), (2, 3)])
    assert lookup(planes.plane(0)).tolist() == [[0, 3], [1, 3]]


def test_more_values_than_a_project_has_classes_for_are_refused():
    plane = np.arange(MAX_VALUES + 1, dtype=np.uint16).reshape(1, -1)
    planes = ZipPlanes(_zip({"a.png": _image(plane, "PNG")}))
    # 0 and MAX_VALUES others fit...
    assert len(count_values(planes)) == MAX_VALUES + 1
    planes = ZipPlanes(_zip({"a.png": _image(plane + 1, "PNG")}))
    # ...one more doesn't.
    with pytest.raises(IngestError, match=f"more than {MAX_VALUES} different"):
        count_values(planes)


def test_slices_of_other_sizes_and_pages_in_a_zip_are_refused(tmp_path):
    planes = ZipPlanes(
        _zip(
            {
                "a.png": _image(np.zeros((4, 4), np.uint8), "PNG"),
                "b.png": _image(np.zeros((4, 5), np.uint8), "PNG"),
            }
        )
    )
    with pytest.raises(IngestError, match="b.png is 5 × 4 pixels and the first"):
        count_values(planes)
    stack = _tiff(tmp_path / "two.tif", [np.zeros((2, 2), np.uint8)] * 2)
    planes = ZipPlanes(_zip({"a.tif": stack.read_bytes()}))
    with pytest.raises(IngestError, match="a.tif has several pages. Upload a TIFF"):
        count_values(planes)


def test_other_files_are_refused():
    with pytest.raises(IngestError, match="isn't a TIFF or a zip"):
        file_kind(io.BytesIO(b"just some text, not labels"))


@pytest.mark.parametrize("keep", [0.1, 0.5, 0.8])
@pytest.mark.filterwarnings("ignore:Corrupt EXIF data")
def test_damaged_files_are_refused_as_such(tmp_path, keep):
    data = _tiff(tmp_path / "a.tif", [np.ones((40, 40), np.uint8)] * 3).read_bytes()
    damaged = data[: int(len(data) * keep)]
    # Damaged past reading, or holding fewer slices than the image.
    with pytest.raises(IngestError):
        count_values(ImagePlanes(io.BytesIO(damaged)), (3, 40, 40))
    with pytest.raises(IngestError):
        count_values(ZipPlanes(_zip({"a.tif": damaged})), (1, 40, 40))


def test_lookups_map_values_and_leave_the_rest_unlabeled():
    lookup = Lookup([(3, 2), (1, 1)])
    for dtype in (np.uint8, np.uint16, np.int32, np.int64):
        plane = np.array([[0, 1, 2, 3]], dtype=dtype)
        assert lookup(plane).tolist() == [[0, 1, 0, 2]]
    negative = Lookup([(-5, 2), (70000, 3)])
    assert negative(np.array([[-5, 0, 70000, 9]], np.int32)).tolist() == [[2, 0, 3, 0]]
    assert Lookup([])(np.array([[1, 2]], np.uint8)).tolist() == [[0, 0]]
    with pytest.raises(ValueError):
        Lookup([(1, 2), (1, 3)])
    with pytest.raises(ValueError):
        Lookup([(1, 0)])


def test_narrow_bands_give_the_same_chunks(tmp_path):
    labels = _labels((65, 130, 70), seed=3, values=(0, 0, 5))
    path = _tiff(tmp_path / "labels.tif", list(labels))
    lookup = Lookup([(5, 2)])
    read = []
    with ImagePlanes(path) as planes:
        whole = _assemble(planes, lookup, labels.shape)
        # A band of one chunk row at a time reads each slice once per band.
        banded = _assemble(
            planes, lookup, labels.shape, max_bytes=1, check=lambda: read.append(1)
        )
    np.testing.assert_array_equal(whole, banded)
    np.testing.assert_array_equal(whole, np.where(labels == 5, 2, 0))
    assert len(read) == 65 * 3


def test_a_chunks_edit_writes_its_labels():
    block = np.zeros((64, 64, 64), dtype=np.uint8)
    block[1:3, 4:9, 10:12] = 2
    block[5, 6, 7] = 3
    empty = np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    for labels in (block, np.where(block == 2, 2, 0).astype(np.uint8)):
        delta = chunk_delta(labels, (64, 0, 128))
        assert delta.key == (1, 0, 2) and delta.only_if == "unlabeled"
        assert (delta.value is None) == (len(np.unique(labels)) > 2)
        applied = apply_delta(empty, empty, delta, Source.IMPORTED)
        np.testing.assert_array_equal(applied.class_chunk, labels)
