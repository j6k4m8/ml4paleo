"""
Ingesting archives: probing image stacks and DICOM series inside a zip read
in place from storage, copying slabs into OME-Zarr, and refusing unsafe or
confusing archives.
"""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from ml4paleo.ingest import (
    IngestError,
    SourceIndex,
    intensity_summary,
    natural_key,
    probe,
    slab_provider,
)
from ml4paleo.ome import OmeImage, write_from_provider
from ml4paleo.storage import StorageGrant, open_object, put_bytes


def _grant(tmp_path, name="store") -> StorageGrant:
    return StorageGrant(url=f"file://{tmp_path}/{name}", access="rw")


def _zip(entries: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _stored(tmp_path, archive: bytes):
    grant = _grant(tmp_path)
    put_bytes(grant, "data", archive)
    # A small buffer, so reads really go through many range requests.
    return lambda: open_object(grant, "data", buffer_size=4096)


def test_an_image_stack_is_probed_and_copied_slab_by_slab(tmp_path):
    rng = np.random.default_rng(0)
    slices = [rng.integers(0, 60000, (6, 9), dtype=np.uint16) for _ in range(12)]
    entries = {f"scan/slice_{z}.png": _png(s) for z, s in enumerate(slices)}
    entries["__MACOSX/scan/._slice_0.png"] = b"resource fork"
    entries["scan/.DS_Store"] = b"finder"
    opener = _stored(tmp_path, _zip(entries))

    index = probe(opener())
    assert index.kind == "images"
    assert index.shape_xyz == (9, 6, 12)
    # Numbers sort as numbers: slice_2 before slice_10.
    assert index.members[:3] == [
        "scan/slice_0.png",
        "scan/slice_1.png",
        "scan/slice_2.png",
    ]
    assert SourceIndex.from_json(index.to_json()) == index

    image = OmeImage.create(
        _grant(tmp_path, "image"),
        shape_czyx=(1, 12, 6, 9),
        dtype=np.uint16,
        chunk_zyx=(4, 4, 4),
        shard_zyx=(4, 4, 8),
    )
    # Each slab job opens the archive itself and copies its own slices.
    for z_range in [(0, 4), (4, 8), (8, 12)]:
        write_from_provider(slab_provider(opener(), index), image, z_range=z_range)
    np.testing.assert_array_equal(np.asarray(image.array(0)[0]), np.stack(slices))


def test_a_dicom_series_is_sorted_by_position(tmp_path, make_dicom_series):
    def pixels(i):
        return np.full((4, 5), 100 * i, dtype=np.uint16)

    paths = make_dicom_series(tmp_path / "series", pixels, count=5)
    opener = _stored(tmp_path, _zip({path.name: path.read_bytes() for path in paths}))

    index = probe(opener())
    assert index.kind == "dicom"
    assert index.shape_xyz == (5, 4, 5)
    assert index.voxel_size_zyx == (2.0, 0.5, 0.25)
    assert index.unit == "millimeter"
    volume = slab_provider(opener(), SourceIndex.from_json(index.to_json()))[:, :, :]
    for z in range(5):
        np.testing.assert_array_equal(volume[:, :, z], pixels(z).T)


def test_mixed_archives_are_refused(tmp_path, make_dicom_series):
    paths = make_dicom_series(tmp_path / "series", lambda i: np.zeros((4, 4)), count=2)
    entries = {path.name: path.read_bytes() for path in paths}
    entries["zz_photo.png"] = _png(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(IngestError, match="mixes DICOM"):
        probe(_stored(tmp_path, _zip(entries))())


@pytest.mark.parametrize(
    ("archive", "message"),
    [
        (b"not a zip at all", "not a zip"),
        (_zip({"../escape.png": b"x"}), "unsafe file name"),
        (_zip({"/etc/passwd": b"x"}), "unsafe file name"),
        (_zip({"__MACOSX/a": b"x", ".hidden": b"y"}), "no slices"),
        (_zip({"bomb.tif": bytes(50_000_000)}), "expands too much"),
    ],
)
def test_bad_archives_are_refused(tmp_path, archive, message):
    with pytest.raises(IngestError, match=message):
        probe(_stored(tmp_path, archive)())


def test_slices_of_different_sizes_are_refused(tmp_path):
    entries = {
        "a_1.png": _png(np.zeros((4, 4), dtype=np.uint8)),
        "a_2.png": _png(np.zeros((5, 4), dtype=np.uint8)),
    }
    opener = _stored(tmp_path, _zip(entries))
    index = probe(opener())
    with pytest.raises(ValueError, match="size"):
        slab_provider(opener(), index)[:, :, :]


def test_natural_sort_and_intensity_summary():
    names = ["s10.png", "s2.png", "S1.png"]
    assert sorted(names, key=natural_key) == ["S1.png", "s2.png", "s10.png"]
    values = np.concatenate([np.zeros(10), np.arange(1000), [65535]])
    summary = intensity_summary(values)
    assert summary["max"] == 65535
    assert 0 <= summary["window"][0] < summary["window"][1] < 65535
    assert sum(summary["histogram"]["counts"]) == values.size
