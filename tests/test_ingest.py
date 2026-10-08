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
    with pytest.raises(IngestError, match="both DICOM files and other images"):
        probe(_stored(tmp_path, _zip(entries))())


def test_files_that_arent_slices_are_left_out(tmp_path, make_dicom_series):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MediaStorageDirectoryStorage

    paths = make_dicom_series(
        tmp_path / "series", lambda i: np.full((4, 5), i), count=3
    )
    entries = {f"DICOM/{path.stem}": path.read_bytes() for path in paths}
    # What a PACS export adds: notes, a manifest, and a DICOMDIR (DICOM, but
    # no image).
    entries["README.TXT"] = b"Exported by a viewer."
    entries["SECTRA/CONTENT.XML"] = b"<content/>"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    meta.MediaStorageSOPInstanceUID = "1.2.3"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    directory = Dataset()
    directory.file_meta = meta
    buffer = io.BytesIO()
    directory.save_as(buffer, enforce_file_format=True)
    entries["INDEX/DIRECTORY"] = buffer.getvalue()

    index = probe(_stored(tmp_path, _zip(entries))())
    assert index.kind == "dicom"
    assert index.shape_xyz == (5, 4, 3)
    assert index.skipped == ["INDEX/DIRECTORY", "README.TXT", "SECTRA/CONTENT.XML"]
    assert index.skipped_count == 3
    assert SourceIndex.from_json(index.to_json()) == index


def test_the_largest_dicom_series_is_used_and_the_others_noted(
    tmp_path, make_dicom_series
):
    big = make_dicom_series(tmp_path / "big", lambda i: np.full((4, 5), i), count=4)
    small = make_dicom_series(tmp_path / "small", lambda i: np.zeros((2, 2)), count=2)
    entries = {f"a/{p.name}": p.read_bytes() for p in big}
    entries |= {f"b/{p.name}": p.read_bytes() for p in small}
    index = probe(_stored(tmp_path, _zip(entries))())
    assert (index.shape_xyz, len(index.members)) == ((5, 4, 4), 4)
    [note] = index.notes
    assert "2 DICOM series" in note and "left out the other 2 files" in note


def test_a_stack_with_a_note_is_read_without_it(tmp_path):
    entries = {
        f"slice_{z}.png": _png(np.full((4, 6), z, dtype=np.uint8)) for z in range(3)
    }
    entries["notes.txt"] = b"scanned on a Tuesday"
    index = probe(_stored(tmp_path, _zip(entries))())
    assert (index.kind, index.shape_xyz) == ("images", (6, 4, 3))
    assert (index.skipped, index.skipped_count) == (["notes.txt"], 1)


def test_many_files_of_an_unknown_kind_are_refused(tmp_path):
    entries = {
        f"slice_{z}.png": _png(np.full((4, 6), z, dtype=np.uint8)) for z in range(3)
    }
    entries |= {f"slice_{z}.jp2": b"\x00\x00\x00\x0cjP  " for z in range(3, 10)}
    with pytest.raises(IngestError, match="doesn't read as slices"):
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


@pytest.mark.parametrize("method", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_unbounded_compression_methods_are_refused(tmp_path, method):
    archive = _zip({"a.png": _png(np.zeros((4, 4), dtype=np.uint8))}, method)
    with pytest.raises(IngestError, match="compression method"):
        probe(_stored(tmp_path, archive)())


def _understate(archive: bytes, name: str, declared: int) -> bytes:
    """
    Rewrite a member's declared uncompressed size, as a hostile archive would.
    """
    data = bytearray(archive)
    info = zipfile.ZipFile(io.BytesIO(archive)).getinfo(name)
    # The local header and the central directory entry both record it.
    data[info.header_offset + 22 : info.header_offset + 26] = declared.to_bytes(
        4, "little"
    )
    central = archive.rfind(b"PK\x01\x02")
    data[central + 24 : central + 28] = declared.to_bytes(4, "little")
    return bytes(data)


def test_a_member_cannot_expand_past_its_declared_size(tmp_path):
    import tracemalloc

    archive = _understate(_zip({"a.tif": bytes(64 * 1024 * 1024)}), "a.tif", 1000)
    opener = _stored(tmp_path, archive)
    tracemalloc.start()
    with pytest.raises(IngestError, match="damaged"):
        probe(opener())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 16 * 1024 * 1024


def test_entry_counts_come_from_the_end_record(tmp_path, monkeypatch):
    from ml4paleo import ingest

    small = _zip({f"s{i}.png": b"x" for i in range(3)})
    assert ingest.entry_count(io.BytesIO(small)) == 3
    # More than 65535 entries need the ZIP64 end record.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for i in range(70_000):
            archive.writestr(f"{i}", b"")
    assert ingest.entry_count(io.BytesIO(buffer.getvalue())) == 70_000
    # A too-large archive is refused before its directory is read.
    monkeypatch.setattr(ingest, "MAX_MEMBERS", 2)
    with pytest.raises(IngestError, match="more than 2 files"):
        probe(_stored(tmp_path, small)())


def test_a_multiframe_dicom_is_ingested(tmp_path):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    frames = np.stack([np.full((4, 6), 10 * i, dtype=np.uint16) for i in range(5)])
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = 4, 6
    ds.NumberOfFrames = 5
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = frames.tobytes()
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    opener = _stored(tmp_path, _zip({"volume.dcm": buffer.getvalue()}))

    index = probe(opener())
    assert (index.kind, index.shape_xyz) == ("dicom", (6, 4, 5))
    provider = slab_provider(opener(), index)
    np.testing.assert_array_equal(provider[:, :, 2:4], frames[2:4].transpose(2, 1, 0))


def test_summaries_skip_values_that_cant_be_displayed():
    floats = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0, 3.0], dtype=np.float32)
    summary = intensity_summary(floats)
    assert (summary["min"], summary["max"]) == (1.0, 3.0)
    assert intensity_summary(np.array([True, False]))["max"] == 1
    assert intensity_summary(np.array([np.nan]))["window"] == [0, 0]


def test_slice_limits_follow_the_memory_budget():
    from ml4paleo.ingest import SliceLimits

    MiB = 1024**2
    assert SliceLimits.for_memory(1024 * MiB).max_member_bytes == 256 * MiB
    assert SliceLimits.for_memory(10 * MiB).max_decoded_bytes == 64 * MiB
    assert SliceLimits.for_memory(64 * 1024 * MiB).max_member_bytes == 2048 * MiB


def test_slices_too_large_for_the_budget_are_refused_before_decoding(tmp_path):
    from ml4paleo.ingest import SliceLimits

    pixels = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    opener = _stored(tmp_path, _zip({"a.png": _png(pixels)}))
    # 64 x 64 pixels decode to 4 KiB, more than this budget allows...
    tight = SliceLimits(max_member_bytes=10**6, max_decoded_bytes=4000)
    with pytest.raises(IngestError, match="too large"):
        probe(opener(), tight)
    # ...and the member itself can be too big as well.
    with pytest.raises(IngestError, match="this server reads slices of up to"):
        probe(opener(), SliceLimits(max_member_bytes=100, max_decoded_bytes=10**6))
    index = probe(opener())
    with pytest.raises(ValueError, match="too large"):
        slab_provider(opener(), index, tight)[:, :, :]


def test_dicom_slices_are_sized_from_their_headers(tmp_path, make_dicom_series):
    from ml4paleo.ingest import SliceLimits

    paths = make_dicom_series(
        tmp_path / "series", lambda i: np.zeros((40, 50)), count=2
    )
    opener = _stored(tmp_path, _zip({path.name: path.read_bytes() for path in paths}))
    # 40 x 50 pixels of 16 bits: 4000 bytes each.
    with pytest.raises(IngestError, match="decodes to"):
        probe(opener(), SliceLimits(max_member_bytes=10**6, max_decoded_bytes=3999))
    assert probe(opener(), SliceLimits(10**6, 4000)).shape_xyz == (50, 40, 2)


def test_names_that_arent_the_utf8_they_claim_are_refused(tmp_path):
    name = "slice_\u00e9.png"
    archive = _zip({name: _png(np.zeros((4, 4), np.uint8))})
    # zipfile flags the name as UTF-8; make its bytes anything but.
    encoded = name.encode()
    damaged = archive.replace(encoded, encoded[:-6] + b"\xff\xfe" + encoded[-4:])
    with pytest.raises(IngestError, match="The archive is damaged"):
        probe(_stored(tmp_path, damaged)())
