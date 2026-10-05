"""
Golden tests for the single (x, y, z) to (c, z, y, x) boundary, the OME-Zarr
metadata, and pyramid building.

Every test uses non-square, anisotropic data, so a swapped or transposed axis
changes a shape or a value and fails loudly.
"""

import numpy as np
import pytest
from conftest import S3_TEST_BUCKET
from PIL import Image

from ml4paleo.ome import (
    OmeImage,
    build_pyramid,
    downsample_level,
    plan_levels,
    write_from_provider,
)
from ml4paleo.storage import StorageGrant
from ml4paleo.volume_providers import ImageStackVolumeProvider, NumpyVolumeProvider
from ml4paleo.volume_providers.dicomvp import DicomVolumeProvider

SMALL_CHUNKS = {"chunk_zyx": (2, 2, 2), "shard_zyx": (2, 4, 4)}


def _grant(tmp_path, name="image") -> StorageGrant:
    return StorageGrant(url=f"file://{tmp_path}/{name}", access="rw")


def _image_for(provider, grant, voxel_size_zyx=None) -> OmeImage:
    width, height, depth = provider.shape
    return OmeImage.create(
        grant,
        shape_czyx=(1, depth, height, width),
        dtype=provider.dtype,
        voxel_size_zyx=voxel_size_zyx,
        **SMALL_CHUNKS,
    )


def test_image_stack_pixels_land_at_z_row_column(tmp_path):
    rng = np.random.default_rng(0)
    # Each slice is 3 rows (y) by 5 columns (x).
    slices = [rng.integers(0, 60000, size=(3, 5), dtype=np.uint16) for _ in range(5)]
    paths = []
    for z, pixels in enumerate(slices):
        paths.append(tmp_path / f"slice_{z}.tif")
        Image.fromarray(pixels).save(paths[-1])
    provider = ImageStackVolumeProvider(paths)
    assert provider.shape == (5, 3, 5)

    image = _image_for(provider, _grant(tmp_path))
    write_from_provider(provider, image)

    stored = np.asarray(image.array(0)[0])
    assert stored.shape == (5, 3, 5)
    for z, pixels in enumerate(slices):
        np.testing.assert_array_equal(stored[z], pixels)


def test_dicom_pixels_and_spacing_land_in_zyx_order(tmp_path, make_dicom_series):
    def pixels(i):
        return np.arange(4 * 6).reshape(4, 6) + 100 * i

    make_dicom_series(tmp_path / "series", pixels, count=3)
    provider = DicomVolumeProvider(tmp_path / "series")
    image = _image_for(
        provider, _grant(tmp_path), voxel_size_zyx=provider.voxel_size_xyz_mm[::-1]
    )
    write_from_provider(provider, image)

    stored = np.asarray(image.array(0)[0])
    for z in range(3):
        np.testing.assert_array_equal(stored[z], pixels(z))
    # PixelSpacing (0.5 between rows, 0.25 between columns), 2.0 between slices.
    assert image.voxel_size_zyx == (2.0, 0.5, 0.25)
    assert image.unit == "millimeter"


def test_sagittal_dicom_slices_sort_along_their_normal(tmp_path, make_dicom_series):
    # Rows run along patient y and columns along patient z, so slices stack
    # along patient x. Sorting by patient z alone would leave them unsorted.
    make_dicom_series(
        tmp_path / "series",
        lambda i: np.full((4, 6), i),
        count=5,
        orientation=(0, 1, 0, 0, 0, -1),
        origin=(-20.0, 0.0, 0.0),
    )
    provider = DicomVolumeProvider(tmp_path / "series")
    stored = np.asarray(provider[:, :, :])
    assert [int(stored[0, 0, z]) for z in range(5)] == [0, 1, 2, 3, 4]


def test_metadata_round_trips(tmp_path):
    grant = _grant(tmp_path)
    OmeImage.create(
        grant,
        shape_czyx=(2, 9, 7, 5),
        dtype="uint8",
        voxel_size_zyx=(4.0, 1.0, 1.0),
        channel_names=["absorption", "phase"],
        **SMALL_CHUNKS,
    )
    image = OmeImage.open(grant.model_copy(update={"access": "r"}))
    assert image.shape_czyx == (2, 9, 7, 5)
    assert image.dtype == np.uint8
    assert image.voxel_size_zyx == (4.0, 1.0, 1.0)
    attrs = image.group.attrs["ome"]
    assert attrs["version"] == "0.5"
    assert [axis["name"] for axis in attrs["multiscales"][0]["axes"]] == [
        "c",
        "z",
        "y",
        "x",
    ]
    assert [c["label"] for c in attrs["omero"]["channels"]] == ["absorption", "phase"]


def test_unknown_voxel_size_has_no_unit(tmp_path):
    image = OmeImage.create(
        _grant(tmp_path), shape_czyx=(1, 4, 4, 4), dtype="uint8", **SMALL_CHUNKS
    )
    assert image.unit is None
    assert image.voxel_size_zyx is None
    assert image.scale_zyx(1) == (2.0, 2.0, 2.0)


def test_level_plan_approaches_isotropic_voxels_first():
    levels = plan_levels(
        (40, 300, 300), voxel_size_zyx=(4.0, 1.0, 1.0), chunk_zyx=(64, 64, 64)
    )
    factors = [level.factor_zyx for level in levels]
    assert factors[:4] == [(1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 8, 8)]
    final = levels[-1].shape_zyx
    assert all(size <= 64 for size in final)


def test_parallel_slabs_fill_the_volume_and_must_be_shard_aligned(tmp_path):
    data = np.arange(5 * 3 * 7, dtype=np.uint16).reshape(5, 3, 7)  # (x, y, z)
    provider = NumpyVolumeProvider(data)
    image = _image_for(provider, _grant(tmp_path))
    with pytest.raises(ValueError):
        write_from_provider(provider, image, z_range=(1, 4))
    for z_range in [(0, 2), (2, 6), (6, 7)]:
        write_from_provider(provider, image, z_range=z_range)
    np.testing.assert_array_equal(
        np.asarray(image.array(0)[0]), data.transpose(2, 1, 0)
    )


def test_mean_pyramid_matches_block_means(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(0, 1000, size=(6, 8, 4), dtype=np.uint16)  # (x, y, z)
    provider = NumpyVolumeProvider(data)
    image = _image_for(provider, _grant(tmp_path))
    write_from_provider(provider, image)
    build_pyramid(image, "mean")

    zyx = data.transpose(2, 1, 0).astype(np.float64)
    expected = zyx.reshape(2, 2, 4, 2, 3, 2).mean(axis=(1, 3, 5))
    np.testing.assert_array_equal(np.asarray(image.array(1)[0]), np.rint(expected))


def test_mode_pyramid_keeps_thin_labels(tmp_path):
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[:, 3, 3] = 7  # a one-voxel-wide rod along x
    labels[0:2, 0:2, 0:2] = 2
    image = _image_for(NumpyVolumeProvider(labels), _grant(tmp_path))
    write_from_provider(NumpyVolumeProvider(labels), image)
    downsample_level(image, 0, "mode")

    level1 = np.asarray(image.array(1)[0])  # (z, y, x)
    assert (level1[1, 1, :] == 7).all()
    assert level1[0, 0, 0] == 2
    assert set(np.unique(level1)) == {0, 2, 7}


def test_images_work_on_s3(tmp_path, s3_endpoint):
    grant = StorageGrant(
        url=f"s3://{S3_TEST_BUCKET}/ome/{tmp_path.name}",
        access="rw",
        endpoint=s3_endpoint,
        credentials={"access_key_id": "test", "secret_access_key": "test"},
    )
    data = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    image = _image_for(NumpyVolumeProvider(data), grant)
    write_from_provider(NumpyVolumeProvider(data), image)
    build_pyramid(image)
    reopened = OmeImage.open(grant.model_copy(update={"access": "r"}))
    np.testing.assert_array_equal(
        np.asarray(reopened.array(0)[0]), data.transpose(2, 1, 0)
    )
    assert reopened.num_levels == image.num_levels


def _reference_pyramid(volume_zyx, levels, method):
    """
    Build the expected pyramid with plain numpy, one level at a time.
    """
    expected = [volume_zyx]
    for previous, level in zip(levels, levels[1:], strict=False):
        step = [
            b // a for a, b in zip(previous.factor_zyx, level.factor_zyx, strict=True)
        ]
        data = expected[-1]
        out_shape = level.shape_zyx
        pad = [
            (0, o * s - n) for o, s, n in zip(out_shape, step, data.shape, strict=True)
        ]
        if method == "mean":
            padded = np.pad(data, pad, mode="edge").astype(np.float64)
        else:
            padded = np.pad(data, pad, mode="constant")
        result = np.zeros(out_shape, dtype=data.dtype)
        for z, y, x in np.ndindex(*out_shape):
            window = padded[
                z * step[0] : (z + 1) * step[0],
                y * step[1] : (y + 1) * step[1],
                x * step[2] : (x + 1) * step[2],
            ].ravel()
            if method == "mean":
                result[z, y, x] = np.rint(window.mean())
            else:
                values, counts = np.unique(window[window != 0], return_counts=True)
                result[z, y, x] = values[counts.argmax()] if len(values) else 0
        expected.append(result)
    return expected


@pytest.mark.parametrize("method", ["mean", "mode"])
def test_pyramids_match_a_reference_on_awkward_shapes(tmp_path, monkeypatch, method):
    import ml4paleo.ome as ome

    # Read tiny source slabs so the bounded-memory path is exercised.
    monkeypatch.setattr(ome, "DOWNSAMPLE_READ_VOXELS", 50)
    rng = np.random.default_rng(3)
    shape_xyz = (13, 9, 7)  # odd, non-square, spans several shards
    if method == "mean":
        data = rng.integers(0, 60000, size=shape_xyz, dtype=np.uint16)
    else:
        data = rng.choice(np.array([0, 0, 2, 3, 9], dtype=np.uint8), size=shape_xyz)
    provider = NumpyVolumeProvider(data)
    image = OmeImage.create(
        _grant(tmp_path),
        shape_czyx=(1, 7, 9, 13),
        dtype=data.dtype,
        voxel_size_zyx=(2.0, 1.0, 1.0),
        chunk_zyx=(2, 2, 2),
        shard_zyx=(2, 4, 4),
    )
    write_from_provider(provider, image)
    build_pyramid(image, method)

    levels = plan_levels((7, 9, 13), (2.0, 1.0, 1.0), (2, 2, 2))
    assert image.num_levels == len(levels) > 2
    expected = _reference_pyramid(data.transpose(2, 1, 0), levels, method)
    for index, level in enumerate(expected):
        np.testing.assert_array_equal(
            np.asarray(image.array(index)[0]), level, err_msg=f"level {index}"
        )


@pytest.mark.parametrize(
    "voxel_size", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (float("nan"), 1.0, 1.0)]
)
def test_level_planning_rejects_bad_voxel_sizes(voxel_size):
    with pytest.raises(ValueError):
        plan_levels((100, 100, 100), voxel_size, (8, 8, 8))


def test_create_refuses_to_overwrite_unless_asked(tmp_path):
    grant = _grant(tmp_path)
    OmeImage.create(grant, shape_czyx=(1, 4, 4, 4), dtype="uint8", **SMALL_CHUNKS)
    with pytest.raises(Exception):  # noqa: B017 - zarr's "already exists" error
        OmeImage.create(grant, shape_czyx=(1, 4, 4, 4), dtype="uint8", **SMALL_CHUNKS)
    again = OmeImage.create(
        grant, shape_czyx=(1, 6, 4, 4), dtype="uint8", overwrite=True, **SMALL_CHUNKS
    )
    assert again.shape_czyx == (1, 6, 4, 4)


def test_lossy_dtype_conversions_are_refused(tmp_path):
    provider = NumpyVolumeProvider(np.full((4, 4, 4), 300, dtype=np.uint16))
    image = OmeImage.create(
        _grant(tmp_path), shape_czyx=(1, 4, 4, 4), dtype="uint8", **SMALL_CHUNKS
    )
    with pytest.raises(ValueError, match="losing values"):
        write_from_provider(provider, image)


def test_bad_chunking_and_huge_voxel_sizes_are_refused(tmp_path):
    with pytest.raises(ValueError):
        OmeImage.create(
            _grant(tmp_path),
            shape_czyx=(1, 4, 4, 4),
            dtype="uint8",
            chunk_zyx=(0, 2, 2),
        )
    with pytest.raises(ValueError):
        OmeImage.create(
            _grant(tmp_path, "huge"),
            shape_czyx=(1, 400, 4, 4),
            dtype="uint8",
            voxel_size_zyx=(1e308, 1.0, 1.0),
            **SMALL_CHUNKS,
        )


def test_wide_integers_are_not_stored_as_floats(tmp_path):
    provider = NumpyVolumeProvider(np.full((4, 4, 4), 2**60, dtype=np.int64))
    image = OmeImage.create(
        _grant(tmp_path), shape_czyx=(1, 4, 4, 4), dtype="float64", **SMALL_CHUNKS
    )
    with pytest.raises(ValueError, match="losing values"):
        write_from_provider(provider, image)
