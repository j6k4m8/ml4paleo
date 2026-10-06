"""
Reading v1's volume folder: job records, where annotation samples land, and
which segmentation finished.
"""

import json

import pytest
import v1_volume

from ml4paleo.v1import import (
    annotations,
    normalize_job_id,
    read_jobs,
    segmentation,
    status,
)


@pytest.fixture
def root(tmp_path):
    return v1_volume.make(tmp_path / "volume")


def test_job_ids_and_records(root):
    assert normalize_job_id(" abc123 ") == "ABC123"
    assert normalize_job_id("ABC12") is None
    assert normalize_job_id("../ABC1") is None
    jobs = read_jobs(root)
    assert sorted(jobs) == ["ABC123", "DEAD00", "DEAD01", "FEED01"]
    assert status(jobs["ABC123"]) == "meshed"
    assert status(jobs["DEAD01"]) == "convert_error"
    assert read_jobs(root / "nowhere") == {}


def test_samples_land_where_they_were_cut_out(root):
    placed, skipped = annotations(root, "ABC123", v1_volume.SHAPE_XYZ)
    # The sample without metadata can't be placed; a mask without its image
    # isn't a sample at all.
    assert skipped == 1
    assert [a.stamp for a in placed] == ["1745400000", "1745400100-z07"]
    middle, polygon = placed
    # The legacy page labeled the middle slice; the polygon file says its own.
    assert middle.box_zyx == [9, 0, 0, 10, 30, 40]
    assert polygon.box_zyx == [11, 0, 0, 12, 30, 40]
    for annotation in placed:
        y0, y1, x0, x1 = v1_volume.FOREGROUND[annotation.stamp]
        foreground = annotation.foreground()
        # Only the volume's part of the sample: the red in the padding is out.
        assert foreground.shape == (30, 40)
        assert foreground.sum() == (y1 - y0) * (x1 - x0)
        assert foreground[y0:y1, x0:x1].all()


def test_samples_outside_the_volume_are_skipped(root):
    meta = root / "training" / "ABC123" / "meta1745400000.json"
    record = json.loads(meta.read_text())
    record["annotated_local_z_index"] = 10
    record["cutout_origin_xyz"] = [0, 0, 15]
    meta.write_text(json.dumps(record))
    placed, skipped = annotations(root, "ABC123", v1_volume.SHAPE_XYZ)
    assert [a.stamp for a in placed] == ["1745400100-z07"]
    assert skipped == 2
    meta.write_text("{not json")
    assert annotations(root, "ABC123", v1_volume.SHAPE_XYZ)[1] == 2


def test_samples_with_impossible_numbers_are_skipped(root):
    meta = root / "training" / "ABC123" / "meta1745400000.json"
    record = json.loads(meta.read_text())
    # JSON can hold infinities (1e400), NaN, and numbers no scan has.
    for key, value in (
        ("cutout_origin_xyz", "[0, 0, 1e400]"),
        ("cutout_shape_xyz", "[40, 30, NaN]"),
        ("padding_before_xyz", f"[{10**30}, 0, 0]"),
        ("requested_shape_xyz", "[512, 512, -Infinity]"),
        ("annotated_local_z_index", "1e400"),
        ("annotated_local_z_index", "null"),
        ("cutout_origin_xyz", "[0, 0]"),
    ):
        text = json.dumps({**record, key: "PLACEHOLDER"})
        meta.write_text(text.replace('"PLACEHOLDER"', value))
        placed, skipped = annotations(root, "ABC123", v1_volume.SHAPE_XYZ)
        assert [a.stamp for a in placed] == ["1745400100-z07"], key
        assert skipped == 2
    # v1 read these with int(), so a whole number written another way places.
    meta.write_text(json.dumps({**record, "annotated_local_z_index": "6"}))
    placed, _ = annotations(root, "ABC123", v1_volume.SHAPE_XYZ)
    assert placed[0].box_zyx == [10, 0, 0, 11, 30, 40]


def test_only_finished_segmentations_count(root):
    jobs = read_jobs(root)
    # The sidecar names this one; the newer one is an unfinished run.
    assert segmentation(root, "ABC123", jobs["ABC123"]) == "1745400150.zarr"
    # Without sidecars, the status decides, and "annotated" can follow
    # anything.
    assert segmentation(root, "FEED01", jobs["FEED01"]) is None
    finished = {**jobs["FEED01"], "status": "JobStatus.SEGMENTED"}
    assert segmentation(root, "FEED01", finished) == "1745400150.zarr"
    assert segmentation(root, "DEAD00", jobs["DEAD00"]) is None


def test_migrated_sidecars_dont_vouch_for_a_segmentation(root):
    # The newer run crashed while segmenting. v1's "migrate metadata" button
    # then named its half-written segmentation in the runner's sidecar, as it
    # does whenever the folder exists.
    models = root / "models" / "ABC123"
    migrated = {
        **v1_volume.runner_sidecar("1745400300"),
        "legacy_metadata_migrated_at": "2026-04-30T12:00:00+00:00",
    }
    (models / "1745400300.json").write_text(json.dumps(migrated))
    jobs = read_jobs(root)
    assert segmentation(root, "ABC123", jobs["ABC123"]) == "1745400150.zarr"
    # Without a sidecar from the runner, the status decides; here, that
    # segmenting failed.
    (models / "1745400150.json").unlink()
    failed = {**jobs["ABC123"], "status": "JobStatus.SEGMENT_ERROR"}
    assert segmentation(root, "ABC123", failed) is None


def test_only_v1_segmentation_names_count(root):
    segmented = root / "segmented" / "FEED01"
    (segmented / "1745400150.zarr").rename(segmented / "latest.zarr")
    finished = {**read_jobs(root)["FEED01"], "status": "JobStatus.SEGMENTED"}
    assert segmentation(root, "FEED01", finished) is None
    models = root / "models" / "FEED01"
    models.mkdir()
    sidecar = {
        **v1_volume.runner_sidecar("1745400150"),
        "segmentation_id": "latest.zarr",
    }
    (models / "1745400150.json").write_text(json.dumps(sidecar))
    assert segmentation(root, "FEED01", finished) is None
