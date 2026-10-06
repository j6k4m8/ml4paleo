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
    assert sorted(jobs) == ["ABC123", "DEAD00", "FEED01"]
    assert status(jobs["ABC123"]) == "meshed"
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
