"""
The final segmentation: labels overrule the prediction, complete ROIs are
background, and specks are removed across shard boundaries.
"""

import itertools

import numpy as np

from ml4paleo.labels import BACKGROUND
from ml4paleo.segmentation.compose import (
    apply_shard,
    complete_mask,
    find_specks,
    label_shard,
    merge,
    shard_grid,
)
from ml4paleo.segmentation.predict import shard_boxes

BONE, TOOTH = 2, 3


def test_labels_overrule_the_prediction_and_complete_rois_are_background():
    prediction = np.full((4, 4, 4), BONE, dtype=np.uint8)
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[0] = TOOTH
    complete = complete_mask((0, 0, 0, 4, 4, 4), [(0, 0, 0, 2, 4, 4)])
    merged = merge(prediction, labels, complete)
    assert (merged[0] == TOOTH).all()  # labeled, inside the complete ROI
    assert (merged[1] == BACKGROUND).all()  # unlabeled, inside it
    assert (merged[2:] == BONE).all()  # outside: the prediction


def compose(volume, labeled, shard, min_voxels):
    """Run the per-shard steps over a whole volume, as the jobs would."""
    boxes = shard_boxes(volume.shape, shard)
    regions = [tuple(slice(b[a], b[a + 3]) for a in range(3)) for b in boxes]
    summaries = [label_shard(volume[r], labeled[r]) for r in regions]
    remove = find_specks(summaries, shard_grid(volume.shape, shard), min_voxels)
    final = np.empty_like(volume)
    for region, ids in zip(regions, remove, strict=True):
        final[region] = apply_shard(volume[region], ids)
    return final


def test_specks_are_counted_across_shards():
    volume = np.full((40, 40, 40), BACKGROUND, dtype=np.uint8)
    labeled = np.zeros(volume.shape, dtype=bool)
    # A thin bar across three shards: 40 voxels, at most 16 in any shard.
    volume[5, 5, :] = BONE
    # A speck of 2 voxels, on a shard corner.
    volume[15:17, 30, 30] = BONE
    # A speck someone labeled part of.
    volume[30, 10, 10:12] = BONE
    labeled[30, 10, 10] = True
    # A tooth speck touching the bar: a different class, so its own piece.
    volume[6, 5, 20] = TOOTH
    final = compose(volume, labeled, (16, 16, 16), min_voxels=20)
    assert (final[5, 5, :] == BONE).all()
    assert (final[15:17, 30, 30] == BACKGROUND).all()
    assert (final[30, 10, 10:12] == BONE).all()
    assert final[6, 5, 20] == BACKGROUND


def test_composing_in_shards_matches_composing_whole():
    rng = np.random.default_rng(0)
    volume = np.where(rng.random((24, 20, 28)) < 0.3, BONE, BACKGROUND).astype(np.uint8)
    volume[rng.random(volume.shape) < 0.05] = TOOTH
    labeled = rng.random(volume.shape) < 0.01
    whole = compose(volume, labeled, (24, 20, 28), min_voxels=6)
    for shard in [(8, 8, 8), (7, 5, 9), (24, 4, 28)]:
        assert np.array_equal(compose(volume, labeled, shard, min_voxels=6), whole), (
            shard
        )


def test_shard_grid_counts_partial_shards():
    assert shard_grid((513, 512, 1)) == (2, 1, 1)
    assert list(itertools.islice(shard_boxes((40, 40, 40), (16, 16, 16)), 2))[1] == (
        0,
        0,
        16,
        16,
        16,
        32,
    )
