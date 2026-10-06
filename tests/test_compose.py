"""
The final segmentation: labels overrule the prediction, complete ROIs are
background, and specks are removed across shard boundaries, within each
job's memory budget.
"""

import io
import itertools
import tracemalloc

import numpy as np
import pytest
from scipy import ndimage

from ml4paleo.labels import BACKGROUND, FIRST_CLASS
from ml4paleo.segmentation.compose import (
    TooLarge,
    apply_shard,
    find_specks,
    label_shard,
    merge,
    seam_pairs,
    seams,
    shard_grid,
    shard_specks,
    slab_depth,
)
from ml4paleo.segmentation.predict import shard_boxes

BONE, TOOTH = 2, 3


def test_labels_overrule_the_prediction_and_complete_rois_are_background():
    prediction = np.full((4, 4, 4), BONE, dtype=np.uint8)
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[0] = TOOTH
    # The box starts at z 10; the ROI covers its first two planes.
    merged = merge(prediction, labels, (10, 0, 0, 14, 4, 4), [(8, 0, 0, 12, 9, 9)])
    assert (merged[0] == TOOTH).all()  # labeled, inside the complete ROI
    assert (merged[1] == BACKGROUND).all()  # unlabeled, inside it
    assert (merged[2:] == BONE).all()  # outside: the prediction


def compose(volume, labeled, shard, min_voxels, slab=2):
    """Run the per-shard steps over a whole volume, as the jobs would."""
    boxes = shard_boxes(volume.shape, shard)
    regions = [tuple(slice(b[a], b[a + 3]) for a in range(3)) for b in boxes]
    pieces = [
        label_shard(
            volume[r], labeled[r], min_voxels, seams(box, volume.shape), slab=slab
        )
        for box, r in zip(boxes, regions, strict=True)
    ]
    joined = find_specks(
        [p.summary for p in pieces], shard_grid(volume.shape, shard), min_voxels
    )
    final = np.empty_like(volume)
    for region, found, seam in zip(regions, pieces, joined.specks, strict=True):
        specks = shard_specks(found.specks, found.seam_ids, seam)
        final[region] = apply_shard(volume[region].copy(), specks, slab=slab)
    return final


def reference(volume, labeled, min_voxels):
    """Specks removed from the whole volume at once, with scipy alone."""
    final = volume.copy()
    for value in np.unique(volume):
        if value < FIRST_CLASS:
            continue
        # scipy's default structure in 3D is 6-connected.
        ids, count = ndimage.label(volume == value)  # type: ignore[misc]
        sizes = np.bincount(ids.ravel(), minlength=count + 1)
        held = np.bincount(ids[labeled], minlength=count + 1) > 0
        speck = (sizes < min_voxels) & ~held
        speck[0] = False
        final[speck[ids]] = BACKGROUND
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


def test_specks_inside_a_shard_are_decided_there():
    # Two shards along x, with a seam between x 7 and x 8.
    volume = np.full((6, 6, 16), BACKGROUND, dtype=np.uint8)
    volume[0, 0, 0] = BONE  # on the volume's corner, which is no seam
    volume[1, 1, 2] = BONE  # inside the first shard
    volume[3, 3, 6:10] = BONE  # a bar across the seam
    labeled = np.zeros(volume.shape, dtype=bool)
    first = seams((0, 0, 0, 6, 6, 8), volume.shape)
    assert first == (False, False, False, False, False, True)
    pieces = label_shard(volume[:, :, :8], labeled[:, :, :8], 3, first)
    # The specks (pieces 1 and 2) are decided here; the bar (piece 3) waits
    # for the merge as seam piece 1, with its 2 voxels in this shard.
    assert pieces.specks.tolist() == [False, True, True, False]
    assert pieces.seam_ids.tolist() == [3]
    summary = np.load(io.BytesIO(pieces.summary))
    assert sorted(summary.files) == ["classes", "last_x", "sizes"]
    assert summary["sizes"].tolist() == [0, 2]
    assert summary["last_x"][3, 3] == 1 and summary["last_x"].sum() == 1
    final = compose(volume, labeled, (6, 6, 8), min_voxels=3)
    assert (final[3, 3, 6:10] == BONE).all()
    assert final[0, 0, 0] == final[1, 1, 2] == BACKGROUND


def test_pieces_are_joined_once_per_pair_however_much_they_touch():
    a = np.ones((64, 64), dtype=np.uint16)
    a[:, 32:] = 2  # bone on the left half, tooth on the right
    b = np.ones((64, 64), dtype=np.uint16)
    b[32:] = 2  # two bone pieces, top and bottom
    classes_a = np.array([0, BONE, TOOTH], dtype=np.uint8)
    classes_b = np.array([0, BONE, BONE], dtype=np.uint8)
    first, second = seam_pairs(a, classes_a, b, classes_b)
    # 2048 voxels of bone touch bone across the seam, in two pairs of pieces.
    assert sorted(zip(first.tolist(), second.tolist(), strict=True)) == [
        (1, 1),
        (1, 2),
    ]
    # One piece filling both sides of a seam is one pair.
    volume = np.full((8, 8, 16), BONE, dtype=np.uint8)
    pieces = [
        label_shard(
            volume[:, :, x : x + 8],
            np.zeros((8, 8, 8), dtype=bool),
            10,
            seams((0, 0, x, 8, 8, x + 8), volume.shape),
        )
        for x in (0, 8)
    ]
    assert find_specks([p.summary for p in pieces], (1, 1, 2), 10).pairs == 1


def _within(budget, held, step, *args, **kwargs):
    """Run a step, checking that it and what it was given fit `budget`."""
    tracemalloc.start()
    try:
        result = step(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert held + peak <= budget, (step.__name__, held + peak, budget)
    return result


@pytest.mark.parametrize("budget_mib", [1, 3, 6, 12])
def test_noisy_volumes_stay_within_the_budget_or_refuse(budget_mib):
    # Each voxel background, bone, or tooth at random: hundreds of thousands
    # of pieces, and seams thick with them.
    budget = budget_mib * 1024**2
    rng = np.random.default_rng(2)
    volume = rng.choice(
        np.array([BACKGROUND, BONE, TOOTH], dtype=np.uint8),
        size=(128, 128, 128),
        p=[0.5, 0.25, 0.25],
    )
    labeled = np.zeros(volume.shape, dtype=bool)
    labeled[:8] = rng.random((8, 128, 128)) < 0.1
    shard = (64, 64, 64)
    slab = slab_depth(budget, shard)
    boxes = shard_boxes(volume.shape, shard)
    regions = [tuple(slice(b[a], b[a + 3]) for a in range(3)) for b in boxes]
    try:
        pieces = [
            _within(
                budget,
                volume[r].nbytes + labeled[r].nbytes,
                label_shard,
                volume[r],
                labeled[r],
                20,
                seams(box, volume.shape),
                slab=slab,
                budget_bytes=budget,
            )
            for box, r in zip(boxes, regions, strict=True)
        ]
        joined = _within(
            budget,
            0,
            find_specks,
            # As the merge job reads them, one at a time.
            (bytes(p.summary) for p in pieces),
            shard_grid(volume.shape, shard),
            20,
            budget_bytes=budget,
        )
        final = np.empty_like(volume)
        for region, found, seam in zip(regions, pieces, joined.specks, strict=True):
            specks = shard_specks(found.specks, found.seam_ids, seam)
            block = volume[region].copy()
            final[region] = _within(
                budget,
                block.nbytes + specks.nbytes,
                apply_shard,
                block,
                specks,
                slab=slab,
                budget_bytes=budget,
            )
    except TooLarge as exc:
        assert "a job may use on this worker" in str(exc)
        assert budget_mib < 12
        return
    assert budget_mib > 1
    assert np.array_equal(final, reference(volume, labeled, 20))
    # Joining refuses too, given less than it needs.
    with pytest.raises(TooLarge, match="Joining the pieces"):
        find_specks(
            (p.summary for p in pieces),
            shard_grid(volume.shape, shard),
            20,
            budget_bytes=512 * 1024,
        )


def test_composing_in_shards_matches_scipy_on_the_whole_volume():
    rng = np.random.default_rng(0)
    volume = np.where(rng.random((24, 20, 28)) < 0.3, BONE, BACKGROUND).astype(np.uint8)
    volume[rng.random(volume.shape) < 0.05] = TOOTH
    labeled = rng.random(volume.shape) < 0.01
    want = reference(volume, labeled, 6)
    assert not np.array_equal(want, volume)  # there are specks to remove
    for shard in [(24, 20, 28), (8, 8, 8), (7, 5, 9), (24, 4, 28), (1, 20, 28)]:
        assert np.array_equal(compose(volume, labeled, shard, 6), want), shard
    # Random volumes, labels, minimums, and shards, partial and one voxel thin.
    for trial in range(60):
        shape = tuple(int(n) for n in rng.integers(1, 12, size=3))
        volume = rng.choice(
            np.array([BACKGROUND, BONE, TOOTH], dtype=np.uint8),
            size=shape,
            p=[0.4, 0.3, 0.3],
        )
        labeled = rng.random(shape) < 0.03
        min_voxels = int(rng.integers(0, 10))
        want = reference(volume, labeled, min_voxels)
        for _ in range(3):
            shard = tuple(int(rng.integers(1, n + 1)) for n in shape)
            got = compose(
                volume, labeled, shard, min_voxels, slab=int(rng.integers(1, 4))
            )
            assert np.array_equal(got, want), (trial, shape, shard, min_voxels)


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
