"""
Segmentation plugins: training sets built from ROIs and sparse labels, and
the random forest plugin trained and scored on a synthetic volume.
"""

import subprocess
import sys

import joblib
import numpy as np
import pytest

from ml4paleo.labels import LABEL_CHUNK_ZYX, PLUGIN_IGNORE
from ml4paleo.segmentation.dataset import RoiSpec, TrainingSet, tiles
from ml4paleo.segmentation.plugin import get_plugin, plugins

SHAPE = (80, 70, 90)  # (z, y, x): edge chunks are partial on every axis
BONE = 2


class DictLabels:
    """Label chunks held in memory, built from a full label volume."""

    def __init__(self, volume: np.ndarray):
        self.chunks: dict[tuple[int, int, int], np.ndarray] = {}
        cz, cy, cx = LABEL_CHUNK_ZYX
        for z in range(0, volume.shape[0], cz):
            for y in range(0, volume.shape[1], cy):
                for x in range(0, volume.shape[2], cx):
                    part = volume[z : z + cz, y : y + cy, x : x + cx]
                    if not part.any():
                        continue
                    chunk = np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
                    chunk[: part.shape[0], : part.shape[1], : part.shape[2]] = part
                    self.chunks[(z // cz, y // cy, x // cx)] = chunk

    def chunk(self, key):
        return self.chunks.get(key)


def synthetic(seed=0):
    """Bright balls (bone) in a dim, noisy background."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(SHAPE)
    truth = np.zeros(SHAPE, dtype=bool)
    for center, radius in [((20, 20, 25), 9), ((55, 45, 60), 12), ((30, 50, 75), 7)]:
        truth |= (
            sum((g - c) ** 2 for g, c in zip((z, y, x), center, strict=True))
            <= radius**2
        )
    image = np.where(truth, 800.0, 200.0) + rng.normal(0, 60, SHAPE)
    return image.astype(np.uint16)[None], truth


def test_crops_follow_roi_status_split_and_free_labels():
    image, _ = synthetic()
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[10:12, 10:12, 10:12] = BONE  # inside the complete ROI
    labels[5, 5, 70] = 1  # free label, outside every ROI
    labels[60, 60, 5] = BONE  # inside the validation ROI
    rois = [
        RoiSpec((8, 8, 8, 16, 16, 16), "complete", "train"),
        RoiSpec((40, 0, 0, 41, 64, 64), "open", "train"),
        RoiSpec((56, 56, 0, 64, 64, 8), "complete", "val"),
        RoiSpec((0, 0, 0, 4, 4, 4), "skipped", "train"),
    ]
    source = DictLabels(labels)
    data = TrainingSet(
        image, source, source.chunks, rois, [BONE], (200.0, 800.0), tile=32
    )
    train = list(data.crops("train", halo=4))
    # The complete ROI: everything known, bone where painted, background elsewhere.
    roi_crop = train[0]
    assert roi_crop.targets.shape == (8, 8, 8)
    assert (roi_crop.targets != PLUGIN_IGNORE).all()
    assert (roi_crop.targets[2:4, 2:4, 2:4] == 1).all() and roi_crop.targets.sum() == 8
    assert roi_crop.image.shape == (1, 16, 16, 16)
    assert roi_crop.interior == (slice(4, 12), slice(4, 12), slice(4, 12))
    # The open ROI has no labels, so it gives no crop. Of the labeled chunks,
    # (0, 0, 0) only has labels inside the complete and validation ROIs, so
    # it gives none either; (0, 0, 1) has the free background label.
    assert len(train) == 2
    free = train[1]
    assert free.targets.shape == (64, 64, 26)
    assert int((free.targets != PLUGIN_IGNORE).sum()) == 1
    assert free.targets[5, 5, 70 - 64] == 0
    # At the image's edges, the crop still has the full halo: edge voxels
    # repeated, as at prediction time.
    assert free.image.shape == (1, 72, 72, 34)
    assert free.interior == (slice(4, 68), slice(4, 68), slice(4, 30))
    edged = np.pad(image[:, :68, :68, 60:], ((0, 0), (4, 0), (4, 0), (0, 4)), "edge")
    np.testing.assert_allclose(free.image, (edged - 200.0) / 600.0, rtol=1e-6)
    val = list(data.crops("val", halo=4))
    assert len(val) == 1 and int((val[0].targets == 1).sum()) == 1
    assert (val[0].targets != PLUGIN_IGNORE).all()


def test_validation_rois_are_held_out_where_they_overlap_training():
    image, _ = synthetic()
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[10, 10, 10] = BONE  # training only
    labels[20, 20, 20] = BONE  # in both ROIs
    labels[28, 28, 28] = BONE  # validation only
    rois = [
        RoiSpec((8, 8, 8, 24, 24, 24), "complete", "train"),
        RoiSpec((16, 16, 16, 32, 32, 32), "open", "val"),
    ]
    source = DictLabels(labels)
    data = TrainingSet(
        image, source, source.chunks, rois, [BONE], (200.0, 800.0), tile=32
    )
    [train] = data.crops("train", halo=2)
    # The overlap is ignored, labeled or not, though the training ROI is complete.
    assert (train.targets[8:, 8:, 8:] == PLUGIN_IGNORE).all()
    assert int((train.targets != PLUGIN_IGNORE).sum()) == 16**3 - 8**3
    assert int((train.targets == 1).sum()) == 1 and train.targets[2, 2, 2] == 1
    [val] = data.crops("val", halo=2)
    assert int((val.targets == 1).sum()) == 2
    assert int((val.targets != PLUGIN_IGNORE).sum()) == 2


def test_overlapping_training_rois_count_each_voxel_once():
    image, _ = synthetic()
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[11, 11, 11] = 1  # only in the open ROI
    labels[17, 17, 17] = BONE  # in both
    rois = [
        RoiSpec((10, 10, 10, 20, 20, 20), "open", "train"),
        RoiSpec((15, 15, 15, 25, 25, 25), "complete", "train"),
    ]
    source = DictLabels(labels)
    data = TrainingSet(
        image, source, source.chunks, rois, [BONE], (200.0, 800.0), tile=32
    )
    first, second = data.crops("train", halo=2)
    # In the open ROI, the part inside the complete ROI is complete too.
    assert (first.targets[5:, 5:, 5:] != PLUGIN_IGNORE).all()
    assert int((first.targets != PLUGIN_IGNORE).sum()) == 5**3 + 1
    # The complete ROI leaves the overlap to the open ROI, which came first.
    assert (second.targets[:5, :5, :5] == PLUGIN_IGNORE).all()
    assert int((second.targets != PLUGIN_IGNORE).sum()) == 10**3 - 5**3
    known = sum(int((c.targets != PLUGIN_IGNORE).sum()) for c in (first, second))
    assert known == 10**3 + 1
    assert sum(int((c.targets == 1).sum()) for c in (first, second)) == 1


def test_rois_are_cut_to_the_image():
    image, _ = synthetic()
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[75, 65, 85] = BONE
    rois = [
        RoiSpec((70, 60, 80, 100, 100, 100), "complete", "train"),
        RoiSpec((90, 0, 0, 100, 10, 10), "complete", "train"),
    ]
    source = DictLabels(labels)
    data = TrainingSet(image, source, source.chunks, rois, [BONE], (200.0, 800.0))
    assert [roi.bbox for roi in data.rois] == [(70, 60, 80, 80, 70, 90)]
    [crop] = data.crops("train", halo=3)
    assert crop.targets.shape == (10, 10, 10)
    assert crop.image.shape == (1, 16, 16, 16)
    assert int((crop.targets == 1).sum()) == 1


def test_labels_assemble_across_chunks_and_edges():
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[60:70, 60:68, 60:90] = BONE
    source = DictLabels(labels)
    image, _ = synthetic()
    data = TrainingSet(image, source, source.chunks, [], [BONE], (0, 1))
    box = (58, 58, 58, 72, 70, 90)
    assert np.array_equal(data.read_labels(box), labels[58:72, 58:70, 58:90])


def test_tiles_cover_a_box_without_overlap():
    box = (0, 5, 10, 70, 37, 11)
    seen = np.zeros((70, 32, 1), dtype=int)
    for t in tiles(box, 32):
        assert all(t[a + 3] - t[a] <= 32 for a in range(3))
        seen[t[0] : t[3], t[1] - 5 : t[4] - 5, t[2] - 10 : t[5] - 10] += 1
    assert (seen == 1).all()


@pytest.mark.parametrize("sigma_max", [1.0, 2.0, 3.0])
def test_random_forest_halo_covers_every_feature(sigma_max):
    from ml4paleo.segmentation.plugins.rf import features, halo_for

    image = np.random.default_rng(0).random((1, 48, 48, 48)).astype(np.float32)
    whole = features(image, sigma_max)[20:28, 20:28, 20:28]
    h = halo_for(sigma_max)
    block = image[:, 20 - h : 28 + h, 20 - h : 28 + h, 20 - h : 28 + h]
    part = features(block, sigma_max)[h:-h, h:-h, h:-h]
    np.testing.assert_allclose(part, whole, atol=1e-6)


def test_random_forest_learns_from_sparse_labels(tmp_path):
    image, truth = synthetic()
    labels = np.zeros(SHAPE, dtype=np.uint8)
    # Sparse strokes: a few voxels of bone and of background.
    labels[20, 18:23, 25] = BONE
    labels[55, 45, 55:66] = BONE
    labels[5, 5:30, 5] = 1
    labels[70, 10, 10:80] = 1
    # A complete validation ROI around the second ball, labeled inside it.
    val_box = (45, 35, 50, 66, 56, 71)
    region = tuple(slice(val_box[a], val_box[a + 3]) for a in range(3))
    labels[region] = np.where(truth[region], BONE, 0)
    rois = [RoiSpec(val_box, "complete", "val")]
    source = DictLabels(labels)
    data = TrainingSet(image, source, source.chunks, rois, [BONE], (200.0, 800.0))
    plugin = get_plugin("rf")()
    params = plugin.Params(
        n_estimators=30, max_depth=10, samples_per_class=2000, sigma_max=2.0
    )

    class Ctx:
        threads = 2

        def __init__(self):
            self.fractions = []

        def progress(self, fraction, message=None):
            self.fractions.append(fraction)

        def check(self):
            pass

    ctx = Ctx()
    result = plugin.train(data, params, tmp_path, ctx)
    assert ctx.fractions[-1] == 1.0
    # The forest fits on the context's threads, not every core.
    assert joblib.load(tmp_path / "forest.joblib").n_jobs == 2
    assert set(result.samples) == {0, 1}
    assert result.metrics["validation_crops"] >= 1
    assert result.metrics["classes"][str(BONE)]["dice"] > 0.8
    predictor = plugin.load(tmp_path)
    h = predictor.halo
    block = np.pad(
        image[:, 10:42, 10:42, 10:42].astype(np.float32),
        ((0, 0), (h, h), (h, h), (h, h)),
        mode="reflect",
    )
    block = (block - 200.0) / 600.0
    proba = predictor.predict_block(block)
    assert proba.shape == (2, 32, 32, 32)
    agreement = (proba.argmax(axis=0) == truth[10:42, 10:42, 10:42]).mean()
    assert agreement > 0.9


def test_training_needs_two_classes(tmp_path):
    image, _ = synthetic()
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[20, 20, 20:30] = BONE
    source = DictLabels(labels)
    data = TrainingSet(image, source, source.chunks, [], [BONE], (200.0, 800.0))
    plugin = get_plugin("rf")()

    class Ctx:
        threads = 1

        def progress(self, fraction, message=None):
            pass

        def check(self):
            pass

    with pytest.raises(ValueError, match="two classes"):
        plugin.train(data, plugin.Params(sigma_max=1.0), tmp_path, Ctx())


def test_plugins_are_listed_and_check_their_params():
    assert "rf" in plugins()
    params = get_plugin("rf").Params
    with pytest.raises(ValueError):
        params(n_estimators=0)
    with pytest.raises(ValueError, match="No segmentation plugin"):
        get_plugin("nope")


def test_listing_plugins_needs_no_scikit():
    code = (
        "import sys\n"
        "from ml4paleo.segmentation.plugin import plugins\n"
        "plugins()['rf'].Params()\n"
        "print(any(m.split('.')[0] in ('sklearn', 'skimage') for m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
