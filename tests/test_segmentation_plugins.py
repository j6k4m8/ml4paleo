"""
Segmentation plugins: training sets built from ROIs and sparse labels, the
random forest plugin trained and scored on a synthetic volume, and
predicting with it block by block.
"""

import json
import subprocess
import sys

import joblib
import numpy as np
import pytest

from ml4paleo.labels import LABEL_CHUNK_ZYX, PLUGIN_IGNORE
from ml4paleo.segmentation.dataset import RoiSpec, TrainingSet, tile_for, tiles
from ml4paleo.segmentation.plugin import CropCost, get_plugin, plugins

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
    # it gives none either; (0, 0, 1) has the free background label, in the
    # first of its tiles.
    assert len(train) == 2
    free = train[1]
    assert free.targets.shape == (32, 32, 26)
    assert int((free.targets != PLUGIN_IGNORE).sum()) == 1
    assert free.targets[5, 5, 70 - 64] == 0
    # At the image's edges, the crop still has the full halo: edge voxels
    # repeated, as at prediction time.
    assert free.image.shape == (1, 40, 40, 34)
    assert free.interior == (slice(4, 36), slice(4, 36), slice(4, 30))
    edged = np.pad(image[:, :36, :36, 60:], ((0, 0), (4, 0), (4, 0), (0, 4)), "edge")
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


def test_missing_label_blobs_say_so(tmp_path):
    from ml4paleo.segmentation.dataset import BlobLabels, MissingLabels
    from ml4paleo.storage import StorageGrant

    grant = StorageGrant(url=f"file://{tmp_path}", access="r")
    labels = BlobLabels(grant, {(0, 0, 0): "ab" * 32})
    assert labels.chunk((0, 0, 1)) is None
    with pytest.raises(MissingLabels):
        labels.chunk((0, 0, 0))


def test_tiles_cover_a_box_without_overlap():
    box = (0, 5, 10, 70, 37, 11)
    seen = np.zeros((70, 32, 1), dtype=int)
    for t in tiles(box, 32):
        assert all(t[a + 3] - t[a] <= 32 for a in range(3))
        seen[t[0] : t[3], t[1] - 5 : t[4] - 5, t[2] - 10 : t[5] - 10] += 1
    assert (seen == 1).all()


# 1.1 and 2.6 round down (4.4 and 10.4), so the halo is as small as can be.
@pytest.mark.parametrize("sigma_max", [1.0, 1.1, 2.0, 2.6, 3.0])
def test_random_forest_halo_covers_every_feature(sigma_max):
    from ml4paleo.segmentation.plugins.rf import features, halo_for

    image = np.random.default_rng(0).random((1, 48, 48, 48)).astype(np.float32)
    whole = features(image, sigma_max)[20:28, 20:28, 20:28]
    h = halo_for(sigma_max)
    block = image[:, 20 - h : 28 + h, 20 - h : 28 + h, 20 - h : 28 + h]
    part = features(block, sigma_max)[h:-h, h:-h, h:-h]
    np.testing.assert_array_equal(part, whole)


def test_tiles_fit_the_memory_budget():
    cost = CropCost(halo=10, bytes_per_voxel=100)
    assert tile_for(100 * 100**3, cost) == 80
    assert tile_for(100 * 100**3 - 1, cost) == 79
    assert tile_for(1024, cost) == 32
    assert tile_for(10**15, cost) == 256


@pytest.mark.parametrize(("channels", "sigma_max"), [(1, 1.0), (2, 3.0), (1, 8.0)])
def test_random_forest_counts_its_features(channels, sigma_max):
    from ml4paleo.segmentation.plugins.rf import feature_count, features

    image = np.zeros((channels, 8, 8, 8), dtype=np.float32)
    assert features(image, sigma_max).shape[-1] == feature_count(channels, sigma_max)
    plugin = get_plugin("rf")()
    count = feature_count(channels, sigma_max)
    cost = plugin.crop_cost(plugin.Params(sigma_max=sigma_max), channels)
    # Each scale scikit-image is working on holds about 160 bytes a voxel.
    assert cost.bytes_per_voxel == max(16 * count, 4 * count + 160)
    # Threads compute scales at once, each with its own working memory.
    many = plugin.crop_cost(plugin.Params(sigma_max=sigma_max), channels, threads=16)
    assert many.bytes_per_voxel >= cost.bytes_per_voxel


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
        memory_budget_bytes = 4 * 1024**3

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
    meta = json.loads((tmp_path / "model.json").read_text())
    assert meta["window"] == [200.0, 800.0]
    assert set(result.samples) == {0, 1}
    assert result.metrics["validation_crops"] >= 1
    assert result.metrics["classes"][str(BONE)]["dice"] > 0.8
    # Models from before model.json kept their feature count still load.
    features = meta.pop("features")
    (tmp_path / "model.json").write_text(json.dumps(meta))
    predictor = plugin.load(tmp_path)
    # Predicting holds the features several times over, and more with each
    # scale computed at once (one per thread, up to this model's two).
    held = {}
    for threads in (1, 2, 8):
        predictor.threads = threads
        held[threads] = predictor.bytes_per_voxel
    assert 16 * features <= held[1] < held[2] == held[8]
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


def test_random_forest_samples_fit_the_memory_budget(tmp_path):
    from ml4paleo.segmentation.plugins.rf import samples_that_fit

    plugin = get_plugin("rf")()
    params = plugin.Params(
        n_estimators=10, max_depth=4, samples_per_class=2000, sigma_max=1.0
    )
    assert samples_that_fit(10**9, 2, 5, 1, params) == 2000
    tight = samples_that_fit(100_000, 2, 5, 1, params)
    assert 100 <= tight < 2000
    assert tight < samples_that_fit(200_000, 2, 5, 1, params)
    assert samples_that_fit(100_000, 2, 5, 4, params) < tight
    # Without max_depth to stop them, trees grow with the samples.
    deep = plugin.Params(n_estimators=10, max_depth=30, samples_per_class=2000)
    assert samples_that_fit(100_000, 2, 5, 1, deep) < tight

    image, truth = synthetic()
    labels = np.where(truth, BONE, 1).astype(np.uint8)
    source = DictLabels(labels)
    data = TrainingSet(image, source, source.chunks, [], [BONE], (200.0, 800.0))

    class Ctx:
        threads = 1
        memory_budget_bytes = 200_000

        def progress(self, fraction, message=None):
            pass

        def check(self):
            pass

    result = plugin.train(data, params, tmp_path, Ctx())
    assert result.metrics["samples_per_class_used"] == tight
    assert result.samples == {0: tight, 1: tight}
    # With too little memory for even a small forest, training refuses.
    Ctx.memory_budget_bytes = 50_000
    with pytest.raises(ValueError, match="memory"):
        plugin.train(data, params, tmp_path, Ctx())


def test_training_needs_two_classes(tmp_path):
    image, _ = synthetic()
    plugin = get_plugin("rf")()

    class Ctx:
        threads = 1
        memory_budget_bytes = 4 * 1024**3

        def progress(self, fraction, message=None):
            pass

        def check(self):
            pass

    # It says which class has labels, by value, and what to do about it.
    for value, only in ((BONE, "class 2"), (1, "background")):
        labels = np.zeros(SHAPE, dtype=np.uint8)
        labels[20, 20, 20:30] = value
        source = DictLabels(labels)
        data = TrainingSet(image, source, source.chunks, [], [BONE], (200.0, 800.0))
        with pytest.raises(ValueError, match=f"two classes.*only {only} has any"):
            plugin.train(data, plugin.Params(sigma_max=1.0), tmp_path, Ctx())


def test_plugins_are_listed_and_check_their_params():
    assert "rf" in plugins()
    params = get_plugin("rf").Params
    with pytest.raises(ValueError):
        params(n_estimators=0)
    with pytest.raises(ValueError):
        params(sigma_max=9.0)
    for seed in (-1, 2**32):
        with pytest.raises(ValueError):
            params(seed=seed)
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


WINDOW = (200.0, 800.0)


def train_forest(path, image, sigma_max=1.0):
    """A small forest trained on a few strokes over the synthetic balls."""
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[20, 18:23, 25] = BONE
    labels[55, 45, 55:66] = BONE
    labels[5, 5:30, 5] = 1
    labels[70, 10, 10:80] = 1
    source = DictLabels(labels)
    data = TrainingSet(image, source, source.chunks, [], [BONE], WINDOW)
    plugin = get_plugin("rf")()

    class Ctx:
        threads = 1
        memory_budget_bytes = 4 * 1024**3

        def progress(self, fraction, message=None):
            pass

        def check(self):
            pass

    params = plugin.Params(
        n_estimators=20, max_depth=8, samples_per_class=2000, sigma_max=sigma_max
    )
    plugin.train(data, params, path, Ctx())
    predictor = plugin.load(path)
    # As a job with one CPU would, so blocks come out the same anywhere.
    predictor.threads = 1
    return predictor


def new_prediction(path):
    from ml4paleo.segmentation.predict import create_prediction
    from ml4paleo.storage import StorageGrant

    return create_prediction(StorageGrant(url=f"file://{path}", access="rw"), SHAPE)


def predict_at_once(predictor, image, box):
    """A box predicted as one block, cut with its halo from the padded image."""
    from ml4paleo.labels import from_plugin_space
    from ml4paleo.segmentation.dataset import normalize

    h = predictor.halo
    padded = np.pad(image, [(0, 0)] + [(h, h)] * 3, mode="edge")
    block = padded[
        (slice(None), *(slice(box[a], box[a + 3] + 2 * h) for a in range(3)))
    ]
    probabilities = predictor.predict_block(normalize(block, WINDOW))
    classes = from_plugin_space(probabilities.argmax(axis=0).astype(np.uint8), [BONE])
    uncertainty = np.round(255 * (1 - probabilities.max(axis=0))).astype(np.uint8)
    return classes, uncertainty


class Reads:
    """An image that keeps the shape of everything read from it."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.shapes = []

    def __getitem__(self, selection):
        part = self.array[selection]
        self.shapes.append(part.shape)
        return part


@pytest.fixture(scope="module")
def forest(tmp_path_factory):
    return train_forest(tmp_path_factory.mktemp("forest"), synthetic()[0], 2.0)


def test_predictions_cover_shards(tmp_path):
    from ml4paleo.segmentation.predict import predict_box, shard_boxes

    image, truth = synthetic()
    predictor = train_forest(tmp_path / "model", image)
    boxes = shard_boxes(SHAPE, (32, 32, 32))
    covered = np.zeros(SHAPE, dtype=int)
    for b in boxes:
        covered[b[0] : b[3], b[1] : b[4], b[2] : b[5]] += 1
    assert (covered == 1).all()

    group = new_prediction(tmp_path / "prediction")
    for b in boxes:
        predict_box(predictor, image, b, WINDOW, [BONE], group, 4 * 1024**3)
    predicted = np.asarray(group["class"][:])
    assert set(np.unique(predicted)) <= {1, BONE}
    assert ((predicted == BONE) == truth).mean() > 0.95
    assert np.asarray(group["uncertainty"][:]).max() <= 255


# 1 byte and 1.5 MB: the box's outputs would take more than a quarter of
# the budget, so they're written a layer of MIN_BLOCK blocks at a time.
# 32 MiB: they're kept whole and written once, with several blocks. 1 GB:
# the box is one block.
@pytest.mark.parametrize(
    ("budget", "whole"), [(1, False), (1_500_000, False), (2**25, True), (10**9, True)]
)
def test_predicting_block_by_block_matches_the_box_at_once(
    tmp_path, monkeypatch, forest, budget, whole
):
    from ml4paleo.segmentation import predict

    image, _ = synthetic()
    # At the image's edges on some sides only.
    box = (8, 0, 30, 80, 70, 90)
    reads = Reads(image)
    written = []
    write_box = predict.write_box

    def counted(group, at, *arrays):
        written.append(at)
        write_box(group, at, *arrays)

    monkeypatch.setattr(predict, "write_box", counted)
    group = new_prediction(tmp_path / "prediction")
    predict.predict_box(forest, reads, box, WINDOW, [BONE], group, budget)
    classes, uncertainty = predict_at_once(forest, image, box)
    region = tuple(slice(box[a], box[a + 3]) for a in range(3))
    np.testing.assert_array_equal(group["class"][region], classes)
    np.testing.assert_array_equal(group["uncertainty"][region], uncertainty)
    assert not np.asarray(group["class"][:8]).any()
    assert not np.asarray(group["class"][:, :, :30]).any()
    side = predict.MIN_BLOCK
    layers = [(z, 0, 30, min(z + side, 80), 70, 90) for z in range(8, 80, side)]
    assert written == ([box] if whole else layers)
    assert (len(reads.shapes) == 1) == (budget == 10**9)


def test_prediction_blocks_fit_the_memory_budget():
    from types import SimpleNamespace

    from ml4paleo.segmentation.predict import MIN_BLOCK, block_for

    predictor = SimpleNamespace(halo=10, bytes_per_voxel=84)
    # With one channel's image, 100 bytes a voxel: (80 + 2 * 10)³ voxels fit.
    assert block_for(100 * 100**3, 1, predictor) == 80
    # Each channel's image takes room too.
    assert block_for(100 * 100**3, 4, predictor) == 67
    assert block_for(0, 1, predictor) == MIN_BLOCK
    assert block_for(10**15, 1, predictor) == 512


def test_predicting_an_image_with_several_channels(tmp_path):
    from ml4paleo.segmentation.predict import MIN_BLOCK, predict_box

    single, truth = synthetic()
    noise = np.random.default_rng(1).integers(0, 1000, (1, *SHAPE), dtype=np.uint16)
    image = np.concatenate([single, single.max() - single, noise])
    predictor = train_forest(tmp_path / "model", image)
    # Around the second ball, out to the image's far edges.
    box = (40, 30, 40, 80, 70, 90)
    region = tuple(slice(box[a], box[a + 3]) for a in range(3))
    classes, uncertainty = predict_at_once(predictor, image, box)
    assert ((classes == BONE) == truth[region]).mean() > 0.95
    for budget in (1, 10**9):
        reads = Reads(image)
        group = new_prediction(tmp_path / f"prediction-{budget}")
        predict_box(predictor, reads, box, WINDOW, [BONE], group, budget)
        np.testing.assert_array_equal(group["class"][region], classes)
        np.testing.assert_array_equal(group["uncertainty"][region], uncertainty)
        assert all(shape[0] == 3 for shape in reads.shapes)
        if budget == 1:
            # Only a block and its halo are read at a time.
            side = MIN_BLOCK + 2 * predictor.halo
            assert len(reads.shapes) > 1
            assert all(max(shape[1:]) <= side for shape in reads.shapes)
