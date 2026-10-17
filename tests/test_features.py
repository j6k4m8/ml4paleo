"""Cache correctness, invalidation, bounds, cancellation and old models."""

import concurrent.futures
import json
import time

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml4paleo.segmentation.dataset import normalize, read_box
from ml4paleo.segmentation.features import (
    FeatureCache,
    FeatureRecipe,
    FeatureVolume,
    extractor,
    multiscale_recipe,
)
from ml4paleo.segmentation.plugins.rf import RandomForestPlugin, features


def test_cached_features_match_original_across_chunks_and_edges(tmp_path):
    image = np.random.default_rng(0).integers(0, 500, (2, 18, 20, 23), dtype=np.uint16)
    recipe = multiscale_recipe(2)
    volume = FeatureVolume(
        image, "scan-a", (0, 500), recipe, FeatureCache(tmp_path), 2, tile=8
    )
    box = (0, 3, 5, 18, 19, 23)
    h = extractor(recipe).halo
    reference = features(normalize(read_box(image, box, h), (0, 500)), 2, 2)[
        h:-h, h:-h, h:-h
    ]
    actual = volume.read(box)
    np.testing.assert_array_equal(actual, reference)
    # A new process has no decoded cache but reads the stored chunks.
    restored = FeatureVolume(
        image,
        "scan-a",
        (0, 500),
        recipe,
        FeatureCache(tmp_path, memory_bytes=0),
        1,
        tile=8,
    )
    restored.extractor.compute = lambda *_: pytest.fail("Recomputed persisted features")
    np.testing.assert_array_equal(restored.read(box), reference)


def test_image_window_and_recipe_have_distinct_entries(tmp_path):
    image = np.random.default_rng(0).random((1, 8, 8, 8)).astype(np.float32)
    cache = FeatureCache(tmp_path)
    box = (0, 0, 0, 8, 8, 8)
    for source, window, recipe in [
        ("a", (0, 1), multiscale_recipe(1)),
        ("b", (0, 1), multiscale_recipe(1)),
        ("a", (0, 2), multiscale_recipe(1)),
        ("a", (0, 1), multiscale_recipe(1, texture=False)),
    ]:
        FeatureVolume(image, source, window, recipe, cache, 1).read(box)
    assert len(list(tmp_path.glob("*.f32.zst"))) == 4
    with pytest.raises(ValueError, match="Unsupported"):
        extractor(FeatureRecipe(name="multiscale", version="999", params={}))


def test_bounds_corruption_and_idle_release(tmp_path):
    value = np.random.default_rng(4).random((4, 4, 4, 2)).astype(np.float32)
    cache = FeatureCache(
        tmp_path, memory_bytes=value.nbytes, disk_bytes=650, idle_seconds=0
    )
    for i in range(4):
        cache.get({"key": i}, value.shape, lambda: value.copy(), lambda: None)
        assert cache.retained_bytes <= value.nbytes
        assert sum(p.stat().st_size for p in tmp_path.glob("*.f32.zst")) <= 650
    cache.prune_memory()
    assert cache.retained_bytes == 0
    path = next(tmp_path.glob("*.f32.zst"))
    path.write_bytes(b"broken cache")
    np.testing.assert_array_equal(
        cache.get({"key": 3}, value.shape, lambda: value.copy(), lambda: None), value
    )
    assert not list(tmp_path.glob("*.partial"))


def test_concurrent_requests_compute_once_and_cancelled_chunks_are_not_saved(tmp_path):
    cache = FeatureCache(tmp_path)
    count = []
    value = np.ones((4, 4, 4, 2), np.float32)

    def compute():
        count.append(1)
        time.sleep(0.03)
        return value.copy()

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(
                lambda _: cache.get(
                    {"key": "same"}, value.shape, compute, lambda: None
                ),
                range(4),
            )
        )
    assert len(count) == 1
    assert all(np.array_equal(result, value) for result in results)
    cancelled = []

    def compute_then_cancel():
        cancelled.append(True)
        return value.copy()

    def check():
        if cancelled:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        cache.get({"key": "cancelled"}, value.shape, compute_then_cancel, check)
    assert len(list(tmp_path.glob("*.f32.zst"))) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_model_pins_recipe_and_cached_prediction_matches_old_path(tmp_path, legacy):
    image = np.random.default_rng(0).random((1, 8, 8, 8)).astype(np.float32)
    recipe = multiscale_recipe(1, texture=legacy)
    h = extractor(recipe).halo
    block = np.pad(image, ((0, 0), (h, h), (h, h), (h, h)), mode="edge")
    feats = extractor(recipe).compute(block, 1)[h:-h, h:-h, h:-h]
    forest = RandomForestClassifier(n_estimators=4, max_depth=4, random_state=0).fit(
        feats.reshape(-1, feats.shape[-1]), (image[0] > 0.5).ravel()
    )
    meta = {"sigma_max": 1, "halo": h, "num_classes": 2}
    if not legacy:
        meta["feature_recipe"] = recipe.model_dump()
    (tmp_path / "model.json").write_text(json.dumps(meta))
    joblib.dump(forest, tmp_path / "forest.joblib")
    predictor = RandomForestPlugin().load(tmp_path)
    volume = FeatureVolume(
        image,
        "a",
        (0, 1),
        predictor.feature_recipe,
        FeatureCache(tmp_path / "cache"),
        1,
    )
    np.testing.assert_array_equal(
        predictor.predict_block(block),
        predictor.predict_features(volume.read((0, 0, 0, 8, 8, 8))),
    )
    if not legacy:
        assert predictor.feature_recipe.params["texture"] is False
    with pytest.raises(ValueError, match="saved model"):
        predictor.predict_features(np.zeros((2, 2, 2, 999), np.float32))
