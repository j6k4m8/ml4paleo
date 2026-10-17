"""
A random forest on 3D image features, trained on CPU from sparse labels.

Features are scikit-image's multiscale intensity, edge, and texture
features of each channel, at Gaussian scales from 1 to `sigma_max`. The
Gaussian is cut off at `int(4 * sigma_max + 0.5)` voxels (scipy's radius for
4 standard deviations), and the edge (Sobel) and texture (Hessian) features
take gradients of it, which reach two more, so a block needs that many plus
two voxels of context on every side. Training
samples are balanced: up to `samples_per_class` voxels of each class,
drawn uniformly from every crop (reservoir sampling), so sparse brush
strokes count as much as large painted regions. Training computes features
and fits trees on the context's `threads`, and keeps fewer samples per class
when they and the forest grown on them wouldn't fit in half the context's
memory budget (see `samples_that_fit`).
"""

import json
import math
import os
import pathlib
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field

from ml4paleo.labels import PLUGIN_IGNORE

from ..features import FeatureRecipe, extractor, multiscale_recipe
from ..metrics import ScoreSheet
from ..plugin import (
    Crop,
    CropCost,
    PluginCaps,
    TrainContext,
    TrainingData,
    TrainResult,
)

MODEL_FILE = "forest.joblib"
META_FILE = "model.json"
# Fewer samples than this per class make no useful forest.
MIN_SAMPLES_PER_CLASS = 100


class RandomForestParams(BaseModel):
    n_estimators: int = Field(100, ge=1, le=500)
    max_depth: int = Field(16, ge=1, le=64)
    samples_per_class: int = Field(50_000, ge=MIN_SAMPLES_PER_CLASS, le=1_000_000)
    sigma_max: float = Field(8.0, ge=1.0, le=8.0)
    feature_set: Literal["full", "intensity_edges"] = Field(
        "full",
        title="Image features",
        description="Texture costs more compute; compare model scores before choosing.",
    )
    # scikit-learn takes seeds from 0 to 2³² - 1.
    seed: int = Field(0, ge=0, le=2**32 - 1)


def halo_for(sigma_max: float) -> int:
    return int(4 * sigma_max + 0.5) + 2


def scale_count(sigma_max: float) -> int:
    """How many Gaussian scales `features` uses: 1, 2, 4, ... up to sigma_max."""
    return int(math.log2(sigma_max) + 1)


def feature_count(channels: int, sigma_max: float) -> int:
    """How many features `features` gives each voxel."""
    # Each scale gives intensity, edges, and the three eigenvalues of the
    # Hessian.
    return channels * 5 * scale_count(sigma_max)


def samples_that_fit(
    memory_bytes: int,
    classes: int,
    features: int,
    threads: int,
    params: RandomForestParams,
) -> int:
    """
    The most samples per class, up to `params.samples_per_class`, whose
    feature rows and the forest grown on them fit in `memory_bytes`.
    """
    # Each sample is a row of float32 features, kept in the reservoir and
    # copied into the array the forest fits on (2.5 times over, with slack),
    # plus scikit-learn's per-sample arrays: the labels as float64, twice,
    # and weights and indices for each tree being grown at once.
    row = 10 * features + 16 + 32 * threads
    # A tree node takes about 64 bytes and a float64 per class. Each tree
    # grows on a bootstrap draw holding about 63% of the samples, so it has
    # under 1.3 nodes per sample (two per distinct one), and at most
    # 2^(max_depth + 1) in all.
    node = 64 + 8 * classes
    full_tree = 2 ** (params.max_depth + 1)
    total = int(memory_bytes // (row + 1.3 * params.n_estimators * node))
    if 1.3 * total > full_tree:
        # The trees stop at max_depth, so beyond that only the rows grow.
        rest = memory_bytes - params.n_estimators * full_tree * node
        total = max(total, rest // row)
    return min(params.samples_per_class, total // classes)


def features(
    image: np.ndarray,
    sigma_max: float,
    threads: int | None = None,
    recipe: FeatureRecipe | None = None,
) -> np.ndarray:
    """
    (C, Z, Y, X) float32 -> (Z, Y, X, F) float32 features, computed on up to
    `threads` threads (None: every core).
    """
    return extractor(recipe or multiscale_recipe(sigma_max)).compute(image, threads)


class Reservoir:
    """
    A uniform sample of up to `size` rows from a stream, per class.
    """

    def __init__(self, size: int, rng: np.random.Generator):
        self.size = size
        self.rng = rng
        self.rows: dict[int, np.ndarray] = {}
        # Whether each kept row is a direct human/imported annotation. The
        # flags undergo the same reservoir replacements as their rows.
        self.human: dict[int, np.ndarray] = {}
        # Preserve one direct annotation per class as a backstop. A very
        # large stream of accepted model labels should never be able to
        # evict every human row and make the resulting model ungradable.
        self.first_human: dict[int, np.ndarray] = {}
        self.seen: dict[int, int] = {}

    def add(
        self, label: int, rows: np.ndarray, human: np.ndarray | None = None
    ) -> None:
        kept = self.rows.get(label)
        kept_human = self.human.get(label)
        flags = (
            np.ones(len(rows), dtype=bool)
            if human is None
            else np.asarray(human, dtype=bool)
        )
        if flags.shape != (len(rows),):
            raise ValueError("Human-label flags must match their feature rows")
        if label not in self.first_human and flags.any():
            self.first_human[label] = rows[np.flatnonzero(flags)[0]].copy()
        seen = self.seen.get(label, 0)
        for start in range(0, len(rows), self.size):
            batch = rows[start : start + self.size]
            batch_human = flags[start : start + self.size]
            if kept is None or len(kept) < self.size:
                room = self.size - (0 if kept is None else len(kept))
                take, batch = batch[:room], batch[room:]
                take_human, batch_human = batch_human[:room], batch_human[room:]
                kept = take.copy() if kept is None else np.concatenate([kept, take])
                kept_human = (
                    take_human.copy()
                    if kept_human is None
                    else np.concatenate([kept_human, take_human])
                )
                seen += len(take)
            if len(batch) == 0:
                continue
            # Each new row replaces a kept one with probability size / seen.
            positions = seen + 1 + np.arange(len(batch))
            slots = (self.rng.random(len(batch)) * positions).astype(np.int64)
            keep = slots < self.size
            kept[slots[keep]] = batch[keep]
            assert kept_human is not None
            kept_human[slots[keep]] = batch_human[keep]
            seen += len(batch)
        assert kept is not None and kept_human is not None
        self.rows[label] = kept
        self.human[label] = kept_human
        self.seen[label] = seen

    def ensure_human(self) -> None:
        """Keep at least one direct annotation for every class that has one."""
        for label, row in self.first_human.items():
            flags = self.human.get(label)
            kept = self.rows.get(label)
            if flags is not None and kept is not None and not flags.any():
                kept[0] = row
                flags[0] = True


class RandomForestPredictor:
    def __init__(self, forest: Any, meta: dict[str, Any], threads: int | None = None):
        self.forest = forest
        self.halo = int(meta["halo"])
        self.num_classes = int(meta["num_classes"])
        # Older models used the full recipe; preserve it on load.
        self.feature_recipe = (
            FeatureRecipe.model_validate(meta["feature_recipe"])
            if "feature_recipe" in meta
            else multiscale_recipe(float(meta["sigma_max"]))
        )
        implementation = extractor(self.feature_recipe)
        self.sigma_max = float(self.feature_recipe.params.get("sigma_max", 1))
        if implementation.halo != self.halo:
            raise ValueError("The model's saved feature halo is inconsistent")
        channels = meta.get("channels")
        if channels is None and (
            self.feature_recipe.name,
            self.feature_recipe.version,
        ) == ("multiscale", "1"):
            # Only this historical extractor had a known linear channel layout.
            groups = implementation.width(1)
            channels = (
                int(self.forest.n_features_in_) // groups
                if int(self.forest.n_features_in_) % groups == 0
                else 0
            )
        if (
            not isinstance(channels, int)
            or channels < 1
            or implementation.width(channels) != self.forest.n_features_in_
        ):
            raise ValueError(
                "The model's feature width does not match its saved recipe"
            )
        self.channels = channels
        # Threads for features (None: every core); the trees predict on one.
        self.threads = threads
        self.forest.n_jobs = 1

    @property
    def bytes_per_voxel(self) -> int:
        # float32 features (the forest knows how many it takes, also for
        # models from before `meta` said), about four times over as for
        # training crops (see `crop_cost`). With several threads,
        # scikit-image computes that many scales at once, each holding about
        # 160 bytes a voxel besides the features. Then the class
        # probabilities as float64, a few times over while scikit-learn adds
        # up its trees.
        return (
            extractor(self.feature_recipe).bytes_per_voxel(
                self.channels, self.threads or os.cpu_count() or 1
            )
            + 32 * self.num_classes
        )

    def predict_block(self, block: np.ndarray) -> np.ndarray:
        # Callers pad blocks at the image's edges by repeating its edge
        # voxels, as training crops are, so every block has the full halo.
        h = self.halo
        feats = features(block, self.sigma_max, self.threads, self.feature_recipe)
        interior = feats[
            h : feats.shape[0] - h, h : feats.shape[1] - h, h : feats.shape[2] - h
        ]
        return self.predict_features(interior)

    def predict_features(self, interior: np.ndarray) -> np.ndarray:
        if interior.ndim != 4 or interior.shape[-1] != self.forest.n_features_in_:
            raise ValueError("Feature array does not match the saved model")
        shape = interior.shape[:3]
        proba = self.forest.predict_proba(interior.reshape(-1, interior.shape[-1]))
        out = np.zeros((self.num_classes, *shape), dtype=np.float32)
        for column, label in enumerate(self.forest.classes_):
            out[int(label)] = proba[:, column].reshape(shape)
        return out


class RandomForestPlugin:
    name: ClassVar[str] = "rf"
    version: ClassVar[str] = "3"
    caps: ClassVar[PluginCaps] = PluginCaps(
        devices=("cpu",),
        display_name="Random forest",
        family="classical",
        learning="debounced",
        debounce_ms=1000,
        min_train_interval_ms=5000,
    )
    Params: ClassVar[type[BaseModel]] = RandomForestParams

    def feature_recipe(self, params: BaseModel) -> FeatureRecipe:
        assert isinstance(params, RandomForestParams)
        return multiscale_recipe(params.sigma_max, texture=params.feature_set == "full")

    def train(
        self,
        data: TrainingData,
        params: BaseModel,
        out: pathlib.Path,
        ctx: TrainContext,
    ) -> TrainResult:
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(params, RandomForestParams)
        implementation = extractor(self.feature_recipe(params))
        halo = implementation.halo
        threads = max(1, ctx.threads)
        # Samples and the forest get half the memory; crops get the rest.
        per_class = samples_that_fit(
            ctx.memory_budget_bytes // 2,
            data.num_classes,
            extractor(self.feature_recipe(params)).width(data.channels),
            threads,
            params,
        )
        if per_class < MIN_SAMPLES_PER_CLASS:
            raise ValueError(
                "This forest doesn't fit in the worker's memory: use fewer or "
                "shallower trees, or a smaller sigma_max"
            )
        rng = np.random.default_rng(params.seed)
        reservoir = Reservoir(per_class, rng)
        crops = 0
        ctx.progress(0.0, "Reading labels and computing features")
        for crop in data.crops("train", halo):
            ctx.check()
            crops += 1
            rows, labels, human = self._samples(
                crop, params.sigma_max, threads, self.feature_recipe(params)
            )
            for label in np.unique(labels):
                selected = labels == label
                reservoir.add(int(label), rows[selected], human[selected])
            # Reading crops takes most of the time before fitting.
            ctx.progress(min(0.5, 0.5 * (1 - 1 / (1 + crops / 20))))
        if not reservoir.rows:
            raise ValueError("There are no labeled voxels to train on")
        if len(reservoir.rows) < 2:
            # Index 0 is background; the others are the project's classes.
            (only,) = reservoir.rows
            has, fix = (
                ("background", "Label what you're looking for too")
                if only == 0
                else ("one class", "Paint some Background, or label another class")
            )
            raise ValueError(
                "Training needs labels of at least two classes (background "
                f"counts), and only {has} has any. {fix}"
            )
        reservoir.ensure_human()
        samples = {k: len(v) for k, v in reservoir.rows.items()}
        width = next(iter(reservoir.rows.values())).shape[1]
        # Fill one array to fit on, letting go of each class's samples once
        # copied, so they're never held twice over.
        x = np.empty((sum(samples.values()), width), dtype=np.float32)
        y = np.empty(len(x), dtype=np.uint8)
        human = np.empty(len(x), dtype=bool)
        start = 0
        for k in sorted(samples):
            rows = reservoir.rows.pop(k)
            flags = reservoir.human.pop(k)
            x[start : start + len(rows)] = rows
            y[start : start + len(rows)] = k
            human[start : start + len(rows)] = flags
            start += len(rows)
        ctx.progress(0.55, f"Fitting {params.n_estimators} trees on {len(y)} voxels")
        ctx.check()
        forest = RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            n_jobs=threads,
            random_state=params.seed,
            # Each training row is predicted only by trees whose bootstrap
            # sample left it out. This gives an internal held-out grade
            # without withholding scarce annotations from the final model.
            oob_score=True,
        )
        forest.fit(x, y)
        del x
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(forest, out / MODEL_FILE, compress=3)
        meta = {
            "plugin": self.name,
            "version": self.version,
            "params": params.model_dump(),
            "sigma_max": params.sigma_max,
            "halo": halo,
            "num_classes": data.num_classes,
            "class_values": data.class_values,
            "window": [float(v) for v in data.window],
            "features": int(width),
            "channels": data.channels,
            "feature_recipe": self.feature_recipe(params).model_dump(),
        }
        (out / META_FILE).write_text(json.dumps(meta, indent=2))
        ctx.progress(0.8, "Scoring on held-out human annotations")
        predictor = RandomForestPredictor(forest, meta, threads)
        sheet = ScoreSheet(data.num_classes)
        validation_crops = 0
        for crop in data.crops("val", halo):
            ctx.check()
            human_targets = np.where(
                crop.human if crop.human is not None else crop.targets != PLUGIN_IGNORE,
                crop.targets,
                PLUGIN_IGNORE,
            ).astype(np.uint8)
            if not (human_targets != PLUGIN_IGNORE).any():
                continue
            validation_crops += 1
            # Crops carry the full halo, so they predict like any block.
            probabilities = (
                predictor.predict_features(crop.features)
                if crop.features is not None
                else predictor.predict_block(crop.image)
            )
            sheet.add(probabilities.argmax(axis=0), human_targets)
        evaluation = "human_validation_rois"
        if sheet.voxels == 0:
            # No explicit human validation labels: use the forest's standard
            # out-of-bag predictions on human rows in its balanced reservoir.
            # Rows with no OOB tree (possible with very few trees) are omitted.
            decisions = np.asarray(forest.oob_decision_function_)
            covered = np.isfinite(decisions).all(axis=1) & (decisions.sum(axis=1) > 0)
            graded = human & covered
            if graded.any():
                predicted = np.asarray(forest.classes_)[
                    decisions[graded].argmax(axis=1)
                ]
                sheet.add(predicted.astype(np.uint8), y[graded])
            evaluation = "human_out_of_bag"
        metrics = sheet.summary(data.class_values) if sheet.voxels else {}
        metrics["evaluation"] = evaluation
        metrics["evaluation_voxels"] = sheet.voxels
        metrics["validation_crops"] = validation_crops
        metrics["training_crops"] = crops
        # Less than asked for when the memory budget is tight.
        metrics["samples_per_class_used"] = per_class
        ctx.progress(1.0)
        return TrainResult(
            metrics=metrics,
            samples=samples,
            files=[MODEL_FILE, META_FILE],
        )

    def crop_cost(self, params: BaseModel, channels: int, threads: int = 1) -> CropCost:
        assert isinstance(params, RandomForestParams)
        # float32 features, about four times over while scikit-image computes
        # each scale and stacks them, and they are cut down to the samples.
        # With several threads it computes that many scales at once, as for
        # prediction (see `RandomForestPredictor.bytes_per_voxel`).
        implementation = extractor(self.feature_recipe(params))
        return CropCost(
            halo=implementation.halo,
            bytes_per_voxel=implementation.bytes_per_voxel(channels, threads),
        )

    def _samples(
        self,
        crop: Crop,
        sigma_max: float,
        threads: int,
        recipe: FeatureRecipe | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = (
            crop.features
            if crop.features is not None
            else features(crop.image, sigma_max, threads, recipe)[crop.interior]
        )
        known = crop.targets != PLUGIN_IGNORE
        human = crop.human if crop.human is not None else known
        return feats[known], crop.targets[known], np.asarray(human, dtype=bool)[known]

    def load(
        self, directory: pathlib.Path, device: str = "cpu"
    ) -> RandomForestPredictor:
        import joblib

        meta = json.loads((directory / META_FILE).read_text())
        return RandomForestPredictor(joblib.load(directory / MODEL_FILE), meta)
