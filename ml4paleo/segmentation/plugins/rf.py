"""
A random forest on 3D image features, trained on CPU from sparse labels.

Features are scikit-image's multiscale intensity, edge, and texture
features of each channel, at Gaussian scales from 1 to `sigma_max`, so a
block needs `4 * sigma_max` voxels of context on every side. Training
samples are balanced: up to `samples_per_class` voxels of each class,
drawn uniformly from every crop (reservoir sampling), so sparse brush
strokes count as much as large painted regions.
"""

import json
import math
import pathlib
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import BaseModel, Field

from ml4paleo.labels import PLUGIN_IGNORE

from ..metrics import ScoreSheet
from ..plugin import Crop, PluginCaps, TrainContext, TrainingData, TrainResult

MODEL_FILE = "forest.joblib"
META_FILE = "model.json"


class RandomForestParams(BaseModel):
    n_estimators: int = Field(100, ge=1, le=500)
    max_depth: int = Field(16, ge=1, le=64)
    samples_per_class: int = Field(50_000, ge=100, le=1_000_000)
    sigma_max: float = Field(8.0, ge=1.0, le=16.0)
    seed: int = 0


def halo_for(sigma_max: float) -> int:
    return math.ceil(4 * sigma_max)


def features(image: np.ndarray, sigma_max: float) -> np.ndarray:
    """
    (C, Z, Y, X) float32 -> (Z, Y, X, F) float32 features.
    """
    from skimage.feature import multiscale_basic_features

    per_channel = [
        multiscale_basic_features(
            channel,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=1.0,
            # Typed as int, but any positive number works.
            sigma_max=cast(Any, sigma_max),
        ).astype(np.float32)
        for channel in image
    ]
    return np.concatenate(per_channel, axis=-1)


class Reservoir:
    """
    A uniform sample of up to `size` rows from a stream, per class.
    """

    def __init__(self, size: int, rng: np.random.Generator):
        self.size = size
        self.rng = rng
        self.rows: dict[int, np.ndarray] = {}
        self.seen: dict[int, int] = {}

    def add(self, label: int, rows: np.ndarray) -> None:
        kept = self.rows.get(label)
        seen = self.seen.get(label, 0)
        for start in range(0, len(rows), self.size):
            batch = rows[start : start + self.size]
            if kept is None or len(kept) < self.size:
                room = self.size - (0 if kept is None else len(kept))
                take, batch = batch[:room], batch[room:]
                kept = take.copy() if kept is None else np.concatenate([kept, take])
                seen += len(take)
            if len(batch) == 0:
                continue
            # Each new row replaces a kept one with probability size / seen.
            positions = seen + 1 + np.arange(len(batch))
            slots = (self.rng.random(len(batch)) * positions).astype(np.int64)
            keep = slots < self.size
            kept[slots[keep]] = batch[keep]
            seen += len(batch)
        assert kept is not None
        self.rows[label] = kept
        self.seen[label] = seen


class RandomForestPredictor:
    def __init__(self, forest: Any, meta: dict[str, Any]):
        self.forest = forest
        self.sigma_max = float(meta["sigma_max"])
        self.halo = int(meta["halo"])
        self.num_classes = int(meta["num_classes"])
        self.forest.n_jobs = 1

    def predict_block(self, block: np.ndarray) -> np.ndarray:
        # Callers pad blocks at the image's edges (for example by
        # reflection), so every block has the full halo.
        h = self.halo
        feats = features(block, self.sigma_max)
        interior = feats[
            h : feats.shape[0] - h, h : feats.shape[1] - h, h : feats.shape[2] - h
        ]
        shape = interior.shape[:3]
        proba = self.forest.predict_proba(interior.reshape(-1, interior.shape[-1]))
        out = np.zeros((self.num_classes, *shape), dtype=np.float32)
        for column, label in enumerate(self.forest.classes_):
            out[int(label)] = proba[:, column].reshape(shape)
        return out


class RandomForestPlugin:
    name: ClassVar[str] = "rf"
    version: ClassVar[str] = "1"
    caps: ClassVar[PluginCaps] = PluginCaps(devices=("cpu",))
    Params: ClassVar[type[BaseModel]] = RandomForestParams

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
        halo = halo_for(params.sigma_max)
        rng = np.random.default_rng(params.seed)
        reservoir = Reservoir(params.samples_per_class, rng)
        crops = 0
        ctx.progress(0.0, "Reading labels and computing features")
        for crop in data.crops("train", halo):
            ctx.check()
            crops += 1
            rows, labels = self._samples(crop, params.sigma_max)
            for label in np.unique(labels):
                reservoir.add(int(label), rows[labels == label])
            # Reading crops takes most of the time before fitting.
            ctx.progress(min(0.5, 0.5 * (1 - 1 / (1 + crops / 20))))
        if not reservoir.rows:
            raise ValueError("There are no labeled voxels to train on")
        if len(reservoir.rows) < 2:
            raise ValueError(
                "Training needs labels of at least two classes (background counts)"
            )
        x = np.concatenate([reservoir.rows[k] for k in sorted(reservoir.rows)])
        y = np.concatenate(
            [
                np.full(len(reservoir.rows[k]), k, dtype=np.uint8)
                for k in sorted(reservoir.rows)
            ]
        )
        ctx.progress(0.55, f"Fitting {params.n_estimators} trees on {len(y)} voxels")
        ctx.check()
        forest = RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            n_jobs=-1,
            random_state=params.seed,
        )
        forest.fit(x, y)
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
            "features": int(x.shape[1]),
        }
        (out / META_FILE).write_text(json.dumps(meta, indent=2))
        ctx.progress(0.8, "Scoring on validation ROIs")
        predictor = RandomForestPredictor(forest, meta)
        sheet = ScoreSheet(data.num_classes)
        validation_crops = 0
        for crop in data.crops("val", halo):
            ctx.check()
            validation_crops += 1
            sheet.add(self._predict_crop(predictor, crop).argmax(axis=0), crop.targets)
        metrics = sheet.summary(data.class_values) if validation_crops else {}
        metrics["validation_crops"] = validation_crops
        metrics["training_crops"] = crops
        ctx.progress(1.0)
        return TrainResult(
            metrics=metrics,
            samples={k: len(v) for k, v in reservoir.rows.items()},
            files=[MODEL_FILE, META_FILE],
        )

    def _samples(self, crop: Crop, sigma_max: float) -> tuple[np.ndarray, np.ndarray]:
        feats = features(crop.image, sigma_max)[crop.interior]
        known = crop.targets != PLUGIN_IGNORE
        return feats[known], crop.targets[known]

    def _predict_crop(self, predictor: RandomForestPredictor, crop: Crop) -> np.ndarray:
        # Crops may have less context than the halo at the image's edges, so
        # compute features on what there is and keep the interior.
        feats = features(crop.image, predictor.sigma_max)[crop.interior]
        shape = feats.shape[:3]
        proba = predictor.forest.predict_proba(feats.reshape(-1, feats.shape[-1]))
        out = np.zeros((predictor.num_classes, *shape), dtype=np.float32)
        for column, label in enumerate(predictor.forest.classes_):
            out[int(label)] = proba[:, column].reshape(shape)
        return out

    def load(
        self, directory: pathlib.Path, device: str = "cpu"
    ) -> RandomForestPredictor:
        import joblib

        meta = json.loads((directory / META_FILE).read_text())
        return RandomForestPredictor(joblib.load(directory / MODEL_FILE), meta)
