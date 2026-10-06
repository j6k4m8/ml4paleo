"""
Segmentation plugins: one interface for every kind of model (random
forests, nnU-Net, MONAI networks), so training, prediction, and the API
treat them alike.

A plugin trains on a `TrainingSet` (crops of image with sparse targets in
the plugin label space: 0 background, 1..K classes, 255 ignore) and writes
its model files into a directory; `load` turns those files back into a
`Predictor`, whose `predict_block` maps an image block (with `halo` voxels
of context on every side) to class probabilities for the block's interior.

Plugin modules keep their heavy imports (scikit-learn, torch) inside the
functions that need them, so the API server can list plugins and check
their parameters without installing them.
"""

import dataclasses
import pathlib
from collections.abc import Iterator
from typing import Any, ClassVar, Literal, Protocol

import numpy as np
from pydantic import BaseModel

Split = Literal["train", "val"]


@dataclasses.dataclass(frozen=True)
class PluginCaps:
    # Where the plugin can train and predict: "cpu", "cuda".
    devices: tuple[str, ...]
    min_vram_gb: float = 0.0
    # The prediction block (z, y, x) the plugin works best with.
    block: tuple[int, int, int] = (64, 64, 64)
    # Whether it can take prompts (clicks) for interactive segmentation.
    prompt: bool = False


@dataclasses.dataclass(frozen=True)
class CropCost:
    """
    What training holds in memory for one crop, so workers can size crops to
    their memory.
    """

    # Voxels of context the plugin asks crops for on every side.
    halo: int
    # Bytes held per voxel of a crop (halo included) while training on it.
    bytes_per_voxel: int


@dataclasses.dataclass
class Crop:
    """
    One piece of training data: an image block and targets for part of it.
    """

    # (C, Z, Y, X) float32, normalized to the image's display window, with
    # the halo the training set was asked for around the targets on every
    # side (where the image ends, its edge voxels repeated, as at prediction
    # time).
    image: np.ndarray
    # (Z, Y, X) uint8 targets in the plugin label space.
    targets: np.ndarray
    # Where `targets` sits within `image`'s spatial axes.
    interior: tuple[slice, slice, slice]
    split: Split


class TrainingData(Protocol):
    """
    What plugins train on (see `ml4paleo.segmentation.dataset`).
    """

    # The project's class values (2..254), in plugin order 1..K.
    class_values: list[int]

    @property
    def num_classes(self) -> int:
        """K + 1: background and the classes."""
        ...

    def crops(self, split: Split, halo: int) -> Iterator[Crop]: ...


class TrainContext(Protocol):
    @property
    def threads(self) -> int:
        """How many CPU threads the plugin may use at once."""
        ...

    def progress(self, fraction: float, message: str | None = None) -> None: ...

    def check(self) -> None:
        """Raise if the job should stop (cancelled, or its lease is gone)."""
        ...


@dataclasses.dataclass
class TrainResult:
    # Per-class and overall scores on validation crops (empty without any).
    metrics: dict[str, Any]
    # Training voxels used, by plugin class index.
    samples: dict[int, int]
    # Files written into the model directory.
    files: list[str]


class Predictor(Protocol):
    # Voxels of context `predict_block` needs on every side.
    halo: int
    num_classes: int

    def predict_block(self, block: np.ndarray) -> np.ndarray:
        """
        Map a (C, Z, Y, X) block, `halo` voxels larger than the region of
        interest on every side, to (K+1, Z-2h, Y-2h, X-2h) float32
        probabilities.
        """
        ...


class SegmentationPlugin(Protocol):
    name: ClassVar[str]
    version: ClassVar[str]
    caps: ClassVar[PluginCaps]
    Params: ClassVar[type[BaseModel]]

    def train(
        self,
        data: TrainingData,
        params: BaseModel,
        out: pathlib.Path,
        ctx: TrainContext,
    ) -> TrainResult: ...

    def crop_cost(self, params: BaseModel, channels: int) -> CropCost:
        """What training with `params` holds per crop of a `channels` image."""
        ...

    def load(self, directory: pathlib.Path, device: str = "cpu") -> Predictor: ...


def plugins() -> dict[str, type[SegmentationPlugin]]:
    """
    The installed plugins, by name.
    """
    from .plugins import rf

    return {plugin.name: plugin for plugin in (rf.RandomForestPlugin,)}


def get_plugin(name: str) -> type[SegmentationPlugin]:
    try:
        return plugins()[name]
    except KeyError:
        raise ValueError(f"No segmentation plugin named {name!r}") from None
