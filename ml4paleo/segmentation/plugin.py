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
    # UI and scheduling semantics, not inferred from a plugin's name/device.
    display_name: str = "Segmentation model"
    family: Literal["classical", "neural", "other"] = "other"
    # Unknown/expensive plugins require explicit training by default. Periodic
    # plugins train in the background; inference keeps the last ready checkpoint.
    learning: Literal["manual", "debounced", "periodic"] = "manual"
    debounce_ms: int = 1000
    min_train_interval_ms: int = 30_000


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
    # Which targets came directly from a person (drawn or imported). Metrics
    # use only these; model-accepted labels and implicit ROI background may
    # still train a model but can never grade it. Older/custom TrainingData
    # may leave this unset, in which case known targets are treated as human.
    human: np.ndarray | None = None
    # Where `targets` sits within `image`'s spatial axes.
    interior: tuple[slice, slice, slice] = (
        slice(None),
        slice(None),
        slice(None),
    )
    split: Split = "train"


class TrainingData(Protocol):
    """
    What plugins train on (see `ml4paleo.segmentation.dataset`).
    """

    # The project's class values (2..254), in plugin order 1..K.
    class_values: list[int]
    # The display window crops are normalized with (low, high); prediction
    # normalizes with the same one.
    window: tuple[float, float]

    @property
    def channels(self) -> int:
        """How many channels the image has."""
        ...

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

    @property
    def memory_budget_bytes(self) -> int:
        """
        How much memory training may use in all. Workers size crops to half
        of it (see `SegmentationPlugin.crop_cost`); what the plugin keeps
        from crop to crop (samples, the model) has to fit in the other half.
        """
        ...

    def progress(self, fraction: float, message: str | None = None) -> None: ...

    def check(self) -> None:
        """Raise if the job should stop (cancelled, or its lease is gone)."""
        ...


@dataclasses.dataclass
class TrainResult:
    # Per-class and overall scores on held-out human annotations.
    metrics: dict[str, Any]
    # Training voxels used, by plugin class index.
    samples: dict[int, int]
    # Files written into the model directory.
    files: list[str]


class Predictor(Protocol):
    # Voxels of context `predict_block` needs on every side.
    halo: int
    num_classes: int
    # CPU threads `predict_block` may use (None: every core).
    threads: int | None

    @property
    def bytes_per_voxel(self) -> int:
        """
        Bytes `predict_block` holds per voxel of the block it's given (halo
        included) on its `threads`, so callers can size blocks to their
        memory.
        """
        ...

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

    def crop_cost(self, params: BaseModel, channels: int, threads: int = 1) -> CropCost:
        """
        What training with `params` holds per crop of a `channels` image,
        on `threads` threads.
        """
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
