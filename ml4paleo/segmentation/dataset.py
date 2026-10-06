"""
Training data from a project's image, labels, and ROIs.

A training set is pinned by its manifest (see the server's training-set
code): the image artifact, the label chunk hashes at one moment, and the
ROIs. Crops come from two places:

- ROIs that are open or complete, tiled into blocks of at most `tile`
  voxels a side. Inside a complete ROI of a crop's split, unlabeled voxels
  are background; elsewhere unlabeled voxels are ignored. Each ROI's
  train/val split carries over to its crops. ROIs may overlap: training
  crops ignore every voxel inside a validation ROI, and a voxel inside
  several ROIs of one split counts once, in the first of them.
- Labeled chunks outside those ROIs ("free" labels, for example from hand
  annotating the whole volume), for training only. Voxels inside open or
  complete ROIs are left to the ROI crops, so validation labels never leak
  into training.

Images are normalized with the image's display window, the same for every
crop and at prediction time.
"""

import dataclasses
import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, Protocol, cast

import numpy as np

from ml4paleo.labels import LABEL_CHUNK_ZYX, PLUGIN_IGNORE, to_plugin_space
from ml4paleo.labels.codec import blob_key, decode_chunk
from ml4paleo.storage import StorageGrant, get_bytes

from .plugin import Crop, Split

ChunkKey = tuple[int, int, int]
Box = tuple[int, int, int, int, int, int]


@dataclasses.dataclass(frozen=True)
class RoiSpec:
    bbox: Box
    status: Literal["open", "complete", "skipped"]
    split: Split


class LabelSource(Protocol):
    def chunk(self, key: ChunkKey) -> np.ndarray | None:
        """A stored label chunk's class array, or None if it's empty."""
        ...


class BlobLabels:
    """
    Label chunks read from their content-addressed blobs.
    """

    def __init__(self, grant: StorageGrant, shas: dict[ChunkKey, str]):
        self.grant = grant
        self.shas = shas

    def chunk(self, key: ChunkKey) -> np.ndarray | None:
        sha = self.shas.get(key)
        if sha is None:
            return None
        data = get_bytes(self.grant, blob_key(sha))
        if data is None:
            raise RuntimeError(f"Label blob {sha} is missing")
        return decode_chunk(data)


class ImageArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, selection: Any) -> Any: ...


def normalize(block: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    low, high = window
    scale = 1.0 / (high - low) if high > low else 1.0
    return ((block.astype(np.float32) - np.float32(low)) * np.float32(scale)).astype(
        np.float32
    )


def tiles(box: Box, tile: int) -> Iterator[Box]:
    """Split a box into blocks of at most `tile` voxels a side."""
    starts = [range(box[a], box[a + 3], tile) for a in range(3)]
    for z, y, x in itertools.product(*starts):
        yield (
            z,
            y,
            x,
            min(z + tile, box[3]),
            min(y + tile, box[4]),
            min(x + tile, box[5]),
        )


class TrainingSet:
    def __init__(
        self,
        image: ImageArray,
        labels: LabelSource,
        labeled_chunks: Iterable[ChunkKey],
        rois: Sequence[RoiSpec],
        class_values: list[int],
        window: tuple[float, float],
        tile: int = 96,
    ):
        """
        `image` is the level-0 (c, z, y, x) array; `labeled_chunks` are the
        keys of label chunks that have any labels.
        """
        self.image = image
        self.labels = labels
        self.labeled_chunks = sorted(set(labeled_chunks))
        self.rois = [roi for roi in rois if roi.status in ("open", "complete")]
        self.class_values = list(class_values)
        self.window = window
        self.tile = tile
        self.shape: tuple[int, int, int] = tuple(int(n) for n in image.shape[1:4])  # type: ignore[assignment]

    @property
    def num_classes(self) -> int:
        return len(self.class_values) + 1

    def read_labels(self, box: Box) -> np.ndarray:
        """Stored label values for a box, assembled from its chunks."""
        out = np.zeros(tuple(box[a + 3] - box[a] for a in range(3)), dtype=np.uint8)
        first = [box[a] // LABEL_CHUNK_ZYX[a] for a in range(3)]
        last = [(box[a + 3] - 1) // LABEL_CHUNK_ZYX[a] for a in range(3)]
        for key in itertools.product(
            *(range(f, t + 1) for f, t in zip(first, last, strict=True))
        ):
            chunk = self.labels.chunk(cast(ChunkKey, key))
            if chunk is None:
                continue
            origin = [k * c for k, c in zip(key, LABEL_CHUNK_ZYX, strict=True)]
            lo = [max(box[a], origin[a]) for a in range(3)]
            hi = [min(box[a + 3], origin[a] + LABEL_CHUNK_ZYX[a]) for a in range(3)]
            out[tuple(slice(lo[a] - box[a], hi[a] - box[a]) for a in range(3))] = chunk[
                tuple(slice(lo[a] - origin[a], hi[a] - origin[a]) for a in range(3))
            ]
        return out

    def read_image(
        self, box: Box, halo: int
    ) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
        """A box of the image with up to `halo` voxels of context, and where the box sits in it."""
        lo = [max(0, box[a] - halo) for a in range(3)]
        hi = [min(self.shape[a], box[a + 3] + halo) for a in range(3)]
        block = np.asarray(
            self.image[(slice(None), *(slice(lo[a], hi[a]) for a in range(3)))]
        )
        interior = tuple(slice(box[a] - lo[a], box[a + 3] - lo[a]) for a in range(3))
        return normalize(block, self.window), interior  # type: ignore[return-value]

    def _inside(self, box: Box, rois: Iterable[RoiSpec]) -> np.ndarray:
        """Which voxels of a box are inside any of `rois`."""
        inside = np.zeros(tuple(box[a + 3] - box[a] for a in range(3)), dtype=bool)
        for roi in rois:
            lo = [max(box[a], roi.bbox[a]) for a in range(3)]
            hi = [min(box[a + 3], roi.bbox[a + 3]) for a in range(3)]
            if all(a < b for a, b in zip(lo, hi, strict=True)):
                inside[
                    tuple(slice(lo[a] - box[a], hi[a] - box[a]) for a in range(3))
                ] = True
        return inside

    def crops(self, split: Split, halo: int) -> Iterator[Crop]:
        mine = [roi for roi in self.rois if roi.split == split]
        complete = [roi for roi in mine if roi.status == "complete"]
        # Validation ROIs are held out of training wherever they overlap it.
        held_out = (
            [roi for roi in self.rois if roi.split == "val"] if split == "train" else []
        )
        for index, roi in enumerate(mine):
            for box in tiles(roi.bbox, self.tile):
                targets = to_plugin_space(
                    self.read_labels(box),
                    self.class_values,
                    complete=self._inside(box, complete),
                )
                # Voxels of earlier ROIs of this split were counted there.
                targets[self._inside(box, mine[:index] + held_out)] = PLUGIN_IGNORE
                if (targets == PLUGIN_IGNORE).all():
                    continue
                image, interior = self.read_image(box, halo)
                yield Crop(image=image, targets=targets, interior=interior, split=split)
        if split != "train":
            return
        for key in self.labeled_chunks:
            origin = [k * c for k, c in zip(key, LABEL_CHUNK_ZYX, strict=True)]
            box: Box = (
                origin[0],
                origin[1],
                origin[2],
                min(origin[0] + LABEL_CHUNK_ZYX[0], self.shape[0]),
                min(origin[1] + LABEL_CHUNK_ZYX[1], self.shape[1]),
                min(origin[2] + LABEL_CHUNK_ZYX[2], self.shape[2]),
            )
            if any(box[a] >= box[a + 3] for a in range(3)):
                continue
            targets = to_plugin_space(self.read_labels(box), self.class_values)
            targets[self._inside(box, self.rois)] = PLUGIN_IGNORE
            if (targets == PLUGIN_IGNORE).all():
                continue
            image, interior = self.read_image(box, halo)
            yield Crop(image=image, targets=targets, interior=interior, split="train")
