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
  annotating the whole volume), tiled the same way, for training only.
  Voxels inside open or complete ROIs are left to the ROI crops, so
  validation labels never leak into training.

Workers choose `tile` so a crop fits in half the job's memory budget at
the plugin's crop cost (see `tile_for`); the plugin keeps the other half.

Images are normalized with the image's display window, the same for every
crop and at prediction time. Every crop has the full halo of context the
plugin asks for; where the image ends, its edge voxels are repeated, as
they are at prediction time, so features near the edges match.
"""

import dataclasses
import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, Protocol, cast

import numpy as np

from ml4paleo.labels import LABEL_CHUNK_ZYX, PLUGIN_IGNORE, Source, to_plugin_space
from ml4paleo.labels.codec import blob_key, decode_chunk
from ml4paleo.storage import StorageGrant, get_bytes

from .plugin import Crop, CropCost, Split

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

    def source(self, key: ChunkKey) -> np.ndarray | None:
        """The matching provenance chunk, or None if it is all zero."""
        ...


class MissingLabels(Exception):
    """
    A label blob that a training set pins is gone, so it can never be read.
    """


class BlobLabels:
    """
    Label chunks read from their content-addressed blobs.
    """

    def __init__(
        self,
        grant: StorageGrant,
        shas: dict[ChunkKey, str],
        source_shas: dict[ChunkKey, str] | None = None,
    ):
        self.grant = grant
        self.shas = shas
        self.source_shas = source_shas

    def chunk(self, key: ChunkKey) -> np.ndarray | None:
        sha = self.shas.get(key)
        if sha is None:
            return None
        data = get_bytes(self.grant, blob_key(sha))
        if data is None:
            raise MissingLabels(f"Label blob {sha} is missing")
        return decode_chunk(data)

    def source(self, key: ChunkKey) -> np.ndarray | None:
        """
        Read provenance pinned beside the class chunk. Version-1 training
        manifests did not pin it; treating their nonzero labels as human
        preserves their historical behavior without weakening new grades.
        """
        if self.source_shas is None:
            chunk = self.chunk(key)
            return (
                None
                if chunk is None
                else np.where(chunk, Source.HUMAN, Source.NONE).astype(np.uint8)
            )
        sha = self.source_shas.get(key)
        if sha is None:
            return None
        data = get_bytes(self.grant, blob_key(sha))
        if data is None:
            raise MissingLabels(f"Label blob {sha} is missing")
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


def read_box(image: ImageArray, box: Box, halo: int) -> np.ndarray:
    """
    A box of a (c, z, y, x) image with `halo` voxels of context on every
    side. Only the box and its halo are read; where the image ends, its edge
    voxels are repeated.
    """
    shape = [int(n) for n in image.shape[1:4]]
    lo = [max(0, box[a] - halo) for a in range(3)]
    hi = [min(shape[a], box[a + 3] + halo) for a in range(3)]
    block = np.asarray(image[(slice(None), *(slice(lo[a], hi[a]) for a in range(3)))])
    padding = [(0, 0)] + [
        (halo - (box[a] - lo[a]), halo - (hi[a] - box[a + 3])) for a in range(3)
    ]
    return np.pad(block, padding, mode="edge")


def clip_box(box: Sequence[int], shape: Sequence[int]) -> Box | None:
    """A box cut to an image's (z, y, x) shape, or None if nothing is left."""
    lo = [min(max(int(box[a]), 0), int(shape[a])) for a in range(3)]
    hi = [min(max(int(box[a + 3]), 0), int(shape[a])) for a in range(3)]
    if any(lo[a] >= hi[a] for a in range(3)):
        return None
    return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])


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


def tile_for(
    memory_budget_bytes: int, cost: CropCost, minimum: int = 32, maximum: int = 256
) -> int:
    """
    The largest tile whose crops, with their halo, fit in a memory budget at
    `cost`, but at least `minimum` and at most `maximum` voxels a side.
    """
    side = round((memory_budget_bytes / cost.bytes_per_voxel) ** (1 / 3))
    while side > 0 and side**3 * cost.bytes_per_voxel > memory_budget_bytes:
        side -= 1
    return max(minimum, min(maximum, side - 2 * cost.halo))


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
        keys of label chunks that have any labels. ROIs are cut to the image.
        """
        self.image = image
        self.labels = labels
        self.labeled_chunks = sorted(set(labeled_chunks))
        self.class_values = list(class_values)
        self.window = window
        self.tile = tile
        self.shape: tuple[int, int, int] = tuple(int(n) for n in image.shape[1:4])  # type: ignore[assignment]
        self.rois = [
            dataclasses.replace(roi, bbox=bbox)
            for roi in rois
            if roi.status in ("open", "complete")
            and (bbox := clip_box(roi.bbox, self.shape)) is not None
        ]

    @property
    def num_classes(self) -> int:
        return len(self.class_values) + 1

    @property
    def channels(self) -> int:
        return int(self.image.shape[0])

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

    def read_sources(self, box: Box) -> np.ndarray:
        """Stored provenance for a box, assembled like its class labels."""
        out = np.zeros(tuple(box[a + 3] - box[a] for a in range(3)), dtype=np.uint8)
        first = [box[a] // LABEL_CHUNK_ZYX[a] for a in range(3)]
        last = [(box[a + 3] - 1) // LABEL_CHUNK_ZYX[a] for a in range(3)]
        source = getattr(self.labels, "source", None)
        for key in itertools.product(
            *(range(f, t + 1) for f, t in zip(first, last, strict=True))
        ):
            classes = self.labels.chunk(cast(ChunkKey, key))
            chunk = source(cast(ChunkKey, key)) if source is not None else None
            # Old in-memory/custom label sources may expose only class
            # chunks. Sources that do expose provenance decide what a
            # missing source chunk means themselves.
            if source is None and classes is not None:
                chunk = np.where(classes, Source.HUMAN, Source.NONE).astype(np.uint8)
            if chunk is None:
                continue
            origin = [k * c for k, c in zip(key, LABEL_CHUNK_ZYX, strict=True)]
            lo = [max(box[a], origin[a]) for a in range(3)]
            hi = [min(box[a + 3], origin[a] + LABEL_CHUNK_ZYX[a]) for a in range(3)]
            out[tuple(slice(lo[a] - box[a], hi[a] - box[a]) for a in range(3))] = chunk[
                tuple(slice(lo[a] - origin[a], hi[a] - origin[a]) for a in range(3))
            ]
        return out

    def _targets(
        self, box: Box, complete: np.ndarray | bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plugin targets and which of them are direct human annotations."""
        targets = to_plugin_space(
            self.read_labels(box), self.class_values, complete=complete
        )
        sources = self.read_sources(box)
        human = ((sources == Source.HUMAN) | (sources == Source.IMPORTED)) & (
            targets != PLUGIN_IGNORE
        )
        return targets, human

    def read_image(
        self, box: Box, halo: int
    ) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
        """
        A box of the image with `halo` voxels of context on every side, and
        where the box sits in it. Where the image ends, its edge voxels are
        repeated.
        """
        block = read_box(self.image, box, halo)
        interior = tuple(slice(halo, halo + box[a + 3] - box[a]) for a in range(3))
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
                targets, human = self._targets(
                    box, complete=self._inside(box, complete)
                )
                # Voxels of earlier ROIs of this split were counted there.
                excluded = self._inside(box, mine[:index] + held_out)
                targets[excluded] = PLUGIN_IGNORE
                human[excluded] = False
                if (targets == PLUGIN_IGNORE).all():
                    continue
                image, interior = self.read_image(box, halo)
                yield Crop(
                    image=image,
                    targets=targets,
                    human=human,
                    interior=interior,
                    split=split,
                )
        if split != "train":
            return
        for key in self.labeled_chunks:
            origin = [k * c for k, c in zip(key, LABEL_CHUNK_ZYX, strict=True)]
            chunk: Box = (
                origin[0],
                origin[1],
                origin[2],
                min(origin[0] + LABEL_CHUNK_ZYX[0], self.shape[0]),
                min(origin[1] + LABEL_CHUNK_ZYX[1], self.shape[1]),
                min(origin[2] + LABEL_CHUNK_ZYX[2], self.shape[2]),
            )
            if any(chunk[a] >= chunk[a + 3] for a in range(3)):
                continue
            for box in tiles(chunk, self.tile):
                targets, human = self._targets(box)
                excluded = self._inside(box, self.rois)
                targets[excluded] = PLUGIN_IGNORE
                human[excluded] = False
                if (targets == PLUGIN_IGNORE).all():
                    continue
                image, interior = self.read_image(box, halo)
                yield Crop(
                    image=image,
                    targets=targets,
                    human=human,
                    interior=interior,
                    split="train",
                )
