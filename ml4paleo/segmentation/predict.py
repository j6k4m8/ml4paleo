"""
Running a trained model over an image.

A prediction is a zarr group in the image's (z, y, x) grid with two uint8
arrays, in 64³ chunks inside 512³ shards:

- `class`: stored label values (1 background, 2..254 classes), 0 where
  nothing was predicted yet.
- `uncertainty`: 255 × (1 − the top class probability), so 0 is sure.

Each shard is predicted and written whole by one job, so jobs never write
the same shard.
"""

import itertools
import math
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import zarr

from ml4paleo.labels import from_plugin_space
from ml4paleo.storage import StorageGrant, zarr_store

from .dataset import Box, ImageArray, normalize
from .plugin import Predictor

CHUNK_ZYX = (64, 64, 64)
SHARD_ZYX = (512, 512, 512)
ARRAYS = ("class", "uncertainty")


def create_prediction(
    grant: StorageGrant,
    shape_zyx: Sequence[int],
    arrays: Sequence[str] = ARRAYS,
    kind: str = "prediction",
) -> zarr.Group:
    """
    Create the arrays of a prediction (or of another volume of label values
    in the same layout, such as a final segmentation).

    Other files already in the artifact stay (a final segmentation keeps its
    pinned labels there), and running this again, as a retried job does,
    replaces the arrays.
    """
    group = zarr.open_group(store=zarr_store(grant), mode="a", zarr_format=3)
    group.attrs["kind"] = kind
    shape = tuple(int(n) for n in shape_zyx)
    chunks = tuple(min(c, n) for c, n in zip(CHUNK_ZYX, shape, strict=True))
    shards = tuple(
        min(s, math.ceil(n / c) * c)
        for s, n, c in zip(SHARD_ZYX, shape, chunks, strict=True)
    )
    for name in arrays:
        group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=np.uint8,
            fill_value=0,
            dimension_names=("z", "y", "x"),
            overwrite=True,
        )
    return group


def open_prediction(grant: StorageGrant) -> zarr.Group:
    mode = "r+" if grant.access == "rw" else "r"
    return zarr.open_group(store=zarr_store(grant), mode=mode)


def shard_boxes(
    shape_zyx: Sequence[int], shard: Sequence[int] = SHARD_ZYX
) -> list[Box]:
    """The boxes of the prediction's shards, in order."""
    starts = [range(0, int(n), int(s)) for n, s in zip(shape_zyx, shard, strict=True)]
    return [
        (
            z,
            y,
            x,
            min(z + shard[0], shape_zyx[0]),
            min(y + shard[1], shape_zyx[1]),
            min(x + shard[2], shape_zyx[2]),
        )
        for z, y, x in itertools.product(*starts)
    ]


def blocks(box: Box, size: Sequence[int]) -> Iterator[Box]:
    starts = [range(box[a], box[a + 3], size[a]) for a in range(3)]
    for z, y, x in itertools.product(*starts):
        yield (
            z,
            y,
            x,
            min(z + size[0], box[3]),
            min(y + size[1], box[4]),
            min(x + size[2], box[5]),
        )


def predict_box(
    predictor: Predictor,
    image: ImageArray,
    box: Box,
    window: tuple[float, float],
    class_values: list[int],
    block: Sequence[int] = CHUNK_ZYX,
    progress: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict a box of the image (level 0, (c, z, y, x)): stored label values
    and uncertainty, each shaped like the box.

    The box is read once with the predictor's halo around it; where the
    image ends, its edge voxels are repeated.
    """
    h = predictor.halo
    shape = tuple(int(n) for n in image.shape[1:4])
    lo = [max(0, box[a] - h) for a in range(3)]
    hi = [min(shape[a], box[a + 3] + h) for a in range(3)]
    region = np.asarray(image[(slice(None), *(slice(lo[a], hi[a]) for a in range(3)))])
    padding = [(0, 0)] + [
        (h - (box[a] - lo[a]), h - (hi[a] - box[a + 3])) for a in range(3)
    ]
    region = np.pad(region, padding, mode="edge")
    size = tuple(box[a + 3] - box[a] for a in range(3))
    classes = np.zeros(size, dtype=np.uint8)
    uncertainty = np.zeros(size, dtype=np.uint8)
    pieces = list(blocks(box, block))
    for done, piece in enumerate(pieces):
        # Where the block sits in the padded region (which starts h before the box).
        start = [piece[a] - box[a] for a in range(3)]
        stop = [piece[a + 3] - box[a] + 2 * h for a in range(3)]
        chunk = normalize(
            region[(slice(None), *(slice(start[a], stop[a]) for a in range(3)))], window
        )
        probabilities = predictor.predict_block(chunk)
        out = tuple(slice(piece[a] - box[a], piece[a + 3] - box[a]) for a in range(3))
        classes[out] = from_plugin_space(
            probabilities.argmax(axis=0).astype(np.uint8), class_values
        )
        uncertainty[out] = np.round(255 * (1 - probabilities.max(axis=0))).astype(
            np.uint8
        )
        if progress:
            progress((done + 1) / len(pieces))
    return classes, uncertainty


def write_box(
    group: zarr.Group, box: Box, classes: np.ndarray, uncertainty: np.ndarray
) -> None:
    region = tuple(slice(box[a], box[a + 3]) for a in range(3))
    group["class"][region] = classes  # type: ignore[index]
    group["uncertainty"][region] = uncertainty  # type: ignore[index]
