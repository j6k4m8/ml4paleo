"""
Running a trained model over an image.

A prediction is a zarr group in the image's (z, y, x) grid with two uint8
arrays, in 64³ chunks inside 512³ shards:

- `class`: stored label values (1 background, 2..254 classes), 0 where
  nothing was predicted yet.
- `uncertainty`: 255 × (1 − the top class probability), so 0 is sure.

Each shard is predicted by one job, so jobs never write the same shard. A
job goes through its shard in blocks sized to its memory budget, reading
each block with the predictor's halo straight from the image (see
`predict_box`).
"""

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import zarr

from ml4paleo.labels import from_plugin_space
from ml4paleo.storage import StorageGrant, zarr_store

from .dataset import Box, ImageArray, normalize, read_box, tile_for, tiles
from .plugin import CropCost, Predictor

CHUNK_ZYX = (64, 64, 64)
SHARD_ZYX = (512, 512, 512)
ARRAYS = ("class", "uncertainty")
# Blocks are at least this many voxels a side, however little memory there is.
MIN_BLOCK = 16


def create_prediction(grant: StorageGrant, shape_zyx: Sequence[int]) -> zarr.Group:
    group = zarr.create_group(
        store=zarr_store(grant),
        zarr_format=3,
        attributes={"kind": "prediction"},
        overwrite=True,
    )
    shape = tuple(int(n) for n in shape_zyx)
    chunks = tuple(min(c, n) for c, n in zip(CHUNK_ZYX, shape, strict=True))
    shards = tuple(
        min(s, math.ceil(n / c) * c)
        for s, n, c in zip(SHARD_ZYX, shape, chunks, strict=True)
    )
    for name in ARRAYS:
        group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=np.uint8,
            fill_value=0,
            dimension_names=("z", "y", "x"),
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


def block_for(memory_budget_bytes: int, channels: int, predictor: Predictor) -> int:
    """
    The side of the largest blocks that fit in a memory budget with their
    halo, counting the image read for them and what the predictor holds,
    but at least `MIN_BLOCK` and at most a shard.
    """
    # The image is held as read (at most 8 bytes a voxel and channel) and
    # normalized to float32 (twice over while normalizing).
    cost = CropCost(
        halo=predictor.halo,
        bytes_per_voxel=16 * channels + predictor.bytes_per_voxel,
    )
    return tile_for(memory_budget_bytes, cost, MIN_BLOCK, max(SHARD_ZYX))


def _predict(
    predictor: Predictor,
    image: ImageArray,
    box: Box,
    window: tuple[float, float],
    class_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stored label values and uncertainty for a box of the image.
    """
    probabilities = predictor.predict_block(
        normalize(read_box(image, box, predictor.halo), window)
    )
    classes = from_plugin_space(
        probabilities.argmax(axis=0).astype(np.uint8), class_values
    )
    uncertainty = np.round(255 * (1 - probabilities.max(axis=0))).astype(np.uint8)
    return classes, uncertainty


def predict_box(
    predictor: Predictor,
    image: ImageArray,
    box: Box,
    window: tuple[float, float],
    class_values: list[int],
    out: zarr.Group,
    memory_budget_bytes: int,
    progress: Callable[[float], None] | None = None,
) -> None:
    """
    Predict a box of the image (level 0, (c, z, y, x)) into the prediction
    `out`, block by block. Each block is read with the predictor's halo;
    where the image ends, its edge voxels are repeated.

    Blocks and the box's outputs share half the memory budget, and the
    model keeps the other half, as in training. When the box's outputs take
    at most half of that share, they are kept in memory and written once;
    otherwise each block's are written as soon as it's predicted, which
    reads and rewrites the shard every time.
    """
    size = tuple(box[a + 3] - box[a] for a in range(3))
    share = memory_budget_bytes // 2
    output_bytes = len(ARRAYS) * math.prod(size)
    outputs = (
        (np.zeros(size, dtype=np.uint8), np.zeros(size, dtype=np.uint8))
        if output_bytes <= share // 2
        else None
    )
    side = block_for(
        share - output_bytes if outputs is not None else share,
        int(image.shape[0]),
        predictor,
    )
    pieces = list(tiles(box, side))
    for done, piece in enumerate(pieces):
        classes, uncertainty = _predict(predictor, image, piece, window, class_values)
        if outputs is None:
            write_box(out, piece, classes, uncertainty)
        else:
            at = tuple(
                slice(piece[a] - box[a], piece[a + 3] - box[a]) for a in range(3)
            )
            outputs[0][at] = classes
            outputs[1][at] = uncertainty
        if progress:
            progress((done + 1) / len(pieces))
    if outputs is not None:
        write_box(out, box, *outputs)


def write_box(
    group: zarr.Group, box: Box, classes: np.ndarray, uncertainty: np.ndarray
) -> None:
    region = tuple(slice(box[a], box[a + 3]) for a in range(3))
    group["class"][region] = classes  # type: ignore[index]
    group["uncertainty"][region] = uncertainty  # type: ignore[index]
