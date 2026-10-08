"""
Coarse levels of a label volume, for display.

A zoomed-out view can't draw every voxel, so viewers draw labels from a
pyramid that shares the image's: level k has the image's level-k shape, and a
voxel there covers the same block of the volume as the image's voxel does.
The levels are only for looking at. Edits never write them, and training,
exports, accepts, and segmentation never read them.

Each level is made from the one below it, one block at a time. A block is two
voxels along each axis that the image halves going down a level, and one along
the others. The block's voxel is:

1. the most common class (value 2 or more) in the block, the lowest value
   among classes that tie, if the block has any class;
2. otherwise background (1), if the block has any background;
3. otherwise unlabeled (0).

A class beats background however few of its voxels there are, and background
beats unlabeled, so thin labeled structures stay visible: a one-voxel line or
a single labeled voxel is still there at the top of the pyramid, where taking
every second voxel, or the most common value of all, would lose it. A coarse
voxel is always a value some voxel in its block has, so classes never mix.
Voxels beyond the edge of the volume count as unlabeled.

Because each level comes from the one below, a coarse voxel follows the
winners of its sub-blocks, not a vote among every full-resolution voxel under
it. The result depends only on the labels.
"""

from collections.abc import Sequence
from itertools import product

import numpy as np

from . import BACKGROUND, FIRST_CLASS

_NO_CLASS = np.uint8(255)
# A vote counts voxels in a byte.
MAX_BLOCK_VOXELS = 255


def downsample_labels(
    labels: np.ndarray, step: Sequence[int] = (2, 2, 2)
) -> np.ndarray:
    """
    Shrink a (z, y, x) block of label values by `step` along each axis, using
    the rule in this module's description. Axes that aren't a multiple of
    their step are padded with unlabeled voxels at the end, so the result has
    `ceil(size / step)` voxels along each axis.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3 or labels.dtype != np.uint8:
        raise ValueError(f"Labels are uint8 (z, y, x) blocks, got {labels.dtype}")
    if len(step) != 3 or any(int(s) != s or s < 1 for s in step):
        raise ValueError(f"Steps are three positive integers, got {step}")
    sz, sy, sx = (int(s) for s in step)
    if sz * sy * sx > MAX_BLOCK_VOXELS:
        raise ValueError(f"Blocks of {sz * sy * sx} voxels are too big to vote in")
    size = tuple(-(-n // s) for n, s in zip(labels.shape, (sz, sy, sx), strict=True))
    padding = [
        (0, out * s - n)
        for out, s, n in zip(size, (sz, sy, sx), labels.shape, strict=True)
    ]
    if any(after for _, after in padding):
        labels = np.pad(labels, padding)
    if (sz, sy, sx) == (1, 1, 1):
        return labels.copy()
    # The voxels of every block: the i-th voxel of each block, for each i.
    views = np.stack(
        [
            labels[i::sz, j::sy, k::sx]
            for i, j, k in product(range(sz), range(sy), range(sx))
        ]
    )
    return _vote(np.ascontiguousarray(views))


def _vote(views: np.ndarray) -> np.ndarray:
    """
    The voxel of each block, from `views` (the block's voxels along axis 0).
    """
    top = views.max(axis=0)
    has_class = top >= FIRST_CLASS
    # Most blocks that have a class have only one kind of class, which wins
    # without counting; only mixed blocks need a vote.
    lowest = np.where(views >= FIRST_CLASS, views, _NO_CLASS).min(axis=0)
    mixed = has_class & (lowest != top)
    result = np.where(has_class, top, (views == BACKGROUND).any(axis=0))
    result = result.astype(np.uint8)
    if mixed.any():
        where = np.flatnonzero(mixed)
        result.flat[where] = _most_common_class(views.reshape(len(views), -1)[:, where])
    return result


def _most_common_class(blocks: np.ndarray) -> np.ndarray:
    """
    The most common class in each column of `blocks` (the lowest of any that
    tie), for columns that have a class.
    """
    size, columns = blocks.shape
    votes = np.ones((size, columns), dtype=np.uint8)
    for i in range(size):
        for j in range(i + 1, size):
            same = blocks[i] == blocks[j]
            votes[i] += same
            votes[j] += same
    # Votes first; among equal votes, the lower value scores higher.
    score = votes.astype(np.uint16)
    score <<= 8
    score |= 255 - blocks
    score *= blocks >= FIRST_CLASS
    return (255 - (score.max(axis=0) & 255)).astype(np.uint8)
