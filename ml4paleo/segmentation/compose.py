"""
The final segmentation: a model's prediction, overruled by people's labels,
with specks removed.

For each voxel, in order:

1. what people labeled (any nonzero label, including accepted predictions);
2. background, if it is inside an ROI marked complete;
3. the prediction.

Then, for each class, connected pieces (6-connected) smaller than
`min_voxels` become background, unless people labeled part of the piece.
Pieces are counted across the whole volume, though each shard is processed
by its own job:

- `label_shard` merges one shard, labels its pieces, and records their sizes,
  classes, whether they hold labels, and the piece ids on the shard's faces;
- `find_specks` joins pieces that touch across shard faces (a union-find,
  as connected components of a sparse graph), adds up their sizes, and lists
  for each shard the pieces to remove;
- `apply_shard` writes the shard's final classes.

Labeling is deterministic, so `apply_shard` relabels instead of storing
piece ids. The final segmentation has the prediction's layout: a `class`
array of stored label values (1 background, 2..254 classes).
"""

import io
import itertools
from collections.abc import Sequence
from typing import cast

import numpy as np
import zarr

from ml4paleo.labels import BACKGROUND, FIRST_CLASS, LABEL_CHUNK_ZYX, UNLABELED

from .dataset import Box, LabelSource
from .predict import SHARD_ZYX, shard_boxes

# Pieces are counted with 6-connectivity: voxels that share a face.
_STRUCTURE = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ],
    dtype=bool,
)


def shard_grid(
    shape_zyx: Sequence[int], shard: Sequence[int] = SHARD_ZYX
) -> tuple[int, int, int]:
    """How many shards there are along each axis."""
    return tuple(-(-int(n) // int(s)) for n, s in zip(shape_zyx, shard, strict=True))  # type: ignore[return-value]


def merge(
    prediction: np.ndarray, labels: np.ndarray, complete: np.ndarray
) -> np.ndarray:
    """Labels where people labeled, background in complete ROIs, else the prediction."""
    merged = np.where(complete, BACKGROUND, prediction).astype(np.uint8)
    return np.where(labels != UNLABELED, labels, merged).astype(np.uint8)


def complete_mask(box: Box, complete_rois: Sequence[Box]) -> np.ndarray:
    inside = np.zeros(tuple(box[a + 3] - box[a] for a in range(3)), dtype=bool)
    for roi in complete_rois:
        lo = [max(box[a], roi[a]) for a in range(3)]
        hi = [min(box[a + 3], roi[a + 3]) for a in range(3)]
        if all(a < b for a, b in zip(lo, hi, strict=True)):
            inside[tuple(slice(lo[a] - box[a], hi[a] - box[a]) for a in range(3))] = (
                True
            )
    return inside


def pieces(merged: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Label the pieces of every class (not background) in a block: piece ids
    (0 for none) and each piece's class (index 0 unused).
    """
    from scipy import ndimage

    ids = np.zeros(merged.shape, dtype=np.int32)
    classes = [0]
    for value in np.unique(merged):
        if value < FIRST_CLASS:
            continue
        # Without an `output` argument, label returns the array and the count.
        labeled, count = cast(
            tuple[np.ndarray, int], ndimage.label(merged == value, structure=_STRUCTURE)
        )
        if count == 0:
            continue
        offset = len(classes) - 1
        ids[labeled > 0] = labeled[labeled > 0] + offset
        classes.extend([int(value)] * count)
    return ids, np.asarray(classes, dtype=np.uint8)


def read_label_box(labels: LabelSource, box: Box) -> np.ndarray:
    """Stored label values for a box, assembled from its 64³ chunks."""
    out = np.zeros(tuple(box[a + 3] - box[a] for a in range(3)), dtype=np.uint8)
    first = [box[a] // LABEL_CHUNK_ZYX[a] for a in range(3)]
    last = [(box[a + 3] - 1) // LABEL_CHUNK_ZYX[a] for a in range(3)]
    for key in itertools.product(
        *(range(f, t + 1) for f, t in zip(first, last, strict=True))
    ):
        chunk = labels.chunk(key)  # type: ignore[arg-type]
        if chunk is None:
            continue
        origin = [k * c for k, c in zip(key, LABEL_CHUNK_ZYX, strict=True)]
        lo = [max(box[a], origin[a]) for a in range(3)]
        hi = [min(box[a + 3], origin[a] + LABEL_CHUNK_ZYX[a]) for a in range(3)]
        out[tuple(slice(lo[a] - box[a], hi[a] - box[a]) for a in range(3))] = chunk[
            tuple(slice(lo[a] - origin[a], hi[a] - origin[a]) for a in range(3))
        ]
    return out


def shard_inputs(
    prediction: zarr.Array,
    labels: LabelSource,
    box: Box,
    complete_rois: Sequence[Box],
) -> tuple[np.ndarray, np.ndarray]:
    """The merged classes of a box, and which of its voxels people labeled."""
    region = tuple(slice(box[a], box[a + 3]) for a in range(3))
    predicted = np.asarray(prediction[region], dtype=np.uint8)
    labeled = read_label_box(labels, box)
    merged = merge(predicted, labeled, complete_mask(box, complete_rois))
    return merged, labeled != UNLABELED


def label_shard(merged: np.ndarray, labeled: np.ndarray) -> bytes:
    """
    The summary `find_specks` needs from one shard, as an `.npz`: each
    piece's size, class, and whether it holds labels, and the piece ids on
    the shard's six faces.
    """
    ids, classes = pieces(merged)
    count = len(classes)
    sizes = np.bincount(ids.ravel(), minlength=count)[:count]
    held = np.bincount(ids[labeled].ravel(), minlength=count)[:count] > 0
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        sizes=sizes.astype(np.int64),
        classes=classes,
        labeled=held,
        first_z=ids[0],
        last_z=ids[-1],
        first_y=ids[:, 0],
        last_y=ids[:, -1],
        first_x=ids[:, :, 0],
        last_x=ids[:, :, -1],
    )
    return buffer.getvalue()


def find_specks(
    summaries: Sequence[bytes], grid: Sequence[int], min_voxels: int
) -> list[np.ndarray]:
    """
    Given each shard's summary (in `shard_boxes` order), the ids of the
    pieces to remove in each shard: pieces of the same class joined across
    shard faces into whole pieces, and the whole pieces smaller than
    `min_voxels` that nobody labeled.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    loaded = [np.load(io.BytesIO(summary)) for summary in summaries]
    offsets = np.cumsum([0] + [len(s["sizes"]) for s in loaded])
    total = int(offsets[-1])
    index = {
        coords: i
        for i, coords in enumerate(itertools.product(*(range(n) for n in grid)))
    }
    edges_a: list[np.ndarray] = []
    edges_b: list[np.ndarray] = []
    for coords, i in index.items():
        for axis, name in enumerate("zyx"):
            neighbor = list(coords)
            neighbor[axis] += 1
            j = index.get(tuple(neighbor))  # type: ignore[arg-type]
            if j is None:
                continue
            a = loaded[i][f"last_{name}"].ravel()
            b = loaded[j][f"first_{name}"].ravel()
            touching = (a > 0) & (b > 0)
            a, b = a[touching], b[touching]
            same = loaded[i]["classes"][a] == loaded[j]["classes"][b]
            edges_a.append(a[same].astype(np.int64) + offsets[i])
            edges_b.append(b[same].astype(np.int64) + offsets[j])
    rows = np.concatenate(edges_a) if edges_a else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(edges_b) if edges_b else np.zeros(0, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(total, total)
    )
    _, component = connected_components(graph, directed=False)
    sizes = np.concatenate([s["sizes"] for s in loaded]).astype(np.float64)
    labeled = np.concatenate([s["labeled"] for s in loaded]).astype(np.float64)
    whole_size = np.bincount(component, weights=sizes)
    whole_labeled = np.bincount(component, weights=labeled) > 0
    classes = np.concatenate([s["classes"] for s in loaded])
    speck = (
        (whole_size[component] < min_voxels)
        & ~whole_labeled[component]
        & (classes >= FIRST_CLASS)
    )
    # Index 0 of each shard is "no piece", never a speck.
    for start in offsets[:-1]:
        speck[start] = False
    return [
        np.flatnonzero(speck[offsets[i] : offsets[i + 1]]).astype(np.int32)
        for i in range(len(loaded))
    ]


def apply_shard(merged: np.ndarray, remove: np.ndarray) -> np.ndarray:
    """The shard's final classes: its merged classes without the specks."""
    if len(remove) == 0:
        return merged
    ids, _ = pieces(merged)
    final = merged.copy()
    final[np.isin(ids, remove)] = BACKGROUND
    return final


__all__ = [
    "apply_shard",
    "complete_mask",
    "find_specks",
    "label_shard",
    "merge",
    "pieces",
    "read_label_box",
    "shard_boxes",
    "shard_grid",
    "shard_inputs",
]
