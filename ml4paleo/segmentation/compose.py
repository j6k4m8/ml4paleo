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

A shard's jobs hold its merged classes and one int32 array of piece ids,
reused for each class in turn; sizes and lookups go a slab of z planes at a
time, so their int64 copies stay small (`slab_depth`).
"""

import io
import itertools
from collections.abc import Iterator, Sequence
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
# A shard's faces, and where each is in a (z, y, x) block.
_FACES = {
    "first_z": (0,),
    "last_z": (-1,),
    "first_y": (slice(None), 0),
    "last_y": (slice(None), -1),
    "first_x": (slice(None), slice(None), 0),
    "last_x": (slice(None), slice(None), -1),
}
# z planes per slab, unless the job's memory budget says otherwise.
SLAB = 32


def shard_grid(
    shape_zyx: Sequence[int], shard: Sequence[int] = SHARD_ZYX
) -> tuple[int, int, int]:
    """How many shards there are along each axis."""
    return tuple(-(-int(n) // int(s)) for n, s in zip(shape_zyx, shard, strict=True))  # type: ignore[return-value]


def slab_depth(budget_bytes: int, shape_zyx: Sequence[int] = SHARD_ZYX) -> int:
    """
    How many z planes of a block to count at once, so that their int64
    copies take at most a sixteenth of a job's memory budget.
    """
    plane = int(shape_zyx[1]) * int(shape_zyx[2])
    return max(1, budget_bytes // 16 // (8 * plane))


def _slabs(depth: int, slab: int) -> Iterator[slice]:
    for z in range(0, depth, slab):
        yield slice(z, z + slab)


def npz(**arrays: np.ndarray) -> bytes:
    """Arrays as the bytes of a compressed `.npz` file."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
    return buffer.getvalue()


def merge(
    prediction: np.ndarray, labels: np.ndarray, box: Box, complete_rois: Sequence[Box]
) -> np.ndarray:
    """
    Labels where people labeled, background in complete ROIs, else the
    prediction. Writes into `prediction` (a box of it) and returns it.
    """
    for roi in complete_rois:
        lo = [max(box[a], roi[a]) for a in range(3)]
        hi = [min(box[a + 3], roi[a + 3]) for a in range(3)]
        if all(a < b for a, b in zip(lo, hi, strict=True)):
            inside = tuple(slice(lo[a] - box[a], hi[a] - box[a]) for a in range(3))
            prediction[inside] = BACKGROUND
    np.copyto(prediction, labels, where=labels != UNLABELED)
    return prediction


def class_values(merged: np.ndarray, slab: int = SLAB) -> list[int]:
    """The classes (not background) in a block, in order."""
    present = np.zeros(256, dtype=bool)
    for z in _slabs(len(merged), slab):
        present[np.unique(merged[z])] = True
    return [int(value) for value in np.flatnonzero(present) if value >= FIRST_CLASS]


def label_class(
    merged: np.ndarray, value: int, mask: np.ndarray, ids: np.ndarray
) -> int:
    """
    Label the pieces of one class into `ids` (0 elsewhere), using `mask` as
    scratch; returns how many there are. Pieces are numbered per class,
    with the classes in order, so piece `n` of a class is `offset + n` in
    the block, where `offset` counts the pieces of the classes before it.
    """
    from scipy import ndimage

    np.equal(merged, value, out=mask)
    # With an `output` array, label fills it and returns only the count.
    return cast(int, ndimage.label(mask, structure=_STRUCTURE, output=ids))


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
    values = read_label_box(labels, box)
    merged = merge(
        np.asarray(prediction[region], dtype=np.uint8), values, box, complete_rois
    )
    return merged, values != UNLABELED


def label_shard(
    merged: np.ndarray,
    labeled: np.ndarray,
    slab: int = SLAB,
) -> bytes:
    """
    The summary `find_specks` needs from one shard, as an `.npz`: each
    piece's size, class, and whether it holds labels, and the piece ids on
    the shard's six faces.
    """
    values = class_values(merged, slab)
    ids = np.empty(merged.shape, dtype=np.int32)
    mask = np.empty(merged.shape, dtype=bool)
    faces = {
        name: np.zeros(ids[at].shape, dtype=np.int32) for name, at in _FACES.items()
    }
    sizes = [np.zeros(1, dtype=np.int64)]
    held = [np.zeros(1, dtype=bool)]
    classes = [np.zeros(1, dtype=np.uint8)]
    offset = 0
    for value in values:
        count = label_class(merged, value, mask, ids)
        size = np.zeros(count + 1, dtype=np.int64)
        hold = np.zeros(count + 1, dtype=bool)
        for z in _slabs(len(ids), slab):
            counted = np.bincount(ids[z].ravel())
            size[: len(counted)] += counted
            hold[ids[z][labeled[z]]] = True
        sizes.append(size[1:])
        held.append(hold[1:])
        classes.append(np.full(count, value, dtype=np.uint8))
        for name, at in _FACES.items():
            np.add(ids[at], offset, out=faces[name], where=ids[at] > 0)
        offset += count
    return npz(
        sizes=np.concatenate(sizes),
        classes=np.concatenate(classes),
        labeled=np.concatenate(held),
        **faces,
    )


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


def apply_shard(
    merged: np.ndarray,
    remove: np.ndarray,
    slab: int = SLAB,
) -> np.ndarray:
    """
    The shard's final classes: its merged classes without the specks (piece
    ids as `label_shard` numbers them). Writes into `merged` and returns it.
    """
    if len(remove) == 0:
        return merged
    drop = np.zeros(int(remove.max()) + 1, dtype=bool)
    drop[remove] = True
    values = class_values(merged, slab)
    ids = np.empty(merged.shape, dtype=np.int32)
    mask = np.empty(merged.shape, dtype=bool)
    offset = 0
    for value in values:
        if offset + 1 >= len(drop):
            break  # no specks in this class or the ones after it
        # Turning this class's specks into background leaves the other
        # classes' pieces as they were.
        count = label_class(merged, value, mask, ids)
        table = np.zeros(count + 1, dtype=bool)
        found = drop[offset + 1 : offset + count + 1]
        table[1 : len(found) + 1] = found
        if table.any():
            for z in _slabs(len(ids), slab):
                merged[z][table[ids[z]]] = BACKGROUND
        offset += count
    return merged


__all__ = [
    "SLAB",
    "apply_shard",
    "class_values",
    "find_specks",
    "label_class",
    "label_shard",
    "merge",
    "npz",
    "read_label_box",
    "shard_boxes",
    "shard_grid",
    "shard_inputs",
    "slab_depth",
]
