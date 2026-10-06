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

- `label_shard` labels one merged shard's pieces and decides the ones that
  touch no seam (a face shared with another shard), since they are whole;
  it summarizes the rest, the seam pieces, for `find_specks`;
- `find_specks` joins seam pieces that touch across seams (a union-find, as
  connected components of a sparse graph), adds up their sizes, and says
  for each shard which seam pieces are specks;
- `apply_shard` writes the shard's final classes.

So joining takes memory for the pieces on seams, however many specks there
are inside shards. Labeling is deterministic, so `apply_shard` relabels
instead of storing piece ids. The final segmentation has the prediction's
layout: a `class` array of stored label values (1 background, 2..254
classes).

A shard's jobs hold its merged classes and one int32 array of piece ids,
reused for each class in turn; sizes and lookups go a slab of z planes at a
time, so their int64 copies stay small (`slab_depth`). Each step raises
`TooLarge` instead of taking more than the memory budget it is given.
"""

import io
import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NamedTuple, cast

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
# What labeling a class takes per voxel (piece ids, a mask, and a byte that
# scipy's labeling uses itself), and per piece: its size, whether it holds
# labels or touches a seam, and a slab's count of it.
_LABEL_BYTES_PER_VOXEL = 6
_LABEL_BYTES_PER_PIECE = 24
# What joining takes at its peak, per seam piece and per pair of touching
# pieces (scipy's connected components copies the graph, and its transpose,
# with float64 weights), and what pairing the pieces on a seam takes per
# voxel of it.
_JOIN_BYTES_PER_PIECE = 48
_JOIN_BYTES_PER_PAIR = 64
_PAIRING_BYTES_PER_VOXEL = 96

# Told the fraction of a step done so far.
Progress = Callable[[float], None]


class TooLarge(Exception):
    """A step would take more memory than its job may use."""


def _fits(needed: int, budget_bytes: int | None, what: str) -> None:
    if budget_bytes is None or needed <= budget_bytes:
        return
    budget = (
        f"{budget_bytes / 1024**3:.1f} GiB"
        if budget_bytes >= 1024**3
        else f"{budget_bytes / 1024**2:.0f} MiB"
    )
    raise TooLarge(
        f"{what} would take more than the {budget} a job may use on this "
        "worker; run it on a worker with more memory per job."
    )


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


def _shape(block: np.ndarray) -> str:
    return "×".join(str(n) for n in block.shape)


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


def seams(box: Box, shape_zyx: Sequence[int]) -> tuple[bool, ...]:
    """
    Which of a shard's faces (first z, last z, first y, ...) are seams: faces
    it shares with another shard.
    """
    return tuple(
        side for a in range(3) for side in (box[a] > 0, box[a + 3] < shape_zyx[a])
    )


class ShardPieces(NamedTuple):
    """What `label_shard` found in one shard."""

    # For `find_specks`, as an `.npz`: the seam pieces' sizes and classes
    # (index 0 is "no piece"), and their numbers on the shard's seams.
    summary: bytes
    # Which pieces are specks, by id in the shard (index 0 is "no piece"):
    # the ones that touch no seam; seam pieces wait for `find_specks`.
    specks: np.ndarray
    # The seam pieces' ids in the shard: seam piece n is `seam_ids[n - 1]`.
    seam_ids: np.ndarray


def label_shard(
    merged: np.ndarray,
    labeled: np.ndarray,
    min_voxels: int,
    seams: Sequence[bool],
    slab: int = SLAB,
    budget_bytes: int | None = None,
    progress: Progress | None = None,
) -> ShardPieces:
    """
    Label one shard's pieces. A piece that touches no seam is whole already,
    so it is decided here: a speck if it is smaller than `min_voxels` and
    nobody labeled any of it. The rest, the seam pieces, are numbered from 1
    for `find_specks`, which only needs to know whether each one is big
    enough to keep, so their sizes are capped at `min_voxels` (and a labeled
    one counts as `min_voxels`, since it stays).

    Raises `TooLarge` rather than take more than `budget_bytes`, counting
    the merged classes and labeled voxels the caller holds. Calls `progress`
    with the fraction done after each slab.
    """
    names = [name for name, seam in zip(_FACES, seams, strict=True) if seam]
    faces = {name: np.zeros(merged[_FACES[name]].shape, np.int32) for name in names}
    sizes = [np.zeros(1, dtype=np.int32)]
    classes = [np.zeros(1, dtype=np.uint8)]
    specks = [np.zeros(1, dtype=bool)]
    seam_ids = [np.zeros(0, dtype=np.int32)]
    # Without a minimum above one voxel, no piece is a speck.
    values = class_values(merged, slab) if min_voxels > 1 else []
    # Labeling (6 bytes a voxel: piece ids, a mask, and scipy's own byte), a
    # slab's int64 copies, and the seams' piece numbers with what sorting out
    # the seam pieces takes.
    seam_voxels = sum(face.size for face in faces.values())
    fixed = (
        merged.nbytes
        + labeled.nbytes
        + _LABEL_BYTES_PER_VOXEL * merged.size
        + 12 * merged[:slab].size
        + 36 * seam_voxels
    )
    if values:
        _fits(fixed, budget_bytes, f"Labeling a {_shape(merged)} shard")
    ids = np.empty(merged.shape if values else 0, dtype=np.int32)
    mask = np.empty(merged.shape if values else 0, dtype=bool)
    offset = numbered = 0
    for done, value in enumerate(values):
        count = label_class(merged, value, mask, ids)
        _fits(
            fixed + _LABEL_BYTES_PER_PIECE * count + offset + 9 * numbered,
            budget_bytes,
            f"Sorting out {count:,} pieces of class {value} in one shard",
        )
        size = np.zeros(count + 1, dtype=np.int32)
        held = np.zeros(count + 1, dtype=bool)
        for z in _slabs(len(ids), slab):
            counted = np.bincount(ids[z].ravel())
            size[: len(counted)] += counted
            held[np.where(labeled[z], ids[z], 0)] = True
            if progress:
                progress((done + min(z.stop, len(ids)) / len(ids)) / len(values))
        on_seam = np.zeros(count + 1, dtype=bool)
        for name in names:
            on_seam[ids[_FACES[name]]] = True
        on_seam[0] = False
        small = size < min_voxels
        small &= ~held
        small &= ~on_seam
        specks.append(small[1:])
        del small
        seam = np.flatnonzero(on_seam)
        capped = np.minimum(size[seam], min_voxels)
        capped[held[seam]] = min_voxels
        sizes.append(capped)
        classes.append(np.full(len(seam), value, dtype=np.uint8))
        for name in names:
            plane = ids[_FACES[name]]
            inside = plane > 0
            faces[name][inside] = np.searchsorted(seam, plane[inside]) + numbered + 1
        seam += offset
        seam_ids.append(seam.astype(np.int32))
        offset += count
        numbered += len(seam)
    del ids, mask
    number = np.min_scalar_type(numbered)
    summary = npz(
        sizes=np.concatenate(sizes),
        classes=np.concatenate(classes),
        **{name: face.astype(number) for name, face in faces.items()},
    )
    return ShardPieces(summary, np.concatenate(specks), np.concatenate(seam_ids))


def seam_pairs(
    a: np.ndarray, classes_a: np.ndarray, b: np.ndarray, classes_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    The distinct pairs of pieces of one class that touch across a seam, given
    the piece numbers on either side of it (0 for none) and their classes.
    """
    a, b = a.ravel(), b.ravel()
    touching = (a > 0) & (b > 0)
    a, b = a[touching], b[touching]
    same = classes_a[a] == classes_b[b]
    pairs = np.unique((a[same].astype(np.int64) << 32) | b[same].astype(np.int64))
    return pairs >> 32, pairs & 0xFFFFFFFF


def merge_bytes(pieces: int, pairs: int) -> int:
    """About how much memory `find_specks` takes to join this many pieces."""
    return pieces * _JOIN_BYTES_PER_PIECE + pairs * _JOIN_BYTES_PER_PAIR


class Joined(NamedTuple):
    """What `find_specks` found."""

    # Which of each shard's seam pieces are specks, by number (index 0 is
    # "no piece").
    specks: list[np.ndarray]
    # How many distinct pairs of pieces touch across seams.
    pairs: int


def find_specks(
    summaries: Iterable[bytes],
    grid: Sequence[int],
    min_voxels: int,
    budget_bytes: int | None = None,
) -> Joined:
    """
    Given each shard's summary from `label_shard` (in `shard_boxes` order),
    which of each shard's seam pieces are specks: seam pieces of one class
    that touch across seams are joined into whole pieces (a union-find, as
    connected components of a sparse graph with one edge per pair of
    pieces), and the whole pieces smaller than `min_voxels` that nobody
    labeled are specks.

    Summaries are read one at a time, and a shard's last faces are kept only
    until the shard after it on that axis has been read. Raises `TooLarge`
    as soon as the pieces and pairs found would take more than
    `budget_bytes` to join.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    grid = tuple(int(n) for n in grid)
    sizes: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    starts = [0]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    pairs = 0
    # Shards whose last faces wait for the next shard along an axis, by
    # (axis, that shard's coords): the summary, its classes, where its
    # pieces start, and its size.
    waiting: dict[tuple[int, tuple[int, ...]], tuple[Any, np.ndarray, int, int]] = {}

    def check(extra: int = 0) -> None:
        kept = {id(shard[0]): shard[3] for shard in waiting.values()}
        _fits(
            merge_bytes(starts[-1], pairs) + sum(kept.values()) + extra,
            budget_bytes,
            "Joining the pieces that cross shard boundaries",
        )

    for here, raw in zip(
        itertools.product(*(range(n) for n in grid)), summaries, strict=True
    ):
        summary = np.load(io.BytesIO(raw))
        start = starts[-1]
        sizes.append(summary["sizes"])
        classes.append(summary["classes"])
        starts.append(start + len(classes[-1]))
        check(len(raw))
        for axis, name in enumerate("zyx"):
            if (before := waiting.pop((axis, here), None)) is not None:
                faces, classes_before, start_before, _ = before
                last, first = faces[f"last_{name}"], summary[f"first_{name}"]
                check(len(raw) + _PAIRING_BYTES_PER_VOXEL * last.size)
                a, b = seam_pairs(last, classes_before, first, classes[-1])
                rows.append(a + start_before)
                cols.append(b + start)
                pairs += len(a)
            if here[axis] + 1 < grid[axis]:
                after = tuple(n + (i == axis) for i, n in enumerate(here))
                waiting[(axis, after)] = (summary, classes[-1], start, len(raw))
    check()
    total = starts[-1]
    row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    del rows, cols
    graph = coo_matrix(
        (np.ones(len(row), dtype=bool), (row, col)), shape=(total, total)
    )
    del row, col
    _, component = connected_components(graph, directed=False)
    del graph
    whole = np.bincount(component, weights=np.concatenate(sizes))
    speck = (whole < min_voxels)[component]
    del whole, component
    speck &= np.concatenate(classes) >= FIRST_CLASS
    return Joined(
        [speck[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)], pairs
    )


def shard_specks(
    specks: np.ndarray, seam_ids: np.ndarray, joined: np.ndarray
) -> np.ndarray:
    """
    Which of a shard's pieces are specks, for `apply_shard`: `label_shard`'s
    `specks`, with its seam pieces filled in from `find_specks` (`joined`).
    Writes into `specks` and returns it.
    """
    specks[seam_ids] = joined[1:]
    return specks


def apply_shard(
    merged: np.ndarray,
    specks: np.ndarray,
    slab: int = SLAB,
    budget_bytes: int | None = None,
    progress: Progress | None = None,
) -> np.ndarray:
    """
    The shard's final classes: its merged classes without the specks (which
    pieces are specks, by id as `label_shard` numbers them). Writes into
    `merged` and returns it.

    Raises `TooLarge` rather than take more than `budget_bytes`, counting
    the merged classes and specks the caller holds. Calls `progress` with
    the fraction done after each class and slab.
    """
    if not specks.any():
        return merged
    # Classes after the last speck's class have none.
    last = len(specks) - 1 - int(np.argmax(specks[::-1]))
    # Labeling, a slab's lookups, and one class's table.
    _fits(
        merged.nbytes
        + 2 * specks.nbytes
        + _LABEL_BYTES_PER_VOXEL * merged.size
        + 2 * merged[:slab].size,
        budget_bytes,
        f"Removing specks from a {_shape(merged)} shard",
    )
    values = class_values(merged, slab)
    ids = np.empty(merged.shape, dtype=np.int32)
    mask = np.empty(merged.shape, dtype=bool)
    offset = 0
    for done, value in enumerate(values):
        if offset >= last:
            break
        # Turning this class's specks into background leaves the other
        # classes' pieces as they were.
        count = label_class(merged, value, mask, ids)
        if offset + count >= len(specks):
            raise ValueError("The shard has more pieces than when it was labeled.")
        table = np.zeros(count + 1, dtype=bool)
        table[1:] = specks[offset + 1 : offset + count + 1]
        if table.any():
            for z in _slabs(len(ids), slab):
                np.copyto(merged[z], BACKGROUND, where=table[ids[z]])
                if progress:
                    progress((done + min(z.stop, len(ids)) / len(ids)) / len(values))
        offset += count
        if progress:
            progress((done + 1) / len(values))
    return merged


__all__ = [
    "SLAB",
    "Joined",
    "Progress",
    "ShardPieces",
    "TooLarge",
    "apply_shard",
    "class_values",
    "find_specks",
    "label_class",
    "label_shard",
    "merge",
    "merge_bytes",
    "npz",
    "read_label_box",
    "seam_pairs",
    "seams",
    "shard_boxes",
    "shard_grid",
    "shard_inputs",
    "shard_specks",
    "slab_depth",
]
