"""
Label edits as per-chunk mask deltas.

The browser turns every committed annotation action (a brush stroke, a closed
polygon, a fill, an accepted proposal) into one op, made of one `ChunkDelta`
per affected chunk. The client rasterizes the shape itself and sends the
resulting mask, so what the user saw locally is exactly what gets committed.
Workers (propagation, interactive models, importers) build deltas the same way
with `split_into_deltas`.

Applying a delta returns an `UndoPatch` with the previous values of every voxel
it changed. Reverting a patch only restores voxels that still hold the value
the op wrote, so undoing an old op never overwrites a collaborator's later
edits.
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
from numcodecs import Zstd
from pydantic import BaseModel, ConfigDict, model_validator

from . import LABEL_CHUNK_ZYX, MAX_CLASS, UNLABELED, Source

ChunkKey = tuple[int, int, int]
Box = tuple[int, int, int, int, int, int]

_zstd = Zstd(level=3)
_ONLY_IF = re.compile(r"any|unlabeled|class:(\d{1,3})")


def pack_mask(mask: np.ndarray) -> bytes:
    """
    Pack a boolean mask (C order, little-endian bit order) and compress it.
    """
    bits = np.packbits(
        np.ascontiguousarray(mask, dtype=bool).ravel(), bitorder="little"
    )
    return bytes(_zstd.encode(bits))


def unpack_mask(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """
    Reverse `pack_mask`.
    """
    bits = np.frombuffer(_zstd.decode(data), dtype=np.uint8)
    count = int(np.prod(shape))
    if bits.size != (count + 7) // 8:
        raise ValueError(f"Packed mask has {bits.size} bytes for {count} voxels")
    return (
        np.unpackbits(bits, count=count, bitorder="little").astype(bool).reshape(shape)
    )


def pack_values(values: np.ndarray) -> bytes:
    return bytes(_zstd.encode(np.ascontiguousarray(values, dtype=np.uint8)))


def unpack_values(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    values = np.frombuffer(_zstd.decode(data), dtype=np.uint8)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"Packed values have {values.size} voxels, expected {shape}")
    return values.reshape(shape).copy()


class ChunkDelta(BaseModel):
    """
    One op's change to one label chunk.

    `box` is chunk-local (z0, y0, x0, z1, y1, x1), half-open. `mask` selects
    voxels within the box (see `pack_mask`). Selected voxels get `value`, or the
    per-voxel `values` (box-shaped, see `pack_values`) for multi-class edits;
    value 0 erases. `only_if` limits which voxels may change: "any",
    "unlabeled", or "class:N".
    """

    model_config = ConfigDict(frozen=True)

    key: ChunkKey
    base_version: int = 0
    box: Box
    mask: bytes
    value: int | None = None
    values: bytes | None = None
    only_if: str = "any"

    @model_validator(mode="after")
    def _check(self) -> "ChunkDelta":
        if any(k < 0 for k in self.key):
            raise ValueError(f"Chunk key must be non-negative: {self.key}")
        lo, hi = self.box[:3], self.box[3:]
        for a, b, size in zip(lo, hi, LABEL_CHUNK_ZYX, strict=True):
            if not 0 <= a < b <= size:
                raise ValueError(
                    f"Box {self.box} is not inside a {LABEL_CHUNK_ZYX} chunk"
                )
        if (self.value is None) == (self.values is None):
            raise ValueError("Set exactly one of value or values")
        if self.value is not None and not 0 <= self.value <= MAX_CLASS:
            raise ValueError(f"Label value {self.value} is out of range")
        match = _ONLY_IF.fullmatch(self.only_if)
        if match is None or (match.group(1) and int(match.group(1)) > MAX_CLASS):
            raise ValueError(f"Invalid only_if: {self.only_if!r}")
        return self

    @property
    def box_shape(self) -> tuple[int, int, int]:
        z0, y0, x0, z1, y1, x1 = self.box
        return (z1 - z0, y1 - y0, x1 - x0)

    @property
    def box_slices(self) -> tuple[slice, slice, slice]:
        z0, y0, x0, z1, y1, x1 = self.box
        return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


class UndoPatch(BaseModel):
    """
    What one delta changed in one chunk, enough to revert it.

    `mask` marks the voxels the delta changed; `prev_*` hold their old values
    and `new_class` the value the delta wrote (all box-shaped).
    """

    model_config = ConfigDict(frozen=True)

    key: ChunkKey
    box: Box
    mask: bytes
    prev_class: bytes
    prev_source: bytes
    new_class: bytes


@dataclass(frozen=True)
class AppliedDelta:
    class_chunk: np.ndarray
    source_chunk: np.ndarray
    changed: int
    undo: UndoPatch


def apply_delta(
    class_chunk: np.ndarray,
    source_chunk: np.ndarray,
    delta: ChunkDelta,
    source: Source,
) -> AppliedDelta:
    """
    Apply a delta to copies of a chunk's `class` and `source` arrays.

    Erased voxels get source `NONE`; written voxels get `source`.
    """
    shape = delta.box_shape
    slices = delta.box_slices
    new_class = class_chunk.copy()
    new_source = source_chunk.copy()
    region = new_class[slices]

    selected = unpack_mask(delta.mask, shape)
    if delta.only_if == "unlabeled":
        selected &= region == UNLABELED
    elif delta.only_if.startswith("class:"):
        selected &= region == int(delta.only_if.split(":")[1])

    if delta.value is not None:
        written = np.full(shape, delta.value, dtype=np.uint8)
    else:
        written = unpack_values(delta.values, shape)  # type: ignore[arg-type]
        if written[selected].max(initial=0) > MAX_CLASS:
            raise ValueError("Delta writes a reserved label value")
    changed = selected & (region != written)

    prev_class = region.copy()
    prev_source = new_source[slices].copy()
    region[changed] = written[changed]
    source_region = new_source[slices]
    source_region[changed] = np.where(
        written[changed] == UNLABELED, Source.NONE, source
    )

    return AppliedDelta(
        class_chunk=new_class,
        source_chunk=new_source,
        changed=int(changed.sum()),
        undo=UndoPatch(
            key=delta.key,
            box=delta.box,
            mask=pack_mask(changed),
            prev_class=pack_values(prev_class),
            prev_source=pack_values(prev_source),
            new_class=pack_values(region),
        ),
    )


def revert_patch(
    class_chunk: np.ndarray, source_chunk: np.ndarray, patch: UndoPatch
) -> AppliedDelta:
    """
    Undo a patch on copies of a chunk, restoring only voxels that still hold
    the value the patch's op wrote. The returned `undo` re-applies the change
    (redo).
    """
    z0, y0, x0, z1, y1, x1 = patch.box
    shape = (z1 - z0, y1 - y0, x1 - x0)
    slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    changed_by_op = unpack_mask(patch.mask, shape)
    prev_class = unpack_values(patch.prev_class, shape)
    prev_source = unpack_values(patch.prev_source, shape)
    op_class = unpack_values(patch.new_class, shape)

    new_class = class_chunk.copy()
    new_source = source_chunk.copy()
    region = new_class[slices]
    source_region = new_source[slices]
    restorable = changed_by_op & (region == op_class)

    redo = UndoPatch(
        key=patch.key,
        box=patch.box,
        mask=pack_mask(restorable),
        prev_class=pack_values(region.copy()),
        prev_source=pack_values(source_region.copy()),
        new_class=pack_values(np.where(restorable, prev_class, region)),
    )
    region[restorable] = prev_class[restorable]
    source_region[restorable] = prev_source[restorable]
    return AppliedDelta(
        class_chunk=new_class,
        source_chunk=new_source,
        changed=int(restorable.sum()),
        undo=redo,
    )


def chunk_keys_for_box(start_zyx, stop_zyx) -> Iterator[ChunkKey]:
    """
    Yield the keys of every label chunk that a global half-open box touches.
    """
    ranges = [
        range(lo // size, (hi - 1) // size + 1)
        for lo, hi, size in zip(start_zyx, stop_zyx, LABEL_CHUNK_ZYX, strict=True)
    ]
    for cz in ranges[0]:
        for cy in ranges[1]:
            for cx in ranges[2]:
                yield (cz, cy, cx)


def split_into_deltas(
    mask: np.ndarray,
    origin_zyx: tuple[int, int, int],
    *,
    value: int | None = None,
    values: np.ndarray | None = None,
    only_if: str = "any",
    base_versions: Mapping[ChunkKey, int] | None = None,
) -> list[ChunkDelta]:
    """
    Split a global mask (placed at `origin_zyx`) into one delta per chunk it
    touches, each cropped to the tight box around its selected voxels.

    Pass `value` for single-class edits or `values` (shaped like `mask`) for
    per-voxel values.
    """
    if (value is None) == (values is None):
        raise ValueError("Pass exactly one of value or values")
    if values is not None and values.shape != mask.shape:
        raise ValueError("values must have the same shape as mask")
    if any(o < 0 for o in origin_zyx):
        raise ValueError(f"Mask origin must be non-negative: {origin_zyx}")
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    nonzero = np.nonzero(mask)
    start = [int(o + idx.min()) for o, idx in zip(origin_zyx, nonzero, strict=True)]
    stop = [int(o + idx.max() + 1) for o, idx in zip(origin_zyx, nonzero, strict=True)]

    deltas = []
    for key in chunk_keys_for_box(start, stop):
        chunk_origin = [k * size for k, size in zip(key, LABEL_CHUNK_ZYX, strict=True)]
        # The part of the mask inside this chunk, in mask coordinates.
        lo = [max(c - o, 0) for c, o in zip(chunk_origin, origin_zyx, strict=True)]
        hi = [
            min(c + size - o, extent)
            for c, size, o, extent in zip(
                chunk_origin, LABEL_CHUNK_ZYX, origin_zyx, mask.shape, strict=True
            )
        ]
        if any(a >= b for a, b in zip(lo, hi, strict=True)):
            continue
        part = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        if not part.any():
            continue
        part_nonzero = np.nonzero(part)
        tight_lo = [int(i.min()) for i in part_nonzero]
        tight_hi = [int(i.max()) + 1 for i in part_nonzero]
        tight = tuple(slice(a, b) for a, b in zip(tight_lo, tight_hi, strict=True))
        # Box in chunk-local coordinates.
        local_lo = [
            o + a + t - c
            for o, a, t, c in zip(origin_zyx, lo, tight_lo, chunk_origin, strict=True)
        ]
        local_hi = [
            lo_ + (b - a)
            for lo_, a, b in zip(local_lo, tight_lo, tight_hi, strict=True)
        ]
        box_values = None
        if values is not None:
            box_values = pack_values(
                values[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]][tight]
            )
        deltas.append(
            ChunkDelta(
                key=key,
                base_version=(base_versions or {}).get(key, 0),
                box=(*local_lo, *local_hi),  # type: ignore[arg-type]
                mask=pack_mask(part[tight]),
                value=value,
                values=box_values,
                only_if=only_if,
            )
        )
    return deltas
