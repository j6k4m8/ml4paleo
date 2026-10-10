"""
Label edits as per-chunk mask deltas, and the claims that make undo exact.

The browser turns every committed annotation action (a brush stroke, a closed
polygon, a fill, an accepted proposal) into one op, made of one `ChunkDelta`
per affected chunk. The client rasterizes the shape itself and sends the
resulting mask, so what the user saw locally is exactly what gets committed.
Workers (propagation, interactive models, importers) build deltas the same way
with `split_into_deltas`.

Applying a delta records a `Claim`: the voxels the op actually wrote (after
its `only_if` condition) and the values it wrote there. The label state of a
chunk is always the overlay of every live (not undone) op's claim, in op
order: each voxel holds the value of the last live claim that covers it, or
0 if none does. Undo and redo therefore only flip an op between live and
undone and recompute that op's voxels from the remaining live claims
(`recompute`). The result never depends on the order of undos, and an undo
never disturbs voxels that a later op also wrote.

Every payload decoded here may come from a browser, so decompression is
bounded by the size the box allows.
"""

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import zstandard
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from . import BACKGROUND, DECLINED, LABEL_CHUNK_ZYX, MAX_CLASS, UNLABELED, Source
from .codec import FRAME_OVERHEAD, decompress_exact

ChunkKey = tuple[int, int, int]
Box = tuple[int, int, int, int, int, int]

_ONLY_IF = re.compile(
    r"any|unlabeled|labeled|declined|class:([0-9]{1,3}(?:,[0-9]{1,3})*)"
)
# The longest list `_ONLY_IF` allows is every class value, three digits each.
_MAX_ONLY_IF = len("class:") + 4 * MAX_CLASS
_MAX_RAW_BYTES = int(np.prod(LABEL_CHUNK_ZYX))


def _compress(raw: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=3).compress(raw)


def _decompress(data: bytes, size: int) -> bytes:
    return decompress_exact(data, size)


def normalize_only_if(only_if: str) -> str:
    """
    The canonical form of an `only_if` condition: "any", "unlabeled", "labeled",
    "declined", or "class:" and the values it names, once each and ascending
    ("class:2,3,5").
    The values may come in any order and with leading zeros; none may be 0
    (unlabeled, which has its own condition), past `MAX_CLASS`, or repeated.
    Raises ValueError for anything else.
    """
    match = _ONLY_IF.fullmatch(only_if) if len(only_if) <= _MAX_ONLY_IF else None
    if match is None:
        raise ValueError(f"Invalid only_if: {only_if!r}")
    listed = match.group(1)
    if listed is None:
        return only_if
    values = [int(value) for value in listed.split(",")]
    if len(set(values)) != len(values):
        raise ValueError(f"Invalid only_if: {only_if!r} names a class twice")
    if not all(BACKGROUND <= value <= MAX_CLASS for value in values):
        raise ValueError(
            f"Invalid only_if: {only_if!r} names a value that is not a class"
        )
    return "class:" + ",".join(str(value) for value in sorted(values))


def only_if_classes(only_if: str) -> list[int] | None:
    """
    The values a normalized "class:..." condition names, or None for the other
    conditions.
    """
    if not only_if.startswith("class:"):
        return None
    return [int(value) for value in only_if[len("class:") :].split(",")]


def pack_mask(mask: np.ndarray) -> bytes:
    """
    Pack a boolean mask (C order, little-endian bit order) and compress it.
    """
    bits = np.packbits(
        np.ascontiguousarray(mask, dtype=bool).ravel(), bitorder="little"
    )
    return _compress(bits.tobytes())


def unpack_mask(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """
    Reverse `pack_mask`.
    """
    count = int(np.prod(shape))
    bits = np.frombuffer(_decompress(data, (count + 7) // 8), dtype=np.uint8)
    return (
        np.unpackbits(bits, count=count, bitorder="little").astype(bool).reshape(shape)
    )


def pack_values(values: np.ndarray) -> bytes:
    return _compress(np.ascontiguousarray(values, dtype=np.uint8).tobytes())


def unpack_values(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    raw = _decompress(data, int(np.prod(shape)))
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def _box_shape(box: Box) -> tuple[int, int, int]:
    z0, y0, x0, z1, y1, x1 = box
    return (z1 - z0, y1 - y0, x1 - x0)


def _box_slices(box: Box) -> tuple[slice, slice, slice]:
    z0, y0, x0, z1, y1, x1 = box
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def _check_box(box: Box) -> None:
    for a, b, size in zip(box[:3], box[3:], LABEL_CHUNK_ZYX, strict=True):
        if not 0 <= a < b <= size:
            raise ValueError(f"Box {box} is not inside a {LABEL_CHUNK_ZYX} chunk")


class ChunkDelta(BaseModel):
    """
    One op's change to one label chunk.

    `box` is chunk-local (z0, y0, x0, z1, y1, x1), half-open. `mask` selects
    voxels within the box (see `pack_mask`). Selected voxels get `value`, or the
    per-voxel `values` (box-shaped, see `pack_values`) for multi-class edits;
    value 0 erases. `only_if` limits which voxels may change (judged on the
    chunk as it is when the delta is applied): "any" (no limit), "unlabeled"
    (voxels holding 0), "labeled" (voxels holding anything but 0, background
    included), "declined" (voxels holding the reserved decline tombstone), or
    "class:2,3" (voxels holding one of the values listed, as
    `normalize_only_if` says). Nothing outside the mask changes.
    """

    model_config = ConfigDict(frozen=True)

    key: ChunkKey
    base_version: int = 0
    box: Box
    mask: bytes
    value: int | None = None
    values: bytes | None = None
    only_if: str = "any"

    @field_validator("only_if")
    @classmethod
    def _normalized(cls, only_if: str) -> str:
        return normalize_only_if(only_if)

    @model_validator(mode="after")
    def _check(self) -> "ChunkDelta":
        if any(k < 0 for k in self.key):
            raise ValueError(f"Chunk key must be non-negative: {self.key}")
        _check_box(self.box)
        if (self.value is None) == (self.values is None):
            raise ValueError("Set exactly one of value or values")
        if self.value is not None and not 0 <= self.value <= DECLINED:
            raise ValueError(f"Label value {self.value} is out of range")
        for payload in (self.mask, self.values or b""):
            if len(payload) > _MAX_RAW_BYTES + FRAME_OVERHEAD:
                raise ValueError("Delta payload is too large")
        return self

    @property
    def box_shape(self) -> tuple[int, int, int]:
        return _box_shape(self.box)

    @property
    def box_slices(self) -> tuple[slice, slice, slice]:
        return _box_slices(self.box)

    def check_within(self, volume_shape_zyx: Sequence[int]) -> None:
        """
        Raise ValueError if the delta's box reaches past the volume's edge.
        Chunks at the far edges of a volume are only partly inside it.
        """
        for k, lo, hi, size, extent in zip(
            self.key,
            self.box[:3],
            self.box[3:],
            LABEL_CHUNK_ZYX,
            volume_shape_zyx,
            strict=True,
        ):
            if k * size + lo < 0 or k * size + hi > extent:
                raise ValueError(
                    f"Delta for chunk {self.key} reaches outside the volume"
                )


class Claim(BaseModel):
    """
    What one op wrote to one chunk: the claimed voxels (`mask`, within `box`),
    the class values written there (`values`, box-shaped), and the source it
    recorded. Erasing claims voxels with value 0.
    """

    model_config = ConfigDict(frozen=True)

    key: ChunkKey
    box: Box
    mask: bytes
    values: bytes
    source: Source

    @property
    def box_shape(self) -> tuple[int, int, int]:
        return _box_shape(self.box)


@dataclass(frozen=True)
class Applied:
    class_chunk: np.ndarray
    source_chunk: np.ndarray
    # Voxels whose class or source changed.
    changed: int


@dataclass(frozen=True)
class AppliedDelta(Applied):
    claim: Claim


def _source_for(values: np.ndarray, source: Source) -> np.ndarray:
    return np.where(values == UNLABELED, Source.NONE, source).astype(np.uint8)


def apply_delta(
    class_chunk: np.ndarray,
    source_chunk: np.ndarray,
    delta: ChunkDelta,
    source: Source,
) -> AppliedDelta:
    """
    Apply a delta to copies of a chunk's `class` and `source` arrays, and
    return the claim to record for the op.

    Written voxels get `source` (erased voxels get `Source.NONE`), even when
    their class doesn't change, so a human repainting a model's label marks it
    as human-made.
    """
    shape = delta.box_shape
    slices = delta.box_slices
    new_class = class_chunk.copy()
    new_source = source_chunk.copy()
    region = new_class[slices]
    source_region = new_source[slices]

    selected = unpack_mask(delta.mask, shape)
    if delta.only_if == "unlabeled":
        selected &= region == UNLABELED
    elif delta.only_if == "labeled":
        selected &= region != UNLABELED
    elif delta.only_if == "declined":
        selected &= region == DECLINED
    elif (classes := only_if_classes(delta.only_if)) is not None:
        selected &= np.isin(region, classes)

    if delta.value is not None:
        written = np.full(shape, delta.value, dtype=np.uint8)
    else:
        written = unpack_values(delta.values or b"", shape)
        if written[selected].max(initial=0) > DECLINED:
            raise ValueError("Delta writes a reserved label value")
    written = np.where(selected, written, 0).astype(np.uint8)
    written_source = _source_for(written, source)

    changed = selected & ((region != written) | (source_region != written_source))
    region[selected] = written[selected]
    source_region[selected] = written_source[selected]
    return AppliedDelta(
        class_chunk=new_class,
        source_chunk=new_source,
        changed=int(changed.sum()),
        claim=Claim(
            key=delta.key,
            box=delta.box,
            mask=pack_mask(selected),
            values=pack_values(written),
            source=source,
        ),
    )


def recompute(
    class_chunk: np.ndarray,
    source_chunk: np.ndarray,
    region: Claim,
    live_claims: Sequence[Claim],
) -> Applied:
    """
    Recompute the voxels of `region` as the overlay of `live_claims` (every
    live op's claim on this chunk, in op order). Voxels no live claim covers
    become unlabeled.

    To undo an op, pass its claim as `region` and the other live claims. To
    redo it, pass its claim as `region` and the live claims including it.
    """
    shape = region.box_shape
    slices = _box_slices(region.box)
    target = unpack_mask(region.mask, shape)
    overlay_class = np.zeros(shape, dtype=np.uint8)
    overlay_source = np.zeros(shape, dtype=np.uint8)

    for claim in live_claims:
        if claim.key != region.key:
            raise ValueError("Claims must all belong to the region's chunk")
        lo = [max(a, b) for a, b in zip(claim.box[:3], region.box[:3], strict=True)]
        hi = [min(a, b) for a, b in zip(claim.box[3:], region.box[3:], strict=True)]
        if any(a >= b for a, b in zip(lo, hi, strict=True)):
            continue
        claim_slices = tuple(
            slice(a - c, b - c) for a, b, c in zip(lo, hi, claim.box[:3], strict=True)
        )
        region_slices = tuple(
            slice(a - r, b - r) for a, b, r in zip(lo, hi, region.box[:3], strict=True)
        )
        covers = unpack_mask(claim.mask, claim.box_shape)[claim_slices]
        covers &= target[region_slices]
        values = unpack_values(claim.values, claim.box_shape)[claim_slices]
        overlay_class[region_slices][covers] = values[covers]
        overlay_source[region_slices][covers] = _source_for(values, claim.source)[
            covers
        ]

    new_class = class_chunk.copy()
    new_source = source_chunk.copy()
    current_class = new_class[slices]
    current_source = new_source[slices]
    changed = target & (
        (current_class != overlay_class) | (current_source != overlay_source)
    )
    current_class[target] = overlay_class[target]
    current_source[target] = overlay_source[target]
    return Applied(
        class_chunk=new_class, source_chunk=new_source, changed=int(changed.sum())
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
    if value is not None and not 0 <= value <= DECLINED:
        raise ValueError(f"Label value {value} is out of range")
    if values is not None:
        if values.shape != mask.shape:
            raise ValueError("values must have the same shape as mask")
        selected_values = values[np.asarray(mask, dtype=bool)]
        if selected_values.size and (
            selected_values.min() < 0 or selected_values.max() > DECLINED
        ):
            raise ValueError("values contain label values out of range")
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
