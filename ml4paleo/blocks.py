"""
Block grids for chunked processing of large arrays.

Jobs that read or write a large volume split it into blocks aligned to a grid
that starts at the origin. Aligning blocks to the storage chunk or shard grid
means each block maps onto whole chunks, so parallel writers never touch the
same chunk.

Blocks are half-open boxes: `start` is inclusive and `stop` is exclusive in
every dimension. Blocks at the far edge of an array are truncated to the array
shape.
"""

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """
    An axis-aligned box `[start, stop)` in array index space.
    """

    start: tuple[int, ...]
    stop: tuple[int, ...]

    def __post_init__(self):
        if len(self.start) != len(self.stop):
            raise ValueError("start and stop must have the same number of dimensions")
        if any(lo > hi for lo, hi in zip(self.start, self.stop, strict=True)):
            raise ValueError(f"Block start {self.start} is past stop {self.stop}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(hi - lo for lo, hi in zip(self.start, self.stop, strict=True))

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def slices(self) -> tuple[slice, ...]:
        """
        Return the slices that select this block from the full array.
        """
        return tuple(
            slice(lo, hi) for lo, hi in zip(self.start, self.stop, strict=True)
        )

    def expand(self, halo: Sequence[int], bounds: Sequence[int]) -> "Block":
        """
        Grow the block by `halo` on every side, clamped to an array of shape `bounds`.

        Use this to read the extra context a model needs around a block.
        """
        _check_dims(self.start, halo, bounds)
        return Block(
            start=tuple(max(0, lo - h) for lo, h in zip(self.start, halo, strict=True)),
            stop=tuple(
                min(size, hi + h)
                for hi, h, size in zip(self.stop, halo, bounds, strict=True)
            ),
        )

    def relative_to(self, outer: "Block") -> tuple[slice, ...]:
        """
        Return the slices that select this block from an array holding `outer`.

        This is how a result computed on an expanded block is cropped back to
        the block itself.
        """
        _check_dims(self.start, outer.start)
        if any(
            lo < outer_lo or hi > outer_hi
            for lo, hi, outer_lo, outer_hi in zip(
                self.start, self.stop, outer.start, outer.stop, strict=True
            )
        ):
            raise ValueError(f"{self} is not inside {outer}")
        return tuple(
            slice(lo - outer_lo, hi - outer_lo)
            for lo, hi, outer_lo in zip(self.start, self.stop, outer.start, strict=True)
        )


def iter_blocks(shape: Sequence[int], block_shape: Sequence[int]) -> Iterator[Block]:
    """
    Yield the blocks of a grid that covers an array of `shape`, in C order.

    The grid starts at the origin. Blocks on the far edges are truncated.
    """
    _check_dims(shape, block_shape)
    if any(b <= 0 for b in block_shape):
        raise ValueError(f"Block shape must be positive, got {tuple(block_shape)}")
    starts_per_dim = [
        range(0, size, step) for size, step in zip(shape, block_shape, strict=True)
    ]
    for start in itertools.product(*starts_per_dim):
        yield Block(
            start=tuple(start),
            stop=tuple(
                min(lo + step, size)
                for lo, step, size in zip(start, block_shape, shape, strict=True)
            ),
        )


def block_count(shape: Sequence[int], block_shape: Sequence[int]) -> int:
    """
    Return the number of blocks `iter_blocks` yields for `shape`.
    """
    _check_dims(shape, block_shape)
    return math.prod(
        math.ceil(size / step) for size, step in zip(shape, block_shape, strict=True)
    )


def block_ranges(
    shape: Sequence[int], block_shape: Sequence[int]
) -> list[tuple[tuple[int, int], ...]]:
    """
    Return each block as a tuple of `(start, stop)` pairs, one per dimension.
    """
    return [
        tuple(zip(block.start, block.stop, strict=True))
        for block in iter_blocks(shape, block_shape)
    ]


def _check_dims(*sequences: Sequence[int]) -> None:
    if len({len(s) for s in sequences}) != 1:
        raise ValueError(f"Dimension mismatch: {[tuple(s) for s in sequences]}")
