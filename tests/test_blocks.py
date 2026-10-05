"""
Block grids must tile an array exactly, so chunked jobs neither skip nor
double-process any voxel.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml4paleo.blocks import Block, block_count, block_ranges, iter_blocks

shapes = st.lists(st.integers(1, 12), min_size=1, max_size=3)


@st.composite
def shape_and_block_shape(draw):
    shape = draw(shapes)
    block_shape = [draw(st.integers(1, 6)) for _ in shape]
    return shape, block_shape


@given(shape_and_block_shape())
def test_blocks_cover_every_voxel_exactly_once(case):
    shape, block_shape = case
    coverage = np.zeros(shape, dtype=np.int32)
    blocks = list(iter_blocks(shape, block_shape))
    for block in blocks:
        coverage[block.slices] += 1
        assert all(
            size <= limit for size, limit in zip(block.shape, block_shape, strict=True)
        )
    assert (coverage == 1).all()
    assert len(blocks) == block_count(shape, block_shape)


@given(shape_and_block_shape(), st.integers(0, 4))
def test_expanding_and_cropping_round_trips(case, halo_size):
    shape, block_shape = case
    data = np.arange(np.prod(shape)).reshape(shape)
    halo = [halo_size] * len(shape)
    for block in iter_blocks(shape, block_shape):
        outer = block.expand(halo, shape)
        assert all(0 <= lo for lo in outer.start)
        assert all(hi <= size for hi, size in zip(outer.stop, shape, strict=True))
        cropped = data[outer.slices][block.relative_to(outer)]
        np.testing.assert_array_equal(cropped, data[block.slices])


def test_block_ranges_matches_the_old_block_compute_format():
    ranges = block_ranges((5, 3, 2), (2, 2, 2))
    assert ((4, 5), (2, 3), (0, 2)) in ranges
    assert len(ranges) == 3 * 2 * 1


def test_invalid_blocks_are_rejected():
    with pytest.raises(ValueError):
        Block(start=(2,), stop=(1,))
    with pytest.raises(ValueError):
        list(iter_blocks((4, 4), (0, 2)))
    with pytest.raises(ValueError):
        Block(start=(1, 1), stop=(2, 2)).relative_to(Block(start=(0, 0), stop=(1, 1)))
