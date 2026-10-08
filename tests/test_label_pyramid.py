"""
Coarse label levels: the rule that shrinks a block of labels for display.
"""

from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml4paleo.labels import LABEL_CHUNK_ZYX
from ml4paleo.labels.pyramid import downsample_labels


def reference(labels, step):
    """
    The rule, one block at a time, written out as plainly as possible.
    """
    size = [-(-n // s) for n, s in zip(labels.shape, step, strict=True)]
    out = np.zeros(size, dtype=np.uint8)
    for index in product(*(range(n) for n in size)):
        window = labels[
            tuple(slice(i * s, (i + 1) * s) for i, s in zip(index, step, strict=True))
        ].ravel()
        classes = [int(v) for v in window if v >= 2]
        if classes:
            # Most common class; the lowest value among those that tie.
            out[index] = min(set(classes), key=lambda v: (-classes.count(v), v))
        elif (window == 1).any():
            out[index] = 1
    return out


def block(rows, shape=(2, 2, 2)):
    return np.array(rows, dtype=np.uint8).reshape(shape)


def test_a_block_is_its_most_common_class():
    assert downsample_labels(block([3, 3, 3, 5, 5, 0, 1, 1]))[0, 0, 0] == 3


def test_the_lowest_class_wins_a_tie():
    assert downsample_labels(block([7, 7, 4, 4, 9, 9, 0, 0]))[0, 0, 0] == 4
    assert downsample_labels(block([9, 4, 7, 4, 7, 9, 0, 0]))[0, 0, 0] == 4


def test_a_class_beats_background_however_few_voxels_it_has():
    assert downsample_labels(block([1, 1, 1, 1, 1, 1, 1, 6]))[0, 0, 0] == 6
    assert downsample_labels(block([6, 1, 1, 1, 0, 0, 0, 0]))[0, 0, 0] == 6


def test_background_beats_unlabeled():
    assert downsample_labels(block([0, 0, 0, 0, 0, 0, 0, 1]))[0, 0, 0] == 1
    assert downsample_labels(block([0, 0, 0, 0, 0, 0, 0, 0]))[0, 0, 0] == 0


def test_every_axis_halves_unless_told_otherwise():
    labels = np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    assert downsample_labels(labels).shape == (32, 32, 32)
    assert downsample_labels(labels, (1, 2, 2)).shape == (64, 32, 32)
    assert downsample_labels(labels, (2, 1, 1)).shape == (32, 64, 64)
    assert downsample_labels(labels, (1, 1, 1)).shape == LABEL_CHUNK_ZYX


def test_the_edge_of_an_odd_volume_counts_beyond_it_as_unlabeled():
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[2, 2, 2] = 4
    out = downsample_labels(labels)
    assert out.shape == (2, 2, 2)
    assert out[1, 1, 1] == 4 and out.sum() == 4
    labels[:] = 1
    labels[2, 2, 2] = 0
    # The corner block holds one real voxel, which is unlabeled; the padding
    # around it is unlabeled too, so it is as well.
    assert downsample_labels(labels)[1, 1, 1] == 0


def test_one_voxel_survives_every_level():
    for position in [(0, 0, 0), (13, 40, 63), (63, 63, 63)]:
        # A class in a sea of background, which fills most of every block.
        labels = np.ones(LABEL_CHUNK_ZYX, dtype=np.uint8)
        labels[position] = 9
        level = labels
        while level.shape != (1, 1, 1):
            level = downsample_labels(level)
            assert (level == 9).sum() == 1
        assert level[0, 0, 0] == 9


def test_a_one_voxel_thick_line_and_sheet_stay_whole():
    line = np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    line[21, 33, :] = 5
    sheet = np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    sheet[:, 17, :] = 6
    # Keeping every second voxel loses both.
    assert not line[::2, ::2, ::2].any() and not sheet[::2, ::2, ::2].any()
    for depth in range(1, 7):
        line, sheet = downsample_labels(line), downsample_labels(sheet)
        length = 64 >> depth
        assert (line == 5).sum() == length
        assert {tuple(v[:2]) for v in np.argwhere(line == 5)} == {
            (21 >> depth, 33 >> depth)
        }
        assert (sheet == 6).sum() == length * length
        assert {v[1] for v in np.argwhere(sheet == 6)} == {17 >> depth}


def test_classes_never_mix():
    rng = np.random.default_rng(3)
    labels = rng.choice([0, 1, 2, 3, 40, 200], size=(16, 16, 16)).astype(np.uint8)
    out = downsample_labels(labels)
    for index in np.ndindex(out.shape):
        window = labels[tuple(slice(2 * i, 2 * i + 2) for i in index)]
        assert out[index] in window


def test_it_does_not_depend_on_the_order_of_voxels():
    rng = np.random.default_rng(5)
    labels = rng.choice([0, 1, 2, 3, 4], size=(8, 8, 8), p=[0.3, 0.2, 0.2, 0.2, 0.1])
    labels = labels.astype(np.uint8)
    before = labels.copy()
    expected = downsample_labels(labels)
    np.testing.assert_array_equal(labels, before)
    np.testing.assert_array_equal(downsample_labels(labels), expected)
    np.testing.assert_array_equal(
        downsample_labels(np.asfortranarray(labels)), expected
    )
    np.testing.assert_array_equal(
        downsample_labels(labels[::-1, ::-1, ::-1]), expected[::-1, ::-1, ::-1]
    )
    np.testing.assert_array_equal(
        downsample_labels(labels.transpose(2, 0, 1)), expected.transpose(2, 0, 1)
    )


def test_it_refuses_what_is_not_a_label_block():
    with pytest.raises(ValueError):
        downsample_labels(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        downsample_labels(np.zeros((4, 4, 4), dtype=np.int32))
    with pytest.raises(ValueError):
        downsample_labels(np.zeros((4, 4, 4), dtype=np.uint8), (2, 2))
    with pytest.raises(ValueError):
        downsample_labels(np.zeros((4, 4, 4), dtype=np.uint8), (2, 0, 2))
    with pytest.raises(ValueError):
        downsample_labels(np.zeros((8, 8, 8), dtype=np.uint8), (8, 8, 8))


@settings(max_examples=60, deadline=None)
@given(
    st.data(),
    st.tuples(*[st.integers(1, 7)] * 3),
    st.tuples(*[st.sampled_from([1, 2, 3])] * 3),
    st.lists(st.integers(0, 255), min_size=1, max_size=4, unique=True),
)
def test_it_follows_the_rule_on_any_block(data, shape, step, classes):
    values = [0, 1, *classes]
    labels = data.draw(
        st.lists(
            st.sampled_from(values),
            min_size=int(np.prod(shape)),
            max_size=int(np.prod(shape)),
        )
    )
    labels = np.array(labels, dtype=np.uint8).reshape(shape)
    np.testing.assert_array_equal(
        downsample_labels(labels, step), reference(labels, step)
    )
