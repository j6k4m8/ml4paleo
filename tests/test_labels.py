"""
Label conventions, the label chunk codec, and label edits (deltas, undo, redo).
"""

import importlib.util
import pathlib

import numpy as np
import pytest
import zarr
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from ml4paleo.labels import (
    BACKGROUND,
    LABEL_CHUNK_ZYX,
    PLUGIN_IGNORE,
    Source,
    from_plugin_space,
    to_plugin_space,
)
from ml4paleo.labels.codec import (
    ZARR_CODECS,
    blob_key,
    content_hash,
    decode_chunk,
    encode_chunk,
)
from ml4paleo.labels.deltas import (
    ChunkDelta,
    apply_delta,
    pack_mask,
    revert_patch,
    split_into_deltas,
    unpack_mask,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "labels"


def _zeros():
    return np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)


def test_plugin_space_round_trip():
    labels = np.array([0, 1, 2, 5, 9, 0], dtype=np.uint8)
    complete = np.array([False, False, False, False, False, True])
    targets = to_plugin_space(labels, class_values=[2, 5], complete=complete)
    # unlabeled -> ignore, background -> 0, classes -> 1..K, deleted class 9 ->
    # ignore, unlabeled inside a complete ROI -> background.
    assert targets.tolist() == [PLUGIN_IGNORE, 0, 1, 2, PLUGIN_IGNORE, 0]
    restored = from_plugin_space(np.array([0, 1, 2], dtype=np.uint8), [2, 5])
    assert restored.tolist() == [BACKGROUND, 2, 5]
    with pytest.raises(ValueError):
        from_plugin_space(np.array([3], dtype=np.uint8), [2, 5])
    with pytest.raises(ValueError):
        to_plugin_space(labels, class_values=[1])


def test_codec_round_trips_and_skips_empty_chunks():
    assert content_hash(_zeros()) is None
    chunk = _zeros()
    chunk[3, 4, 5] = 7
    decoded = decode_chunk(encode_chunk(chunk))
    np.testing.assert_array_equal(decoded, chunk)
    np.testing.assert_array_equal(decode_chunk(None), _zeros())
    assert blob_key(content_hash(chunk)).startswith("blobs/")
    with pytest.raises(ValueError):
        blob_key("../../etc/passwd")


def test_codec_bytes_are_zarr_v3_chunks(tmp_path):
    chunk = np.random.default_rng(0).integers(0, 4, LABEL_CHUNK_ZYX, dtype=np.uint8)
    # Our encoded bytes, dropped into a zarr v3 array, read back correctly...
    array = zarr.create_array(
        store=str(tmp_path / "labels.zarr"),
        shape=(64, 128, 64),
        chunks=LABEL_CHUNK_ZYX,
        dtype="uint8",
        compressors=[ZARR_CODECS[1]],
        fill_value=0,
    )
    (tmp_path / "labels.zarr" / "c" / "0" / "1").mkdir(parents=True)
    (tmp_path / "labels.zarr" / "c" / "0" / "1" / "0").write_bytes(encode_chunk(chunk))
    np.testing.assert_array_equal(array[:, 64:, :], chunk)
    # ...and chunks zarr writes decode with our codec.
    array[:, :64, :] = chunk[::-1]
    raw = (tmp_path / "labels.zarr" / "c" / "0" / "0" / "0").read_bytes()
    np.testing.assert_array_equal(decode_chunk(raw), chunk[::-1])


@given(st.lists(st.booleans(), min_size=1, max_size=300))
def test_masks_round_trip(bits):
    mask = np.array(bits, dtype=bool).reshape(1, 1, len(bits))
    np.testing.assert_array_equal(unpack_mask(pack_mask(mask), mask.shape), mask)


@settings(max_examples=40, deadline=None)
@given(
    origin=st.tuples(*(st.integers(0, 70) for _ in range(3))),
    shape=st.tuples(*(st.integers(1, 40) for _ in range(3))),
    seed=st.integers(0, 2**16),
)
def test_split_deltas_reassemble_the_mask(origin, shape, seed):
    mask = np.random.default_rng(seed).random(shape) < 0.3
    volume = np.zeros((128, 128, 128), dtype=np.uint8)
    expected = volume.copy()
    region = tuple(slice(o, o + s) for o, s in zip(origin, shape, strict=True))
    expected[region][mask] = 6

    for delta in split_into_deltas(mask, origin, value=6):
        chunk_slices = tuple(
            slice(k * c, (k + 1) * c)
            for k, c in zip(delta.key, LABEL_CHUNK_ZYX, strict=True)
        )
        chunk = volume[chunk_slices]
        applied = apply_delta(chunk, _zeros(), delta, Source.HUMAN)
        volume[chunk_slices] = applied.class_chunk
    np.testing.assert_array_equal(volume, expected)


def _delta(value=2, only_if="any", box=(0, 0, 0, 4, 4, 4)):
    shape = (box[3] - box[0], box[4] - box[1], box[5] - box[2])
    return ChunkDelta(
        key=(0, 0, 0),
        box=box,
        mask=pack_mask(np.ones(shape, dtype=bool)),
        value=value,
        only_if=only_if,
    )


def test_only_if_limits_which_voxels_change():
    chunk = _zeros()
    chunk[0, 0, :2] = 3  # two voxels of class 3
    unlabeled_only = apply_delta(
        chunk, _zeros(), _delta(only_if="unlabeled"), Source.HUMAN
    )
    assert unlabeled_only.class_chunk[0, 0, :4].tolist() == [3, 3, 2, 2]
    class_only = apply_delta(
        chunk, _zeros(), _delta(value=0, only_if="class:3"), Source.HUMAN
    )
    assert class_only.changed == 2
    assert class_only.class_chunk.max() == 0


def test_source_tracks_writes_and_erases():
    applied = apply_delta(_zeros(), _zeros(), _delta(value=2), Source.INTERACTIVE)
    assert (applied.source_chunk[:4, :4, :4] == Source.INTERACTIVE).all()
    erased = apply_delta(
        applied.class_chunk, applied.source_chunk, _delta(value=0), Source.HUMAN
    )
    assert erased.source_chunk.max() == Source.NONE


def test_undo_keeps_a_collaborators_later_edits_and_redo_restores():
    base = _zeros()
    base[0, 0, 0] = 5
    mine = apply_delta(base, _zeros(), _delta(value=2), Source.HUMAN)
    # A collaborator then repaints part of my stroke.
    theirs = apply_delta(
        mine.class_chunk,
        mine.source_chunk,
        _delta(value=3, box=(0, 0, 0, 1, 1, 4)),
        Source.HUMAN,
    )
    undone = revert_patch(theirs.class_chunk, theirs.source_chunk, mine.undo)
    assert undone.class_chunk[0, 0, :4].tolist() == [3, 3, 3, 3]  # theirs kept
    assert undone.class_chunk[1, 1, 1] == 0  # mine reverted
    assert undone.changed == mine.changed - 4

    redone = revert_patch(undone.class_chunk, undone.source_chunk, undone.undo)
    np.testing.assert_array_equal(redone.class_chunk, theirs.class_chunk)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"box": (0, 0, 0, 65, 1, 1)},
        {"box": (3, 0, 0, 3, 1, 1)},
        {"value": 255},
        {"only_if": "class:999"},
        {"only_if": "everything"},
        {"value": None},
    ],
)
def test_invalid_deltas_are_rejected(kwargs):
    fields = {"key": (0, 0, 0), "box": (0, 0, 0, 1, 1, 1), "mask": b"", "value": 2}
    fields.update(kwargs)
    with pytest.raises(ValidationError):
        ChunkDelta(**fields)


def test_golden_fixtures_match_the_implementation():
    spec = importlib.util.spec_from_file_location("generate", FIXTURES / "generate.py")
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    assert (FIXTURES / "cases.json").read_text() == generate.render(), (
        "Label semantics changed: regenerate tests/fixtures/labels/cases.json "
        "and update the TypeScript implementation to match."
    )
