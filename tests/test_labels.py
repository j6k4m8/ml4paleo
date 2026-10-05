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
    recompute,
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


def _apply_all(ops):
    """
    Apply (box, value, only_if) ops to an empty chunk; return the chunks and
    the claims.
    """
    class_chunk, source_chunk, claims = _zeros(), _zeros(), []
    for box, value, only_if in ops:
        applied = apply_delta(
            class_chunk, source_chunk, _delta(value, only_if, box), Source.HUMAN
        )
        class_chunk, source_chunk = applied.class_chunk, applied.source_chunk
        claims.append(applied.claim)
    return class_chunk, source_chunk, claims


def _overlay(claims):
    """
    The state a chunk must have: the overlay of live claims on an empty chunk.
    """
    full = ChunkDelta(
        key=(0, 0, 0),
        box=(0, 0, 0, *LABEL_CHUNK_ZYX),
        mask=pack_mask(np.ones(LABEL_CHUNK_ZYX, dtype=bool)),
        value=0,
    )
    region = apply_delta(_zeros(), _zeros(), full, Source.HUMAN).claim
    return recompute(_zeros(), _zeros(), region, claims)


def test_undo_never_disturbs_a_later_same_value_overwrite():
    left, full = (0, 0, 0, 4, 4, 2), (0, 0, 0, 4, 4, 4)
    class_chunk, source_chunk, (mine, theirs) = _apply_all(
        [(left, 2, "any"), (full, 2, "any")]
    )
    undone = recompute(class_chunk, source_chunk, mine, [theirs])
    np.testing.assert_array_equal(undone.class_chunk, class_chunk)
    assert undone.changed == 0


def test_undoing_overlapping_ops_in_any_order_clears_them():
    full, left = (0, 0, 0, 4, 4, 4), (0, 0, 0, 4, 4, 2)
    class_chunk, source_chunk, (first, second) = _apply_all(
        [(full, 2, "any"), (left, 3, "any")]
    )
    step = recompute(class_chunk, source_chunk, first, [second])
    assert step.class_chunk[0, 0, :4].tolist() == [3, 3, 0, 0]
    step = recompute(step.class_chunk, step.source_chunk, second, [])
    assert step.class_chunk.max() == 0
    assert step.source_chunk.max() == Source.NONE
    # Redo the first op: it comes back everywhere it claimed.
    redone = recompute(step.class_chunk, step.source_chunk, first, [first])
    assert redone.class_chunk[:4, :4, :4].min() == 2


@settings(max_examples=60, deadline=None)
@given(
    ops=st.lists(
        st.tuples(
            st.tuples(
                st.integers(0, 6),
                st.integers(0, 6),
                st.integers(0, 6),
                st.integers(1, 6),
                st.integers(1, 6),
                st.integers(1, 6),
            ),
            st.integers(0, 4),
            st.sampled_from(["any", "unlabeled", "class:2"]),
        ),
        min_size=1,
        max_size=8,
    ),
    toggles=st.lists(st.integers(0, 7), max_size=10),
)
def test_state_always_equals_the_overlay_of_live_claims(ops, toggles):
    ops = [
        ((z, y, x, z + dz, y + dy, x + dx), value, only_if)
        for (z, y, x, dz, dy, dx), value, only_if in ops
    ]
    class_chunk, source_chunk, claims = _apply_all(ops)
    live = [True] * len(claims)
    for toggle in toggles:
        index = toggle % len(claims)
        live[index] = not live[index]
        result = recompute(
            class_chunk,
            source_chunk,
            claims[index],
            [c for c, alive in zip(claims, live, strict=True) if alive],
        )
        class_chunk, source_chunk = result.class_chunk, result.source_chunk
    expected = _overlay([c for c, alive in zip(claims, live, strict=True) if alive])
    np.testing.assert_array_equal(class_chunk, expected.class_chunk)
    np.testing.assert_array_equal(source_chunk, expected.source_chunk)


def test_repainting_a_model_label_marks_it_as_human():
    proposed = apply_delta(_zeros(), _zeros(), _delta(value=3), Source.MODEL_VERIFIED)
    repainted = apply_delta(
        proposed.class_chunk, proposed.source_chunk, _delta(value=3), Source.HUMAN
    )
    assert repainted.changed == 64
    assert (repainted.source_chunk[:4, :4, :4] == Source.HUMAN).all()


def test_compressed_payloads_cannot_expand_past_their_box():
    import zstandard

    bomb = zstandard.ZstdCompressor(level=19).compress(b"\0" * (1 << 28))
    assert len(bomb) < 70_000
    delta = ChunkDelta(key=(0, 0, 0), box=(0, 0, 0, 1, 1, 1), mask=bomb, value=2)
    with pytest.raises(ValueError):
        apply_delta(_zeros(), _zeros(), delta, Source.HUMAN)
    # Frames that don't declare their size are bounded too.
    stream = zstandard.ZstdCompressor(write_content_size=False)
    with pytest.raises(ValueError):
        unpack_mask(stream.compress(b"\xff" * 100), (1, 1, 8))
    # A correctly sized frame without a declared size is fine.
    assert unpack_mask(stream.compress(b"\xff"), (1, 1, 8)).all()


def test_split_rejects_out_of_range_values():
    mask = np.ones((1, 1, 2), dtype=bool)
    with pytest.raises(ValueError):
        split_into_deltas(mask, (0, 0, 0), value=300)
    with pytest.raises(ValueError):
        split_into_deltas(mask, (0, 0, 0), values=np.array([[[2, 300]]]))


def test_deltas_can_be_checked_against_the_volume_shape():
    edge = _delta(value=2, box=(0, 0, 0, 4, 4, 4))
    edge.check_within((4, 4, 4))
    with pytest.raises(ValueError):
        edge.check_within((4, 4, 3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"box": (0, 0, 0, 65, 1, 1)},
        {"box": (3, 0, 0, 3, 1, 1)},
        {"value": 255},
        {"only_if": "class:999"},
        {"only_if": "everything"},
        {"value": None},
        {"mask": b"x" * 300_000},
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


def test_declared_huge_frames_never_expand():
    import resource

    import zstandard

    from ml4paleo.labels.codec import decompress_exact

    bomb = zstandard.ZstdCompressor(level=19).compress(b"\0" * (512 << 20))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for size in (32768, len(bomb) - 100):
        with pytest.raises(ValueError):
            decompress_exact(bomb, size)
    with pytest.raises(ValueError):
        decode_chunk(bomb)
    growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    # ru_maxrss is bytes on macOS and kilobytes on Linux; either way, far
    # below the 512 MB the frame declares.
    assert growth < 64 << 20


def test_trailing_frames_cannot_hide_extra_data():
    import zstandard

    from ml4paleo.labels.codec import decompress_exact

    frame = zstandard.ZstdCompressor(write_content_size=False).compress(b"ab")
    assert decompress_exact(frame, 2) == b"ab"
    with pytest.raises(ValueError):
        decompress_exact(frame + frame, 2)
