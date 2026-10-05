"""
Generate the golden label-edit fixtures shared by the Python and TypeScript
implementations.

Run `uv run python tests/fixtures/labels/generate.py` after changing label
semantics, and commit the updated `cases.json`. Masks and values are stored as
raw packed bits and raw bytes (base64), not zstd output, because different
zstd implementations may produce different (equally valid) compressed bytes.

Three kinds of cases:

- `apply`: one delta applied to a base chunk; the resulting chunk hashes and
  the claim it records.
- `split`: a global mask split into per-chunk deltas.
- `history`: a sequence of ops and undos/redos on one chunk; the final chunk
  must equal the overlay of the live claims.
"""

import base64
import json
import pathlib

import numpy as np

from ml4paleo.labels import LABEL_CHUNK_ZYX, Source
from ml4paleo.labels.codec import content_hash
from ml4paleo.labels.deltas import (
    ChunkDelta,
    apply_delta,
    pack_mask,
    pack_values,
    recompute,
    split_into_deltas,
    unpack_mask,
    unpack_values,
)

OUTPUT = pathlib.Path(__file__).with_name("cases.json")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _bits(mask: np.ndarray) -> str:
    return _b64(np.packbits(mask.astype(bool).ravel(), bitorder="little").tobytes())


def _ball(shape, center, radius) -> np.ndarray:
    grid = np.indices(shape)
    distance = sum((g - c) ** 2 for g, c in zip(grid, center, strict=True))
    return distance <= radius**2


def _base_chunk(name: str) -> tuple[np.ndarray, np.ndarray]:
    chunk = np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    if name == "quadrants":
        chunk[:32, :32] = 1
        chunk[32:, 32:] = 3
    source = np.where(chunk > 0, Source.IMPORTED, Source.NONE).astype(np.uint8)
    return chunk, source


def _delta(box, mask, value=None, values=None, only_if="any") -> ChunkDelta:
    return ChunkDelta(
        key=(0, 0, 0),
        box=box,
        mask=pack_mask(mask),
        value=value,
        values=pack_values(values) if values is not None else None,
        only_if=only_if,
    )


def _apply_cases() -> list[dict]:
    cases = []
    specs = [
        ("paint_empty", "zeros", 2, None, "any"),
        ("paint_over_classes", "quadrants", 2, None, "any"),
        ("paint_only_unlabeled", "quadrants", 2, None, "unlabeled"),
        ("erase_class_3_only", "quadrants", 0, None, "class:3"),
        ("multiclass_values", "quadrants", None, "gradient", "any"),
    ]
    for name, base, value, values_kind, only_if in specs:
        box = (16, 20, 24, 48, 44, 56)
        shape = (box[3] - box[0], box[4] - box[1], box[5] - box[2])
        mask = _ball(shape, (16, 12, 16), 13)
        values = None
        if values_kind == "gradient":
            values = (2 + (np.indices(shape)[2] % 3)).astype(np.uint8)
        class_chunk, source_chunk = _base_chunk(base)
        applied = apply_delta(
            class_chunk,
            source_chunk,
            _delta(box, mask, value, values, only_if),
            Source.HUMAN,
        )
        cases.append(
            {
                "name": name,
                "base": base,
                "box": list(box),
                "mask_bits": _bits(mask),
                "value": value,
                "values": _b64(values.tobytes()) if values is not None else None,
                "only_if": only_if,
                "source": int(Source.HUMAN),
                "expected": {
                    "class_sha256": content_hash(applied.class_chunk),
                    "source_sha256": content_hash(applied.source_chunk),
                    "changed": applied.changed,
                    "claim_mask_bits": _bits(
                        unpack_mask(applied.claim.mask, applied.claim.box_shape)
                    ),
                    "claim_values": _b64(
                        unpack_values(
                            applied.claim.values, applied.claim.box_shape
                        ).tobytes()
                    ),
                },
            }
        )
    return cases


def _split_cases() -> list[dict]:
    cases = []
    # A brush stroke (a slab of a ball) that crosses chunk boundaries in y and x.
    for name, origin, shape, center, radius in [
        ("ball_across_corner", (60, 58, 50), (9, 13, 29), (4, 6, 14), 7),
        ("thin_plane_stroke", (10, 0, 0), (1, 70, 130), (0, 35, 65), 40),
    ]:
        mask = _ball(shape, center, radius)
        deltas = split_into_deltas(mask, origin, value=4)
        cases.append(
            {
                "name": name,
                "origin": list(origin),
                "shape": list(shape),
                "mask_bits": _bits(mask),
                "value": 4,
                "expected_deltas": [
                    {
                        "key": list(d.key),
                        "box": list(d.box),
                        "mask_bits": _bits(unpack_mask(d.mask, d.box_shape)),
                    }
                    for d in deltas
                ],
            }
        )
    return cases


def _history_cases() -> list[dict]:
    """
    Each step is either {"op": ...} (apply a delta, giving it the next op
    number) or {"undo": n} / {"redo": n} (toggle op n).
    """
    full = (0, 0, 0, 8, 8, 8)
    left = (0, 0, 0, 8, 8, 4)
    histories = {
        # B repaints A's voxels with the same class; undoing A keeps B's paint.
        "same_value_overwrite": [
            {"op": {"box": left, "value": 2}},
            {"op": {"box": full, "value": 2}},
            {"undo": 0},
        ],
        # Undoing two overlapping ops in either order leaves nothing behind.
        "out_of_order_undo": [
            {"op": {"box": full, "value": 2}},
            {"op": {"box": left, "value": 3}},
            {"undo": 0},
            {"undo": 1},
        ],
        # Redo brings an op back underneath a later op.
        "redo_under_later_op": [
            {"op": {"box": full, "value": 2}},
            {"op": {"box": left, "value": 3}},
            {"undo": 0},
            {"redo": 0},
        ],
        # A fill of unlabeled voxels keeps its original claim after undo.
        "only_unlabeled_then_undo_under": [
            {"op": {"box": left, "value": 2}},
            {"op": {"box": full, "value": 4, "only_if": "unlabeled"}},
            {"undo": 0},
        ],
    }
    cases = []
    for name, steps in histories.items():
        class_chunk, source_chunk = _base_chunk("zeros")
        claims, live = [], []
        for step in steps:
            if "op" in step:
                spec = step["op"]
                box = spec["box"]
                shape = (box[3] - box[0], box[4] - box[1], box[5] - box[2])
                applied = apply_delta(
                    class_chunk,
                    source_chunk,
                    _delta(
                        box,
                        np.ones(shape, dtype=bool),
                        spec["value"],
                        only_if=spec.get("only_if", "any"),
                    ),
                    Source.HUMAN,
                )
                class_chunk, source_chunk = applied.class_chunk, applied.source_chunk
                claims.append(applied.claim)
                live.append(True)
            else:
                index = step.get("undo", step.get("redo"))
                live[index] = "redo" in step
                result = recompute(
                    class_chunk,
                    source_chunk,
                    claims[index],
                    [c for c, alive in zip(claims, live, strict=True) if alive],
                )
                class_chunk, source_chunk = result.class_chunk, result.source_chunk
        cases.append(
            {
                "name": name,
                "steps": [
                    {
                        **{k: v for k, v in step.items() if k != "op"},
                        **(
                            {"op": {**step["op"], "box": list(step["op"]["box"])}}
                            if "op" in step
                            else {}
                        ),
                    }
                    for step in steps
                ],
                "expected": {
                    "class_sha256": content_hash(class_chunk),
                    "source_sha256": content_hash(source_chunk),
                },
            }
        )
    return cases


def generate() -> dict:
    return {
        "chunk_shape_zyx": list(LABEL_CHUNK_ZYX),
        "bases": {
            "zeros": "all voxels 0",
            "quadrants": "[:32, :32] = 1 and [32:, 32:] = 3; source is IMPORTED (5) "
            "wherever class > 0",
        },
        "apply": _apply_cases(),
        "split": _split_cases(),
        "history": _history_cases(),
    }


def render() -> str:
    return json.dumps(generate(), indent=1, sort_keys=True) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(render())
    print(f"Wrote {OUTPUT}")
