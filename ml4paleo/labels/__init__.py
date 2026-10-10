"""
Label values, label chunk storage, and label edits.

Label arrays are uint8 and share the image's (z, y, x) grid. Two arrays are
stored per label layer, both in 64³ chunks:

- `class`: what each voxel is labeled.
- `source`: who or what produced that label (see `Source`), so training can
  tell hand-drawn labels from model proposals without joining the op log.

Value conventions for `class`:

- 0 (`UNLABELED`): nobody has labeled the voxel. Training ignores it, unless
  the voxel is inside a training ROI marked complete, where it counts as
  background. Missing chunks are all unlabeled.
- 1 (`BACKGROUND`): explicitly labeled as background.
- 2-254: the project's classes. Values are never reused after a class is
  deleted, so old labels and models keep their meaning.
- 255 (`DECLINED`): a model suggestion explicitly declined by a person. It
  suppresses that suggestion, is ignored by training, and never appears as a
  segmentation class.

Segmentation plugins see a compact space instead: 0 is background, 1..K are
the project's classes in order, and 255 (`PLUGIN_IGNORE`) marks voxels to
leave out of the loss. `to_plugin_space` and `from_plugin_space` convert.
"""

from collections.abc import Sequence
from enum import IntEnum

import numpy as np

UNLABELED = 0
BACKGROUND = 1
FIRST_CLASS = 2
MAX_CLASS = 254
RESERVED = 255
# Kept as an alias because older callers describe 255 only as reserved.
DECLINED = RESERVED

PLUGIN_BACKGROUND = 0
PLUGIN_IGNORE = 255

LABEL_CHUNK_ZYX = (64, 64, 64)


class Source(IntEnum):
    """
    Where a voxel's current label came from, stored in the `source` array.
    """

    NONE = 0
    HUMAN = 1
    MODEL_VERIFIED = 2
    INTERACTIVE = 3
    PROPAGATED = 4
    IMPORTED = 5
    DECLINED = 6


def check_class_values(class_values: Sequence[int]) -> None:
    """
    Raise ValueError unless `class_values` are distinct project class values.
    """
    if len(set(class_values)) != len(class_values):
        raise ValueError(f"Class values must be distinct: {list(class_values)}")
    for value in class_values:
        if not FIRST_CLASS <= value <= MAX_CLASS:
            raise ValueError(
                f"Class value {value} is outside {FIRST_CLASS}..{MAX_CLASS}"
            )
    if len(class_values) >= PLUGIN_IGNORE:
        raise ValueError("Too many classes for the plugin label space")


def to_plugin_space(
    labels: np.ndarray,
    class_values: Sequence[int],
    complete: np.ndarray | bool = False,
) -> np.ndarray:
    """
    Convert stored labels into a plugin's training targets.

    `complete` marks voxels inside training ROIs marked complete (a boolean
    array shaped like `labels`, or one bool for the whole array). Unlabeled
    voxels there become background; elsewhere they become `PLUGIN_IGNORE`.
    Voxels with values that are not in `class_values` (for example a class
    that was deleted) are ignored.
    """
    check_class_values(class_values)
    lookup = np.full(256, PLUGIN_IGNORE, dtype=np.uint8)
    lookup[BACKGROUND] = PLUGIN_BACKGROUND
    for index, value in enumerate(class_values, start=1):
        lookup[value] = index
    targets = lookup[labels]
    unlabeled_and_complete = (labels == UNLABELED) & np.asarray(complete, dtype=bool)
    targets[unlabeled_and_complete] = PLUGIN_BACKGROUND
    return targets


def from_plugin_space(
    predictions: np.ndarray, class_values: Sequence[int]
) -> np.ndarray:
    """
    Convert a plugin's predicted class indices back into stored label values.

    Plugin background becomes `BACKGROUND`, so a prediction never reads as
    "unlabeled".
    """
    check_class_values(class_values)
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[PLUGIN_BACKGROUND] = BACKGROUND
    for index, value in enumerate(class_values, start=1):
        lookup[index] = value
    if predictions.max(initial=0) > len(class_values):
        raise ValueError("Prediction contains a class index the project does not have")
    return lookup[predictions]
