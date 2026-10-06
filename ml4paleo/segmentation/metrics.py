"""
Scores for predicted segmentations, per class, ignoring voxels without a
target.
"""

from typing import Any

import numpy as np

from ml4paleo.labels import PLUGIN_IGNORE


class ScoreSheet:
    """
    Running per-class confusion counts over many crops.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.tp = np.zeros(num_classes, dtype=np.int64)
        self.fp = np.zeros(num_classes, dtype=np.int64)
        self.fn = np.zeros(num_classes, dtype=np.int64)
        self.voxels = 0
        self.correct = 0

    def add(self, predicted: np.ndarray, targets: np.ndarray) -> None:
        known = targets != PLUGIN_IGNORE
        p = predicted[known].astype(np.int64)
        t = targets[known].astype(np.int64)
        self.voxels += int(t.size)
        self.correct += int((p == t).sum())
        for k in range(self.num_classes):
            pk, tk = p == k, t == k
            self.tp[k] += int((pk & tk).sum())
            self.fp[k] += int((pk & ~tk).sum())
            self.fn[k] += int((~pk & tk).sum())

    def summary(self, class_values: list[int]) -> dict[str, Any]:
        """
        Dice and IoU per class (keyed by class value) and their mean over the
        classes that appear, plus voxel accuracy.
        """
        per_class: dict[str, dict[str, float]] = {}
        for k, value in enumerate(class_values, start=1):
            tp, fp, fn = int(self.tp[k]), int(self.fp[k]), int(self.fn[k])
            if tp + fn == 0:
                continue
            per_class[str(value)] = {
                "dice": 2 * tp / (2 * tp + fp + fn),
                "iou": tp / (tp + fp + fn),
                "voxels": tp + fn,
            }
        scores = list(per_class.values())
        return {
            "voxels": self.voxels,
            "accuracy": self.correct / self.voxels if self.voxels else None,
            "mean_dice": float(np.mean([s["dice"] for s in scores]))
            if scores
            else None,
            "mean_iou": float(np.mean([s["iou"] for s in scores])) if scores else None,
            "classes": per_class,
        }
