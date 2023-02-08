from abc import ABC

import numpy as np

from skimage.segmentation import join_segmentations


class Combiner(ABC):
    def combine(self, canvas: np.ndarray, images: list, positions: list) -> np.ndarray:
        """
        Combine two images together onto a canvas.
        """
        ...


class OverwriteCombiner(Combiner):
    """
    An image combiner that overwrites data if nonzero voxels overlap.

    RIGHT overwrites LEFT overwrites CANVAS.
    """

    def combine(self, canvas: np.ndarray, images: list, positions: list) -> np.ndarray:
        """
        Combine two images together onto a canvas.
        """
        # Now put each image into the correct location:
        for img, pos in zip(images, positions):
            slices = tuple(
                [slice(pos[i], pos[i] + img.shape[i]) for i in range(len(pos))]
            )
            canvas[slices] = img
        return canvas


class MeanValueCombiner(Combiner):
    """
    An image combiner that averages data if nonzero voxels overlap.

    RIGHT overwrites LEFT overwrites CANVAS.
    """

    def combine(self, canvas: np.ndarray, images: list, positions: list) -> np.ndarray:
        """
        Combine two images together onto a canvas.
        """
        # Now put each image into the correct location:
        for img, pos in zip(images, positions):
            slices = tuple(
                [slice(pos[i], pos[i] + img.shape[i]) for i in range(len(pos))]
            )
            canvas[slices] += img
        return canvas / len(images)


class MaxValueCombiner(Combiner):
    """
    An image combiner that averages data if nonzero voxels overlap.

    RIGHT overwrites LEFT overwrites CANVAS.
    """

    def combine(self, canvas: np.ndarray, images: list, positions: list) -> np.ndarray:
        """
        Combine two images together onto a canvas.
        """
        # Now put each image into the correct location:
        for img, pos in zip(images, positions):
            slices = tuple(
                [slice(pos[i], pos[i] + img.shape[i]) for i in range(len(pos))]
            )
            canvas[slices] = np.maximum(canvas[slices], img)
        return canvas


class SegmentationBooleanCombiner(Combiner):
    """
    Best used on segmentation or blobs.

    This combiner looks at each segment in overlapping regions and chooses
    to overwrite the images in `images` with the ID from `canvas` so that there
    is one contiguous segment ID.

    """

    def combine(self, canvas: np.ndarray, images: list, positions: list) -> np.ndarray:
        """
        Combine two images together onto a canvas.
        """
        # Now put each image into the correct location.
        # Create a set of layers for each image:
        canvas = canvas[:]
        for img, pos in zip(images, positions):
            layer = np.zeros_like(canvas)
            slices = tuple(
                [slice(pos[i], pos[i] + img.shape[i]) for i in range(len(pos))]
            )
            layer[slices] = img
            canvas = join_segmentations(canvas, layer)
        return canvas
