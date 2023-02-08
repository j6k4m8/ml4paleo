from abc import ABC
from typing import List, Tuple, Optional
import numpy as np

from ..combiners import Combiner, OverwriteCombiner


class AlignedStitcher(ABC):
    """
    A class to merge images.

    """

    def __init__(self, image: np.ndarray, combiner: Optional[Combiner] = None):
        """
        Create a new stitcher.

        Arguments:
            image (np.ndarray): The image to stitch.
        """
        self._image = image
        self._combiner = combiner or OverwriteCombiner()

    def stitch(
        self, others: List[np.ndarray], origins: List[tuple]
    ) -> "AlignedStitcher":
        """
        Stitch two images together.

        Arguments:
            other (np.ndarray): The image to stitch.
            origin (tuple[int]): The origin of the second image.

        Returns:
            np.ndarray: The stitched image.
        """
        # Create a new image with the correct size.
        (
            new_image,
            positions,
            my_new_position,
        ) = self.get_new_canvas_and_candidate_positions(others, origins)

        # Put me in the correct position:
        new_image[
            tuple(
                [
                    slice(my_new_position[i], my_new_position[i] + self._image.shape[i])
                    for i in range(len(my_new_position))
                ]
            )
        ] = self._image

        # Now put each image into the correct location:
        new_image = self._combiner.combine(new_image, others, positions)
        return AlignedStitcher(new_image)

    def get_new_canvas_and_candidate_positions(
        self, others: List[np.ndarray], origins: List[tuple]
    ) -> Tuple[np.ndarray, List[list], list]:
        """
        Get the new canvas and candidate positions.

        Arguments:
            others (List[np.ndarray]): The images to stitch.
            origin (tuple[int]): The origin of the second image.

        Returns:
            np.ndarray: The stitched image.
        """

        # If the origin is smaller than 0 in any dimension, then we need to
        # include the negative values in the size.
        # If the origin plus the size of other is larger than the size of the
        # current image, then we need to include the extra values in the size.

        if len(others) != len(origins) or len(others) > 1:
            raise NotImplementedError()

        other = others[0]
        origin = origins[0]

        # Get the size of the current image.
        current_image_size = self._image.shape

        # Get the size of the other image.
        other_image_size = other.shape

        # Get the minimum of the origin and the current image size.
        min_origin = np.min([origin, [0] * len(origin)], axis=0)

        # Get the maximum of the origin plus the other image size and the
        # current image size.
        max_endpt = np.max(
            [np.array(origin) + other_image_size, current_image_size], axis=0
        )

        # Create a new image with the correct size.
        new_image = np.zeros(
            [(max_endpt[i]) - min_origin[i] for i in range(len(min_origin))],
            dtype=self._image.dtype,
        )

        new_positions = [
            # Add the new origin back in:
            [-1 * min_origin[i] + origin[i] for i in range(len(min_origin))]
            for origin in origins
        ]

        my_new_position = [-1 * min_origin[i] for i in range(len(min_origin))]

        return new_image, new_positions, my_new_position
