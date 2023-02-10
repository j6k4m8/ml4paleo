import abc
import numpy as np


class Segmenter3D(abc.ABC):
    """
    An abstract base class for 3D segmentation algorithms.

    Does not handle stitching.

    """

    @abc.abstractmethod
    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment the given volume.

        Arguments:
            volume (np.ndarray<any>): The volume to segment.

        Returns:
            np.ndarray<u64>: The segmentation mask.

        """
        ...

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """
        Some way to save the model, given a path on disk.
        """
        ...

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """
        Some way to load the model, given a path on disk.
        """
        ...

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the given data.

        Arguments:
            X (np.ndarray<any>): The input data.
            y (np.ndarray<any>): The target data.

        """
        ...
