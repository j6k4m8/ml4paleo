from typing import Tuple
import numpy as np
from .volume_provider import VolumeProvider


class NumpyVolumeProvider(VolumeProvider):
    """
    A VolumeProvider that provides a 3D volume of data from a numpy array.
    """

    def __init__(self, data: np.ndarray):
        """
        Create a new NumpyVolumeProvider.

        Arguments:
            data: The 3D numpy array to provide.

        Raises:
            ValueError: If the data is not 3D.

        """
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D, but has shape {data.shape}.")
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
