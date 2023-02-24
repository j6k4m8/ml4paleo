from typing import Tuple, Union
import pathlib
import numpy as np
import zarr

from .volume_provider import VolumeProvider


class ZarrVolumeProvider(VolumeProvider):
    """
    A volume provider that provides a 3D volume of data from a zarr array.
    """

    def __init__(self, file_path: Union[str, pathlib.Path]):
        """
        Create a new ZarrVolumeProvider.

        Arguments:
            file_path: The path to the zarr file.

        """
        self.zarr = zarr.open(str(file_path), mode="r")

    def __getitem__(self, key):
        return self.zarr[key]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.zarr.shape

    @property
    def dtype(self) -> np.dtype:
        return self.zarr.dtype
