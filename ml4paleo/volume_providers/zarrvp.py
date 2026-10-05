import pathlib

import numpy as np
import zarr

from ..storage import StorageGrant, zarr_store
from .volume_provider import VolumeProvider


class ZarrVolumeProvider(VolumeProvider):
    """
    A volume provider that provides a 3D volume of data from a zarr array.

    Reads both zarr v2 arrays (written by ml4paleo v1) and zarr v3 arrays.
    """

    def __init__(self, location: str | pathlib.Path | StorageGrant):
        """
        Create a new ZarrVolumeProvider.

        Arguments:
            location: A local path to the zarr array, or a `StorageGrant` for
                an array on local disk, S3, or GCS.

        """
        if isinstance(location, StorageGrant):
            store = zarr_store(location.model_copy(update={"access": "r"}))
            self.zarr = zarr.open_array(store=store, mode="r")
        else:
            self.zarr = zarr.open_array(str(location), mode="r")

    def __getitem__(self, key):
        return self.zarr[key]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.zarr.shape

    @property
    def dtype(self) -> np.dtype:
        return self.zarr.dtype

    @property
    def voxel_size_xyz_mm(self) -> tuple[float, float, float] | None:
        voxel_size = self.zarr.attrs.get("voxel_size_xyz_mm")
        if voxel_size is None:
            return None
        return tuple(float(size) for size in voxel_size)
