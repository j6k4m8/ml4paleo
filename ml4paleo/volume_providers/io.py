import math
import pathlib
from typing import Tuple, Union
import zarr
import numpy as np
import tqdm
from numcodecs import Blosc


def export_zarr_array(
    volume_provider,
    zarr_file: Union[pathlib.Path, str],
    downsample_factor: Tuple[int, int, int] = None,
    dtype: np.dtype = None,
    compression=None,
    chunk_size: Tuple[int, int, int] = None,
    slice_count: int = None,
    progress: bool = False,
    **kwargs,
):
    """
    Export the volume to a zarr array.

    Arguments:
        volume_provider: The volume provider to export.
        zarr_file: The path to the zarr file to write to.
        downsample_factor: The factor to downsample the volume by.
        dtype: The numpy dtype to use for the zarr array.
        compression: The compression to use for the zarr array.
        chunk_size: The chunk size to use for the zarr array.
        slice_count: The number of slices to write at a time.
        progress: Whether to show a progress bar.

    """
    if downsample_factor is None:
        downsample_factor = (1, 1, 1)
    if dtype is None:
        dtype = volume_provider.dtype
    if compression is None:
        compression = Blosc()
    if chunk_size is None:
        chunk_size = (256, 256, 256)

    # Open the zarr file
    zarr_file = pathlib.Path(zarr_file)
    zarr_file.parent.mkdir(parents=True, exist_ok=True)

    shape = [
        math.ceil(volume_provider.shape[0] / downsample_factor[0]),
        math.ceil(volume_provider.shape[1] / downsample_factor[1]),
        math.ceil(volume_provider.shape[2] / downsample_factor[2]),
    ]
    zarr_array = zarr.open(
        str(zarr_file),
        mode="w",
        shape=shape,
        chunks=chunk_size,
        dtype=dtype,
        compressor=compression,
        **kwargs,
    )

    # Write the data
    if slice_count is None:
        slice_count = 8

    prog_bar = tqdm.tqdm if progress else lambda x: x
    for i in prog_bar(range(0, volume_provider.shape[2], slice_count)):
        zstart = i
        zend = min(i + slice_count, volume_provider.shape[2])
        vol = volume_provider[:, :, zstart:zend]
        vol = vol.astype(dtype)[
            :: downsample_factor[0], :: downsample_factor[1], :: downsample_factor[2]
        ]
        zarr_array[
            :, :, (zstart // downsample_factor[2]) : (zend // downsample_factor[2])
        ] = vol

    return zarr_array
