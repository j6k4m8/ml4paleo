import math
import pathlib
from typing import Tuple, Union
import zarr
import numpy as np
import tqdm
from joblib import Parallel, delayed
from numcodecs import Blosc
from PIL import Image

from . import VolumeProvider


def export_zarr_array(
    volume_provider: VolumeProvider,
    zarr_file: Union[pathlib.Path, str],
    downsample_factor: Tuple[int, int, int] = None,
    dtype: np.dtype = None,
    compression=None,
    chunk_size: Tuple[int, int, int] = None,
    slice_count: int = None,
    progress: bool = False,
    parallel_jobs: int = False,
    cuboid_transform_fn=None,
    progress_callback=None,
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
        parallel_jobs: The number of parallel jobs to use. If False, will not
            use parallel jobs.
        cuboid_transform_fn: A function to apply to each cuboid before
            writing it to the zarr array. (Happens before dtype casting.)

    """
    if downsample_factor is None:
        downsample_factor = (1, 1, 1)
    if dtype is None:
        dtype = volume_provider.dtype
    if compression is None:
        compression = Blosc()
    if chunk_size is None:
        chunk_size = (256, 256, 256)

    cuboid_transform_fn = cuboid_transform_fn or (lambda x: x)

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
        zarr_format=2,
        shape=shape,
        chunks=chunk_size,
        dtype=dtype,
        compressor=compression,
        **kwargs,
    )

    # Write the data
    if slice_count is None:
        if parallel_jobs:
            slice_count = chunk_size[-1]
        else:
            slice_count = 8

    prog_bar = tqdm.tqdm if progress else lambda x: x

    if progress_callback is not None:
        # Send the callback the current progress out of the total.
        def _prog(x):
            for i, y in prog_bar(enumerate(x)):
                progress_callback(i, y, len(x))
                yield y

    else:
        _prog = prog_bar  # type: ignore

    if parallel_jobs is False:
        for i in _prog(range(0, volume_provider.shape[2], slice_count)):
            zstart = i
            zend = min(i + slice_count, volume_provider.shape[2])
            vol = volume_provider[:, :, zstart:zend]
            vol = cuboid_transform_fn(vol)
            vol = vol.astype(dtype)[
                :: downsample_factor[0],
                :: downsample_factor[1],
                :: downsample_factor[2],
            ]
            zarr_array[
                :, :, (zstart // downsample_factor[2]) : (zend // downsample_factor[2])
            ] = vol

    else:
        if slice_count < chunk_size[2]:
            # Warn if number of slices is less than chunk size
            # TODO
            pass

        def _racey_export_chunk_parallel(zstart):
            zend = min(zstart + slice_count, volume_provider.shape[2])
            vol = volume_provider[:, :, zstart:zend]
            vol = cuboid_transform_fn(vol)
            vol = vol.astype(dtype)[
                :: downsample_factor[0],
                :: downsample_factor[1],
                :: downsample_factor[2],
            ]
            zarr_array[
                :, :, (zstart // downsample_factor[2]) : (zend // downsample_factor[2])
            ] = vol

        _ = Parallel(n_jobs=parallel_jobs)(
            delayed(_racey_export_chunk_parallel)(zstart)
            for zstart in _prog(range(0, volume_provider.shape[2], slice_count))
        )

    return zarr_array


def get_random_tile(
    volume_provider,
    tile_size: Tuple[int, int],
) -> np.ndarray:
    """
    Get a random tile from the volume.

    Arguments:
        volume_provider: The volume provider to get the tile from.
        tile_size: The size of the tile to get.

    Returns:
        np.ndarray: A random tile from the volume.

    """
    tile_width = min(volume_provider.shape[0], tile_size[0])
    tile_height = min(volume_provider.shape[1], tile_size[1])

    x_high = volume_provider.shape[0] - tile_width
    y_high = volume_provider.shape[1] - tile_height
    x = np.random.randint(0, x_high + 1) if x_high > 0 else 0
    y = np.random.randint(0, y_high + 1) if y_high > 0 else 0
    z = np.random.randint(0, volume_provider.shape[2])
    tile = volume_provider[x : x + tile_width, y : y + tile_height, z]

    if tile.shape == tile_size:
        return tile

    padded_tile = np.zeros(tile_size, dtype=tile.dtype)
    x_pad = (tile_size[0] - tile.shape[0]) // 2
    y_pad = (tile_size[1] - tile.shape[1]) // 2
    padded_tile[
        x_pad : x_pad + tile.shape[0],
        y_pad : y_pad + tile.shape[1],
    ] = tile
    return padded_tile


def get_random_zyx_subvolume(
    volume_provider,
    subvolume_size_zyx: Tuple[int, int, int],
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Get a random subvolume from the volume.

    Arguments:
        volume_provider: The volume provider to get the subvolume from.
        subvolume_size: The size of the subvolume to get.

    Returns:
        np.ndarray: A random subvolume from the volume.
        tuple[np.ndarray, dict]: The subvolume and metadata if
            `return_metadata` is True.

    """
    requested_z, requested_y, requested_x = subvolume_size_zyx
    actual_x = min(volume_provider.shape[0], requested_x)
    actual_y = min(volume_provider.shape[1], requested_y)
    actual_z = min(volume_provider.shape[2], requested_z)

    x_high = volume_provider.shape[0] - actual_x
    y_high = volume_provider.shape[1] - actual_y
    z_high = volume_provider.shape[2] - actual_z
    x = np.random.randint(0, x_high + 1) if x_high > 0 else 0
    y = np.random.randint(0, y_high + 1) if y_high > 0 else 0
    z = np.random.randint(0, z_high + 1) if z_high > 0 else 0

    subvolume = volume_provider[
        x : x + actual_x,
        y : y + actual_y,
        z : z + actual_z,
    ].swapaxes(0, 2)

    z_pad = requested_z - subvolume.shape[0]
    y_pad = requested_y - subvolume.shape[1]
    x_pad = requested_x - subvolume.shape[2]
    z_pad_before = z_pad // 2
    y_pad_before = y_pad // 2
    x_pad_before = x_pad // 2
    z_pad_after = z_pad - z_pad_before
    y_pad_after = y_pad - y_pad_before
    x_pad_after = x_pad - x_pad_before

    if subvolume.shape != subvolume_size_zyx:
        subvolume = np.pad(
            subvolume,
            (
                (z_pad_before, z_pad_after),
                (y_pad_before, y_pad_after),
                (x_pad_before, x_pad_after),
            ),
            mode="constant",
        )

    if not return_metadata:
        return subvolume

    annotated_local_z_index = requested_z // 2
    source_local_z_index = annotated_local_z_index - z_pad_before
    if 0 <= source_local_z_index < actual_z:
        annotated_global_z_index = int(z + source_local_z_index)
    else:
        annotated_global_z_index = None

    sample_metadata = {
        "cutout_origin_xyz": [int(x), int(y), int(z)],
        "cutout_shape_xyz": [int(actual_x), int(actual_y), int(actual_z)],
        "requested_shape_xyz": [int(requested_x), int(requested_y), int(requested_z)],
        "padding_before_xyz": [int(x_pad_before), int(y_pad_before), int(z_pad_before)],
        "padding_after_xyz": [int(x_pad_after), int(y_pad_after), int(z_pad_after)],
        "annotated_local_z_index": int(annotated_local_z_index),
        "annotated_global_z_index": annotated_global_z_index,
    }
    return subvolume, sample_metadata


def export_to_img_stack(
    volume_provider: VolumeProvider,
    img_dir: Union[str, pathlib.Path],
    img_format: str = "png",
    downsample_factor: Tuple[int, int, int] = (1, 1, 1),
    progress: bool = True,
    parallel_jobs: Union[int, bool] = False,
    **kwargs,
):
    """
    Export a volume to an image stack.

    Arguments:
        volume_provider: The volume provider to export.
        img_dir: The directory to save the image stack to.
        img_format: The image format to use.
        downsample_factor: The downsample factor to use.
        progress: Whether to show a progress bar.
        parallel_jobs: The number of parallel jobs to use. If False, no parallel
            jobs are used. If True, the number of jobs is set to the number of
            cores.
        **kwargs: Additional arguments to pass to `skimage.io.imsave`.

    """
    if downsample_factor is None:
        downsample_factor = (1, 1, 1)

    img_dir = pathlib.Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    def _export_slice(i):
        img = volume_provider[:, :, i]
        img = img[
            :: downsample_factor[0],
            :: downsample_factor[1],
        ]
        img = img.squeeze()

        # Cast if file format requires it:
        if img_format in ["png", "jpg", "jpeg"]:
            img = img.astype(np.uint8)

        img_path = img_dir / f"{i:04d}.{img_format}"
        Image.fromarray(img).save(img_path, **kwargs)

    _prog = tqdm.tqdm if progress else lambda x: x
    if parallel_jobs is False:
        for i in _prog(range(volume_provider.shape[2])):
            _export_slice(i)
    else:
        _ = Parallel(n_jobs=parallel_jobs)(
            delayed(_export_slice)(i) for i in _prog(range(volume_provider.shape[2]))
        )
