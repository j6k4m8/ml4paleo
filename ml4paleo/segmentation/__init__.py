import pathlib
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from intern.utils.parallel import block_compute
import zarr
import numpy as np
from ml4paleo.volume_providers import VolumeProvider
from .segmenter import Segmenter3D
from .rf import RandomForest3DSegmenter

import tqdm
from joblib import Parallel, delayed


def segment_chunk_and_write(
    xs: Tuple[int, int],
    ys: Tuple[int, int],
    zs: Tuple[int, int],
    volume_provider: VolumeProvider,
    segmenter: Segmenter3D,
    seg_path: str,
) -> bool:
    """
    Segment a chunk of a job.
    """
    # Get the volume for the chunk:
    volume = volume_provider[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]]
    # Segment the volume:
    # seg_volume = np.zeros(volume.shape, dtype=np.uint64)
    seg_volume = segmenter.segment(volume)
    # Write the seg to the seg path zarr:
    seg_zarr = zarr.open(seg_path, mode="a")
    seg_zarr[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]] = seg_volume
    return True


def segment_volume_to_zarr(
    vol_provider: VolumeProvider,
    seg_path: pathlib.Path,
    segmenter: Segmenter3D,
    chunk_size,
    parallel: Union[bool, int] = True,
    progress: bool = True,
    progress_callback: Optional[Callable[[int, Any, int], Any]] = None,
):
    seg_path.mkdir(parents=True, exist_ok=True)

    # Create the Zarr file for the segmentation.
    zarr.open(
        str(seg_path),
        mode="w",
        dtype="uint64",
        shape=vol_provider.shape,
        chunks=chunk_size,
        write_empty_chunks=False,
    )

    # Now segment the job.
    # We segment the job in chunks, and save the results in the
    # CONFIG.segmentation_directory as another Zarr file.
    chunks_to_segment = block_compute(
        0,
        vol_provider.shape[0],
        0,
        vol_provider.shape[1],
        0,
        vol_provider.shape[2],
        block_size=chunk_size,
    )

    _progfn = tqdm.tqdm if progress else lambda x: x
    if progress_callback is not None:
        # Send the callback the current progress out of the total.
        def _prog(x):
            for i, y in _progfn(enumerate(x)):
                progress_callback(i, y, len(x))
                yield y

    else:
        _prog = _progfn  # type: ignore

    # for xs, ys, zs in chunks_to_segment
    _ = Parallel(n_jobs=parallel)(
        delayed(segment_chunk_and_write)(xs, ys, zs, vol_provider, segmenter, seg_path)
        for xs, ys, zs in _prog(chunks_to_segment)
    )


__all__ = ["Segmenter3D", "RandomForest3DSegmenter", "segment_chunk_and_write"]
