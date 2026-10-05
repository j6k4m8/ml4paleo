"""
OME-Zarr 0.5 images for ml4paleo volumes.

In storage, every image is laid out as `c, z, y, x`, as OME-Zarr expects. The
volume providers in `ml4paleo.volume_providers` index `(x, y, z)` instead.
`write_from_provider` is the only place that converts between the two. Keep it
that way: v1 shipped both a transposed PNG export and mirrored meshes because
axis conversions were scattered through the code.

Each image is a zarr v3 group holding one array per resolution level ("0" is
full resolution). Arrays are sharded: small chunks for fast random reads in
the viewer, inside larger shards so the object count stays manageable. Writing
into a shard rewrites the whole shard, so writers must own whole shards; the
helpers here always work in shard-aligned blocks.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import zarr

from .blocks import Block, iter_blocks
from .storage import StorageGrant, zarr_store
from .volume_providers.volume_provider import VolumeProvider

OME_VERSION = "0.5"
DEFAULT_CHUNK_ZYX = (64, 64, 64)
DEFAULT_SHARD_ZYX = (512, 512, 512)
SPATIAL_AXES = ("z", "y", "x")

DownsampleMethod = Literal["mean", "mode"]


@dataclass(frozen=True)
class LevelSpec:
    """
    One level of a multiscale pyramid.
    """

    path: str
    shape_zyx: tuple[int, int, int]
    # Cumulative downsampling factor relative to level 0.
    factor_zyx: tuple[int, int, int]


def plan_levels(
    shape_zyx: Sequence[int],
    voxel_size_zyx: Sequence[float] | None = None,
    chunk_zyx: Sequence[int] = DEFAULT_CHUNK_ZYX,
) -> list[LevelSpec]:
    """
    Plan the levels of a pyramid for a volume of `shape_zyx`.

    Each level halves the axes whose voxels are finest, until the whole level
    fits in one chunk. An axis whose voxels are already at least twice as
    large as the finest axis is left alone, so anisotropic scans approach
    isotropic voxels before shrinking further.
    """
    shape = [int(s) for s in shape_zyx]
    size = [float(s) for s in (voxel_size_zyx or (1.0, 1.0, 1.0))]
    factor = [1, 1, 1]
    levels = [LevelSpec("0", _as_zyx(shape), _as_zyx(factor))]
    while any(s > c for s, c in zip(shape, chunk_zyx, strict=True)):
        finest = min(sz for sz, s in zip(size, shape, strict=True) if s > 1)
        step = [
            2 if s > 1 and sz < 2 * finest else 1
            for s, sz in zip(shape, size, strict=True)
        ]
        shape = [math.ceil(s / st) for s, st in zip(shape, step, strict=True)]
        size = [sz * st for sz, st in zip(size, step, strict=True)]
        factor = [f * st for f, st in zip(factor, step, strict=True)]
        levels.append(LevelSpec(str(len(levels)), _as_zyx(shape), _as_zyx(factor)))
    return levels


class OmeImage:
    """
    A multiscale OME-Zarr 0.5 image with axes `c, z, y, x`.
    """

    def __init__(self, group: zarr.Group):
        self.group = group
        multiscales = group.attrs["ome"]["multiscales"][0]
        self._datasets = multiscales["datasets"]
        self.unit: str | None = multiscales["axes"][1].get("unit")

    @classmethod
    def create(
        cls,
        grant: StorageGrant,
        *,
        shape_czyx: Sequence[int],
        dtype: np.dtype | str,
        voxel_size_zyx: Sequence[float] | None = None,
        unit: str | None = "millimeter",
        chunk_zyx: Sequence[int] = DEFAULT_CHUNK_ZYX,
        shard_zyx: Sequence[int] = DEFAULT_SHARD_ZYX,
        channel_names: Sequence[str] | None = None,
        name: str = "image",
    ) -> "OmeImage":
        """
        Create an empty image with every pyramid level allocated.

        If the voxel size is unknown, pass `voxel_size_zyx=None`: scales are
        then in voxels and no unit is recorded.
        """
        if len(shape_czyx) != 4:
            raise ValueError(f"shape_czyx must have 4 dimensions, got {shape_czyx}")
        if any(s % c for s, c in zip(shard_zyx, chunk_zyx, strict=True)):
            raise ValueError("shard_zyx must be a multiple of chunk_zyx")
        num_channels, *spatial = (int(s) for s in shape_czyx)
        if voxel_size_zyx is None:
            unit = None
        base_size = tuple(float(v) for v in (voxel_size_zyx or (1.0, 1.0, 1.0)))
        levels = plan_levels(spatial, voxel_size_zyx, chunk_zyx)

        axes: list[dict[str, str]] = [{"name": "c", "type": "channel"}]
        for axis in SPATIAL_AXES:
            axes.append({"name": axis, "type": "space", **({"unit": unit} if unit else {})})
        datasets = [
            {
                "path": level.path,
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0]
                        + [b * f for b, f in zip(base_size, level.factor_zyx, strict=True)],
                    }
                ],
            }
            for level in levels
        ]
        attributes = {
            "ome": {
                "version": OME_VERSION,
                "multiscales": [{"name": name, "axes": axes, "datasets": datasets}],
                "omero": {
                    "channels": [
                        {"label": label}
                        for label in (
                            channel_names or [f"channel {i}" for i in range(num_channels)]
                        )
                    ]
                },
            }
        }
        group = zarr.create_group(
            store=zarr_store(grant), zarr_format=3, attributes=attributes
        )
        for level in levels:
            chunks = _fit_to_shape(chunk_zyx, level.shape_zyx)
            shards = tuple(
                min(shard, math.ceil(size / chunk) * chunk)
                for shard, size, chunk in zip(shard_zyx, level.shape_zyx, chunks, strict=True)
            )
            group.create_array(
                level.path,
                shape=(num_channels, *level.shape_zyx),
                chunks=(1, *chunks),
                shards=(1, *shards),
                dtype=dtype,
                fill_value=0,
                dimension_names=("c", *SPATIAL_AXES),
            )
        return cls(group)

    @classmethod
    def open(cls, grant: StorageGrant) -> "OmeImage":
        mode = "r+" if grant.access == "rw" else "r"
        return cls(zarr.open_group(store=zarr_store(grant), mode=mode))

    @property
    def num_levels(self) -> int:
        return len(self._datasets)

    def array(self, level: int = 0) -> zarr.Array:
        array = self.group[self._datasets[level]["path"]]
        assert isinstance(array, zarr.Array)
        return array

    @property
    def shape_czyx(self) -> tuple[int, int, int, int]:
        return tuple(self.array(0).shape)  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        return self.array(0).dtype

    def scale_zyx(self, level: int = 0) -> tuple[float, float, float]:
        """
        Return the voxel size of a level, in `unit` (or in level-0 voxels when
        `unit` is None).
        """
        scale = self._datasets[level]["coordinateTransformations"][0]["scale"]
        return (float(scale[1]), float(scale[2]), float(scale[3]))

    @property
    def voxel_size_zyx(self) -> tuple[float, float, float] | None:
        """
        The physical voxel size at full resolution, or None if unknown.
        """
        return self.scale_zyx(0) if self.unit else None

    def shard_blocks(self, level: int = 0) -> list[Block]:
        """
        Return the shard-aligned blocks of a level, in z, y, x index space.
        """
        array = self.array(level)
        return list(iter_blocks(array.shape[1:], array.shards[1:]))  # type: ignore[index]


def write_from_provider(
    provider: VolumeProvider,
    image: OmeImage,
    *,
    channel: int = 0,
    z_range: tuple[int, int] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """
    Copy a volume provider into level 0 of `image`, one z-slab of shards at a
    time.

    This is the single place where ml4paleo converts (x, y, z) volumes into
    (c, z, y, x) storage. `z_range` lets parallel jobs each write their own
    slabs; its bounds must sit on shard boundaries so that no two jobs write
    the same shard.
    """
    array = image.array(0)
    _, depth, height, width = array.shape
    provider_shape = tuple(int(s) for s in provider.shape)
    if provider_shape != (width, height, depth):
        raise ValueError(
            f"Provider shape (x, y, z) {provider_shape} does not match image "
            f"shape (z, y, x) {(depth, height, width)}"
        )
    shard_depth = array.shards[1]  # type: ignore[index]
    z0, z1 = z_range or (0, depth)
    for bound in (z0, z1):
        if bound % shard_depth and bound != depth:
            raise ValueError(f"z_range bound {bound} is not on a shard boundary")
    slabs = list(range(z0, z1, shard_depth))
    for index, slab_start in enumerate(slabs):
        slab_stop = min(slab_start + shard_depth, z1)
        data_xyz = np.asarray(provider[:, :, slab_start:slab_stop])
        if data_xyz.ndim == 2:
            data_xyz = data_xyz[:, :, np.newaxis]
        array[channel, slab_start:slab_stop, :, :] = data_xyz.transpose(2, 1, 0)
        if progress is not None:
            progress(index + 1, len(slabs))


def downsample_level(
    image: OmeImage,
    level: int,
    method: DownsampleMethod = "mean",
    blocks: Sequence[Block] | None = None,
) -> None:
    """
    Compute level `level + 1` from level `level`.

    `blocks` are output shards of the coarser level (by default all of them);
    pyramid jobs pass a subset so they can run in parallel. "mean" suits
    intensity images; "mode" suits label images and prefers non-zero labels,
    so thin labeled structures survive downsampling.
    """
    source = image.array(level)
    target = image.array(level + 1)
    step = tuple(
        round(c / f)
        for c, f in zip(image.scale_zyx(level + 1), image.scale_zyx(level), strict=True)
    )
    for block in blocks if blocks is not None else image.shard_blocks(level + 1):
        source_start = [lo * s for lo, s in zip(block.start, step, strict=True)]
        source_stop = [
            min(hi * s, size)
            for hi, s, size in zip(block.stop, step, source.shape[1:], strict=True)
        ]
        data = np.asarray(
            source[(slice(None), *(slice(a, b) for a, b in zip(source_start, source_stop, strict=True)))]
        )
        reduced = _reduce(data, step, block.shape, method).astype(target.dtype)
        target[(slice(None), *block.slices)] = reduced


def build_pyramid(image: OmeImage, method: DownsampleMethod = "mean") -> None:
    """
    Fill every level below level 0, in order.
    """
    for level in range(image.num_levels - 1):
        downsample_level(image, level, method)


def _reduce(
    data: np.ndarray,
    step: Sequence[int],
    out_shape_zyx: Sequence[int],
    method: DownsampleMethod,
) -> np.ndarray:
    # Pad each axis up to a whole number of windows. Mean pads with edge
    # values; mode pads with 0, which never wins a vote.
    pad = [(0, 0)] + [
        (0, out * s - size)
        for out, s, size in zip(out_shape_zyx, step, data.shape[1:], strict=True)
    ]
    if method == "mean":
        padded = np.pad(data, pad, mode="edge")
    else:
        padded = np.pad(data, pad, mode="constant", constant_values=0)
    c = padded.shape[0]
    (oz, oy, ox), (sz, sy, sx) = out_shape_zyx, step
    windows = (
        padded.reshape(c, oz, sz, oy, sy, ox, sx)
        .transpose(0, 1, 3, 5, 2, 4, 6)
        .reshape(c, oz, oy, ox, sz * sy * sx)
    )
    if method == "mean":
        mean = windows.mean(axis=-1, dtype=np.float64)
        return np.rint(mean) if np.issubdtype(data.dtype, np.integer) else mean
    return _mode_of_nonzero(windows)


def _mode_of_nonzero(windows: np.ndarray) -> np.ndarray:
    """
    Return the most common non-zero value in each window (0 if all are zero).
    Ties go to the smaller value.
    """
    values = np.unique(windows)
    values = values[values != 0]
    if values.size == 0:
        return np.zeros(windows.shape[:-1], dtype=windows.dtype)
    counts = np.stack([(windows == v).sum(axis=-1) for v in values], axis=-1)
    best = values[counts.argmax(axis=-1)]
    return np.where(counts.max(axis=-1) > 0, best, 0).astype(windows.dtype)


def _fit_to_shape(chunk_zyx: Sequence[int], shape_zyx: Sequence[int]) -> tuple[int, ...]:
    return tuple(min(c, s) for c, s in zip(chunk_zyx, shape_zyx, strict=True))


def _as_zyx(values: Sequence[int]) -> tuple[int, int, int]:
    z, y, x = (int(v) for v in values)
    return (z, y, x)
