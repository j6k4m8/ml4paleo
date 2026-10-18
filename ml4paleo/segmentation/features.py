"""Versioned image feature recipes and bounded, disposable worker caches.

Features depend on the immutable image, normalization and recipe, not labels
or classifier weights. Each extractor owns its output order and halo. Neural
plugins can continue to consume raw images without using this interface.
"""

import contextlib
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import logging
import os
import pathlib
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import zstandard
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dataset import Box, ImageArray, normalize, read_box

log = logging.getLogger(__name__)


class FeatureRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    version: str
    params: dict[str, Any]


class Extractor(Protocol):
    halo: int

    def width(self, channels: int) -> int: ...

    def bytes_per_voxel(self, channels: int, threads: int) -> int: ...

    def compute(self, image: np.ndarray, threads: int | None) -> np.ndarray: ...


class MultiscaleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sigma_max: float = Field(8, ge=1, le=8)
    intensity: bool = True
    edges: bool = True
    texture: bool = True

    @model_validator(mode="after")
    def nonempty(self):
        if not (self.intensity or self.edges or self.texture):
            raise ValueError("Choose at least one feature group")
        return self


class Multiscale:
    def __init__(self, params: dict[str, Any]):
        self.params = MultiscaleParams(**params)
        self.halo = int(4 * self.params.sigma_max + 0.5) + 2

    def width(self, channels: int) -> int:
        scales = int(np.log2(self.params.sigma_max)) + 1
        groups = (
            int(self.params.intensity)
            + int(self.params.edges)
            + 3 * int(self.params.texture)
        )
        return channels * scales * groups

    def compute(self, image: np.ndarray, threads: int | None) -> np.ndarray:
        from skimage.feature import multiscale_basic_features

        return np.concatenate(
            [
                multiscale_basic_features(
                    channel, **self.params.model_dump(), sigma_min=1, workers=threads
                ).astype(np.float32)
                for channel in image
            ],
            axis=-1,
        )

    def bytes_per_voxel(self, channels: int, threads: int) -> int:
        count = self.width(channels)
        scales = int(np.log2(self.params.sigma_max)) + 1
        at_once = min(max(1, threads), scales)
        # Conservative for both recipes: intermediate filter arrays, stacking,
        # and concurrent scale computation. Each extractor owns this estimate.
        return max(16 * count, 4 * count + 160 * at_once)


def extractor(recipe: FeatureRecipe) -> Extractor:
    # Add extractors here, rather than making callers assume one featurization.
    registry: dict[tuple[str, str], Callable[[dict[str, Any]], Extractor]] = {
        ("multiscale", "1"): Multiscale
    }
    implementation = registry.get((recipe.name, recipe.version))
    if implementation is None:
        raise ValueError(
            f"Unsupported feature recipe {recipe.name} version {recipe.version}"
        )
    return implementation(recipe.params)


def multiscale_recipe(sigma_max: float, *, texture: bool = True) -> FeatureRecipe:
    return FeatureRecipe(
        name="multiscale",
        version="1",
        params=MultiscaleParams(sigma_max=sigma_max, texture=texture).model_dump(),
    )


class FeatureCache:
    """Compressed disk chunks plus a byte-bounded LRU of decoded features.

    Atomic replacement and striped locks let CPU/GPU workers share a cache
    directory. Only cache-owned files are evicted. Cache errors fall back to
    computation, and cancellation is checked before publishing a chunk.
    """

    def __init__(
        self,
        directory: pathlib.Path,
        memory_bytes: int = 128 * 1024**2,
        disk_bytes: int = 2 * 1024**3,
        idle_seconds: float = 600,
    ):
        self.directory = directory
        self.memory_bytes = max(0, memory_bytes)
        self.disk_bytes = max(0, disk_bytes)
        self.idle_seconds = idle_seconds
        self._memory: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.prune()

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def prune_memory(self) -> None:
        with self._lock:
            cutoff = time.monotonic() - self.idle_seconds
            for key in list(self._memory):
                if self._memory[key][1] < cutoff:
                    self._bytes -= self._memory.pop(key)[0].nbytes

    def _remember(self, key: str, value: np.ndarray) -> None:
        value.flags.writeable = False
        with self._lock:
            previous = self._memory.pop(key, None)
            if previous:
                self._bytes -= previous[0].nbytes
            while self._memory and self._bytes + value.nbytes > self.memory_bytes:
                self._bytes -= self._memory.popitem(last=False)[1][0].nbytes
            if value.nbytes <= self.memory_bytes:
                self._memory[key] = (value, time.monotonic())
                self._bytes += value.nbytes

    @contextlib.contextmanager
    def _file_lock(self, name: str):
        with (self.directory / name).open("a+b") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    def prune(self) -> None:
        self.prune_memory()
        with self._file_lock("eviction.lock"):
            files = []
            for path in self.directory.glob("*.f32.zst"):
                with contextlib.suppress(FileNotFoundError):
                    stat = path.stat()
                    files.append((stat.st_mtime, stat.st_size, path))
            size = sum(entry[1] for entry in files)
            for _, length, path in sorted(files):
                if size <= self.disk_bytes:
                    break
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                size -= length
            # A killed worker may leave a partial write. Never touch recent
            # scratch files another worker could still be writing.
            for path in self.directory.glob("*.partial"):
                with contextlib.suppress(FileNotFoundError):
                    if path.stat().st_mtime < time.time() - 3600:
                        path.unlink()

    def get(
        self,
        identity: dict[str, Any],
        shape: tuple[int, ...],
        compute: Callable[[], np.ndarray],
        check: Callable[[], None],
    ) -> np.ndarray:
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self.prune_memory()
        # A fixed number of lock files avoids accumulating one per chunk.
        try:
            with self._file_lock(f"stripe-{int(key[:4], 16) % 64}.lock"):
                check()
                with self._lock:
                    found = self._memory.pop(key, None)
                    if found:
                        self._memory[key] = (found[0], time.monotonic())
                        return found[0]
                path = self.directory / f"{key}.f32.zst"
                expected = int(np.prod(shape)) * 4
                try:
                    encoded = path.read_bytes()
                    if zstandard.frame_content_size(encoded) != expected:
                        raise ValueError("Wrong feature cache frame size")
                    raw = zstandard.ZstdDecompressor().decompress(
                        encoded, max_output_size=expected
                    )
                    if len(raw) != expected:
                        raise ValueError("Wrong feature cache shape")
                    value = np.frombuffer(raw, dtype="<f4").reshape(shape)
                    os.utime(path, None)
                except (FileNotFoundError, ValueError, zstandard.ZstdError):
                    value = np.ascontiguousarray(compute(), dtype="<f4")
                    if value.shape != shape:
                        raise ValueError(
                            "Feature extractor returned the wrong shape"
                        ) from None
                    check()
                    encoded = zstandard.ZstdCompressor(
                        level=1, write_checksum=True
                    ).compress(value)
                    if len(encoded) <= self.disk_bytes:
                        scratch = None
                        try:
                            with tempfile.NamedTemporaryFile(
                                dir=self.directory, suffix=".partial", delete=False
                            ) as file:
                                scratch = pathlib.Path(file.name)
                                file.write(encoded)
                            scratch.replace(path)
                        finally:
                            if scratch:
                                scratch.unlink(missing_ok=True)
                        self.prune()
                check()
                self._remember(key, value)
                return value
        except OSError as error:
            log.warning("Feature cache unavailable; computing this chunk: %s", error)
            check()
            return compute()


class FeatureVolume:
    """Assemble arbitrary regions from reusable image-aligned feature chunks."""

    def __init__(
        self,
        image: ImageArray,
        image_id: str,
        window: tuple[float, float],
        recipe: FeatureRecipe,
        cache: FeatureCache,
        threads: int,
        tile: int = 64,
        check: Callable[[], None] = lambda: None,
    ):
        self.image, self.window, self.recipe = image, window, recipe
        self.cache, self.threads, self.tile, self.check = cache, threads, tile, check
        self.extractor = extractor(recipe)
        self.width = self.extractor.width(image.shape[0])
        self.identity = {
            "format": 1,
            "image_id": image_id,
            "image_shape": list(image.shape),
            "window": list(window),
            "normalization": "float32-window-v1",
            "boundary": "edge-pad-v1",
            "recipe": recipe.model_dump(),
            "libraries": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-image")
            },
        }

    def read(self, box: Box) -> np.ndarray:
        if any(
            box[a] < 0 or box[a + 3] > self.image.shape[a + 1] or box[a] >= box[a + 3]
            for a in range(3)
        ):
            raise ValueError("Feature region is outside the image")
        result = np.empty(
            (*[box[a + 3] - box[a] for a in range(3)], self.width), np.float32
        )
        for key in itertools.product(
            *(
                range(box[a] // self.tile, (box[a + 3] - 1) // self.tile + 1)
                for a in range(3)
            )
        ):
            self.check()
            origin = [n * self.tile for n in key]
            end = [
                min(origin[a] + self.tile, self.image.shape[a + 1]) for a in range(3)
            ]
            chunk_box: Box = (*origin, *end)  # type: ignore[assignment]
            shape = (*[end[a] - origin[a] for a in range(3)], self.width)

            def compute(chunk_box=chunk_box, shape=shape):
                h = self.extractor.halo
                block = normalize(read_box(self.image, chunk_box, h), self.window)
                features = self.extractor.compute(block, self.threads)
                return np.ascontiguousarray(
                    features[tuple(slice(h, h + shape[a]) for a in range(3))]
                )

            chunk = self.cache.get(
                {**self.identity, "box": list(chunk_box)}, shape, compute, self.check
            )
            low = [max(box[a], origin[a]) for a in range(3)]
            high = [min(box[a + 3], end[a]) for a in range(3)]
            result[
                tuple(slice(low[a] - box[a], high[a] - box[a]) for a in range(3))
            ] = chunk[
                tuple(slice(low[a] - origin[a], high[a] - origin[a]) for a in range(3))
            ]
        return result
