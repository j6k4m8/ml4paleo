"""
Exports: one zip archive of an artifact's files, or of a volume's slices as
TIFF or PNG images.

An archive is written as numbered parts (`parts/00000`, ...) of `PART_BYTES`
each (the last one shorter), so every storage backend takes them in single
requests, the storage proxy included; the API serves the parts back to back
as one file. Entries have a fixed date and a stable order, so the same input
gives the same bytes, and a retried job writes the same parts again.

Slices keep the orientation of an uploaded stack: one image per z, with y
down its rows and x along its columns.
"""

import io
import zipfile
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal

import numpy as np

PART_BYTES = 64 * 1024**2
# Entries' date, the earliest a zip file can hold.
ZIP_DATE = (1980, 1, 1, 0, 0, 0)

SliceFormat = Literal["tiff", "png"]
EXTENSIONS: dict[SliceFormat, str] = {"tiff": "tif", "png": "png"}
# PNG holds 8- and 16-bit unsigned values; TIFF holds any of ours.
PNG_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16))


def part_key(index: int) -> str:
    return f"parts/{index:05d}"


class Parts(io.RawIOBase):
    """
    A write-only, unseekable file that passes its bytes to `put(index,
    data)` in parts of `part_bytes`. Call `finish` once everything is
    written; it returns the parts' sizes.
    """

    def __init__(
        self, put: Callable[[int, bytes], None], part_bytes: int | None = None
    ):
        self._put = put
        self._part_bytes = part_bytes or PART_BYTES
        # One part's worth, filled and handed over again and again, so memory
        # stays at a part however large the writes are.
        self._buffer = bytearray(self._part_bytes)
        self._filled = 0
        self._aborted = False
        self.sizes: list[int] = []

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        view = memoryview(data).cast("B")
        written = view.nbytes
        while view.nbytes and not self._aborted:
            take = min(self._part_bytes - self._filled, view.nbytes)
            self._buffer[self._filled : self._filled + take] = view[:take]
            self._filled += take
            view = view[take:]
            if self._filled == self._part_bytes:
                self._emit()
        return written

    def abort(self) -> None:
        """Drop everything written from now on (the job is stopping)."""
        self._aborted = True

    def finish(self) -> list[int]:
        if self._filled or not self.sizes:
            self._emit()
        return self.sizes

    def _emit(self) -> None:
        self._put(len(self.sizes), bytes(memoryview(self._buffer)[: self._filled]))
        self.sizes.append(self._filled)
        self._filled = 0


def open_archive(out: Parts) -> zipfile.ZipFile:
    return zipfile.ZipFile(out, mode="w", allowZip64=True)


def add_entry(
    archive: zipfile.ZipFile,
    name: str,
    chunks: Iterable[Any],
    size: int,
    compress: bool = False,
) -> None:
    """
    Add one entry of `size` bytes, written from `chunks` as they come.
    Compress only what isn't compressed already.
    """
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    # The declared size decides whether the entry needs zip64 fields.
    info.file_size = size
    with archive.open(info, mode="w", force_zip64=size >= 2**31) as entry:
        for chunk in chunks:
            entry.write(chunk)


def encode_slice(plane: np.ndarray, fmt: SliceFormat) -> bytes:
    """
    One 2D (y, x) plane as a TIFF (zlib-compressed) or PNG file.
    """
    buffer = io.BytesIO()
    if fmt == "png":
        from PIL import Image

        if plane.dtype not in PNG_DTYPES:
            raise ValueError(
                f"PNG holds 8- or 16-bit unsigned values, not {plane.dtype}; "
                "export TIFF instead"
            )
        Image.fromarray(np.ascontiguousarray(plane)).save(buffer, format="PNG")
    else:
        import tifffile

        tifffile.imwrite(buffer, np.ascontiguousarray(plane), compression="zlib")
    return buffer.getvalue()


def slice_names(
    shape_czyx: tuple[int, int, int, int], fmt: SliceFormat, folder: str
) -> Callable[[int, int], str]:
    """
    Names for each (channel, z) image: `<folder>/z00000.tif`, or with more
    than one channel, `<folder>/c0/z00000.tif`.
    """
    channels, depth = shape_czyx[0], shape_czyx[1]
    width = max(5, len(str(depth - 1)))
    extension = EXTENSIONS[fmt]

    def name(c: int, z: int) -> str:
        channel = f"c{c}/" if channels > 1 else ""
        return f"{folder}/{channel}z{z:0{width}d}.{extension}"

    return name


def slab_depth(
    shape_czyx: tuple[int, int, int, int],
    itemsize: int,
    chunk_z: int,
    budget_bytes: int,
) -> int:
    """
    How many z to read at once: a whole chunk's depth (so no chunk is read
    twice) when that fits in half the budget, else fewer, and 0 when not
    even one z of every channel fits (then read a channel at a time).
    """
    channels, _, y, x = shape_czyx
    per_z = max(1, channels * y * x * itemsize)
    return int(min(chunk_z, budget_bytes // 2 // per_z))


def slices(array: Any, depth: int) -> Iterator[tuple[int, int, np.ndarray]]:
    """
    Every (channel, z, plane) of a (c, z, y, x) or (z, y, x) array, read
    `depth` z at a time (with depth 0, one plane at a time), in z order and
    then channel order.
    """
    planar = len(array.shape) == 3
    total = array.shape[-3]
    channels = 1 if planar else array.shape[0]
    for z0 in range(0, total, max(1, depth)):
        z1 = min(total, z0 + max(1, depth))
        if depth == 0 and not planar:
            for c in range(channels):
                yield c, z0, np.asarray(array[c, z0])
            continue
        slab = np.asarray(array[z0:z1] if planar else array[:, z0:z1])
        if planar:
            slab = slab[np.newaxis]
        for i in range(slab.shape[1]):
            for c in range(slab.shape[0]):
                yield c, z0 + i, slab[c, i]
