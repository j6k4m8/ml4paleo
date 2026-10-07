"""
Labels made elsewhere, brought into a project from image files.

A label file is a stack of 2D images the size of the project's image, one per
z slice: a TIFF with a page per slice, or a zip of TIFF or PNG slices in name
order (numbers compared as numbers, as ingest orders a scan's slices). Rows
run along y and columns along x, as in a scan's slices. Each pixel holds a
whole number, and before the labels come in the person says what each
number becomes: background, a class, or nothing (left unlabeled).

`file_kind` tells the formats apart; `ZipPlanes` and `ImagePlanes` read a
file's slices one at a time; `count_values` reads them all once, checking
their size, and counts each value's voxels; `chunk_blocks` turns them,
through a `Lookup` from values to label values, into the label chunks to
write, and `chunk_delta` into the edit that writes one. Problems with the file
raise `ml4paleo.ingest.IngestError`, with a message for the person who
uploaded it.
"""

import os
import struct
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import BinaryIO, Literal

import numpy as np
from PIL import Image

from .ingest import (
    DEFAULT_LIMITS,
    IngestError,
    SliceLimits,
    entry_count,
    open_archive,
    slice_members,
)
from .labels import BACKGROUND, FIRST_CLASS, LABEL_CHUNK_ZYX, MAX_CLASS
from .labels.deltas import ChunkDelta, split_into_deltas

Kind = Literal["tiff", "png", "zip"]
ChunkKey = tuple[int, int, int]

# A project has room for this many classes, so a file may hold this many
# values besides 0.
MAX_VALUES = MAX_CLASS - FIRST_CLASS + 1
# What `chunk_blocks` may hold at once, unless told otherwise.
DEFAULT_BLOCK_BYTES = 256 * 1024**2
# Bytes per pixel of the image modes that hold one whole number a pixel.
# Palette images keep their palette indices, which is what label images
# saved with a palette mean.
MODE_BYTES = {
    "1": 1,
    "L": 1,
    "P": 1,
    "I;16": 2,
    "I;16L": 2,
    "I;16B": 2,
    "I;16N": 2,
    "I": 4,
    "F": 4,
}
# What Pillow raises for files it can't read (damaged TIFF tags raise all
# sorts).
READ_ERRORS = (
    OSError,
    ValueError,
    EOFError,
    SyntaxError,
    TypeError,
    KeyError,
    IndexError,
    struct.error,
    Image.DecompressionBombError,
)
NOT_LABELS = (
    "The file isn't a TIFF or a zip. Upload a TIFF stack, or a zip of TIFF or "
    "PNG slices."
)
# Only lossless formats, which keep each pixel's number exactly (and fewer
# of Pillow's decoders see what people upload).
FORMATS = ["TIFF", "PNG"]
UNREADABLE = (
    "it isn't a TIFF or PNG Pillow can read. Labels need to be whole numbers "
    "(8, 16, or 32 bits) or 32-bit floats; if they are, the file may be damaged."
)
TOO_MANY = (
    f"The labels have more than {MAX_VALUES} different values, more than a "
    "project has room for as classes."
)


def file_kind(fileobj: BinaryIO) -> Kind:
    """
    Whether a file is a TIFF, a PNG, or a zip, from its first bytes (and a
    zip's end record).
    """
    fileobj.seek(0)
    head = fileobj.read(8)
    fileobj.seek(0)
    # Little- and big-endian TIFF, and BigTIFF.
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    if head == b"\x89PNG\r\n\x1a\n":
        return "png"
    found = entry_count(fileobj)
    fileobj.seek(0)
    if found is not None:
        return "zip"
    raise IngestError(NOT_LABELS)


def _size(shape_zyx: Sequence[int]) -> str:
    z, y, x = shape_zyx
    return f"{x} × {y} × {z}"


def _whole(pixels: np.ndarray, name: str) -> np.ndarray:
    """
    A slice's pixels as whole numbers: integers in the machine's byte order,
    and floats that hold whole numbers as integers. Anything else is refused.
    """
    if pixels.dtype == np.bool_:
        return pixels.astype(np.uint8)
    if pixels.dtype.kind in "iu":
        return pixels.astype(pixels.dtype.newbyteorder("="), copy=False)
    if pixels.dtype.kind == "f":
        with np.errstate(invalid="ignore"):
            whole = np.isfinite(pixels).all() and (pixels == np.trunc(pixels)).all()
        largest = np.abs(pixels).max(initial=0) if whole else np.inf
        if largest < 2**31:
            return pixels.astype(np.int32)
        if largest < 2**53:
            return pixels.astype(np.int64)
    raise IngestError(
        f"Label images hold whole numbers, but {name} has values that aren't."
    )


def _read(image: Image.Image, name: str, limits: SliceLimits) -> np.ndarray:
    """
    One slice's pixels, after checking its decoded size against `limits`
    from its header.
    """
    per_pixel = MODE_BYTES.get(image.mode)
    if per_pixel is None:
        raise IngestError(
            "Label images have one channel of whole numbers, but "
            f"{name} has {image.mode} pixels."
        )
    width, height = image.size
    if width * height * per_pixel > limits.max_decoded_bytes:
        raise IngestError(
            f"Couldn't read {name}: it's {width} × {height} pixels, too large "
            "for one slice on this server."
        )
    try:
        image.load()
        pixels = np.asarray(image)
    except READ_ERRORS as exc:
        raise IngestError(f"Couldn't read {name}: {exc}") from None
    if pixels.ndim != 2:
        raise IngestError(
            f"Label images have one channel of whole numbers, but {name} has more."
        )
    return _whole(pixels, name)


class Planes:
    """
    A label file's slices, read one at a time as (y, x) arrays of whole
    numbers. Every slice must be the size of the first.
    """

    def __init__(self, limits: SliceLimits):
        self.limits = limits
        self._size_yx: tuple[int, int] | None = None

    def __len__(self) -> int:
        raise NotImplementedError

    def name(self, z: int) -> str:
        """How to name slice `z` in a message."""
        raise NotImplementedError

    def _header(self, z: int) -> tuple[int, int]:
        """Slice `z`'s width and height, from its header."""
        raise NotImplementedError

    def _pixels(self, z: int) -> np.ndarray:
        raise NotImplementedError

    @property
    def size_yx(self) -> tuple[int, int]:
        """The size of a slice, from the first one's header."""
        if self._size_yx is None:
            width, height = self._header(0)
            self._size_yx = (height, width)
        return self._size_yx

    def plane(self, z: int) -> np.ndarray:
        pixels = self._pixels(z)
        if pixels.shape != self.size_yx:
            height, width = pixels.shape
            first_height, first_width = self.size_yx
            raise IngestError(
                f"The slices must all be the same size, but {self.name(z)} is "
                f"{width} × {height} pixels and the first is {first_width} × "
                f"{first_height}."
            )
        return pixels

    def close(self) -> None:
        pass

    def __enter__(self) -> "Planes":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ZipPlanes(Planes):
    """
    The slices in a zip, in name order, read in place one member at a time
    (`ml4paleo.ingest` checks the archive is safe to read).
    """

    def __init__(self, fileobj: BinaryIO, limits: SliceLimits = DEFAULT_LIMITS):
        super().__init__(limits)
        self._members = slice_members(open_archive(fileobj), limits)

    def __len__(self) -> int:
        return len(self._members)

    def name(self, z: int) -> str:
        return self._members[z].name

    def _open(self, z: int) -> Image.Image:
        member = self._members[z]
        image = None
        try:
            image = Image.open(member.open(), formats=FORMATS)
            pages = getattr(image, "n_frames", 1)
        except Image.UnidentifiedImageError:
            raise IngestError(f"Couldn't read {member.name}: {UNREADABLE}") from None
        except READ_ERRORS as exc:
            if image is not None:
                image.close()
            raise IngestError(f"Couldn't read {member.name}: {exc}") from None
        if pages > 1:
            image.close()
            raise IngestError(
                f"{member.name} has several pages. Upload a TIFF stack by itself, "
                "or a zip with one slice in each file."
            )
        return image

    def _header(self, z: int) -> tuple[int, int]:
        with self._open(z) as image:
            return image.size

    def _pixels(self, z: int) -> np.ndarray:
        with self._open(z) as image:
            return _read(image, self.name(z), self.limits)


class ImagePlanes(Planes):
    """
    The pages of a TIFF, or a single PNG. Give a TIFF as a local file: Pillow
    reads a whole file to decode one compressed page of anything else.
    """

    def __init__(
        self,
        source: str | os.PathLike[str] | BinaryIO,
        limits: SliceLimits = DEFAULT_LIMITS,
    ):
        super().__init__(limits)
        try:
            self._image = Image.open(source, formats=FORMATS)
        except Image.UnidentifiedImageError:
            raise IngestError(f"Couldn't read the labels file: {UNREADABLE}") from None
        except READ_ERRORS as exc:
            raise IngestError(f"Couldn't read the labels file: {exc}") from None
        try:
            pages = getattr(self._image, "n_frames", 1)
        except READ_ERRORS as exc:
            self._image.close()
            raise IngestError(f"Couldn't read the labels file: {exc}") from None
        # An animated PNG's frames aren't slices.
        self._pages = pages if self._image.format == "TIFF" else 1

    def __len__(self) -> int:
        return self._pages

    def name(self, z: int) -> str:
        return f"page {z + 1}" if self._pages > 1 else "the file"

    def _seek(self, z: int) -> Image.Image:
        try:
            self._image.seek(z)
        except READ_ERRORS as exc:
            raise IngestError(f"Couldn't read {self.name(z)}: {exc}") from None
        return self._image

    def _header(self, z: int) -> tuple[int, int]:
        return self._seek(z).size

    def _pixels(self, z: int) -> np.ndarray:
        return _read(self._seek(z), self.name(z), self.limits)

    def close(self) -> None:
        self._image.close()


def check_size(planes: Planes, shape_zyx: Sequence[int]) -> None:
    """
    Refuse labels that aren't the image's size (z, y, x), from the number of
    slices and the first one's header.
    """
    found = (len(planes), *planes.size_yx)
    if tuple(found) != tuple(shape_zyx):
        raise IngestError(
            f"The labels are {_size(found)} voxels, but the image is "
            f"{_size(shape_zyx)} (x × y × z). Upload labels the size of the image."
        )


def _distinct(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A slice's values and how many pixels hold each."""
    if plane.dtype in (np.uint8, np.uint16):
        counts = np.bincount(plane.ravel())
        values = np.flatnonzero(counts)
        return values, counts[values]
    return np.unique(plane, return_counts=True)


def count_values(
    planes: Planes,
    shape_zyx: Sequence[int] | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict[int, int]:
    """
    Read every slice once and count each value's voxels. Refuses labels that
    aren't `shape_zyx` (before reading them), slices of other sizes, values
    that aren't whole numbers, and more values besides 0 than a project has
    room for as classes.
    """
    if shape_zyx is not None:
        check_size(planes, shape_zyx)
    counts: dict[int, int] = {}
    for z in range(len(planes)):
        values, voxels = _distinct(planes.plane(z))
        for value, count in zip(values.tolist(), voxels.tolist(), strict=True):
            counts[value] = counts.get(value, 0) + count
        if len(counts) - (0 in counts) > MAX_VALUES:
            raise IngestError(TOO_MANY)
        if progress is not None:
            progress((z + 1) / len(planes))
    return counts


class Lookup:
    """
    What each value in a label file becomes: background or a class (stored
    label values 1 to 254). Values it doesn't name stay unlabeled (0).
    """

    def __init__(self, pairs: Iterable[Sequence[int]]):
        table = sorted((int(value), int(label)) for value, label in pairs)
        values = [value for value, _ in table]
        if len(set(values)) != len(values):
            raise ValueError("Each value can become only one label")
        if any(not BACKGROUND <= label <= MAX_CLASS for _, label in table):
            raise ValueError(f"Labels must be {BACKGROUND} to {MAX_CLASS}")
        self.values = np.array(values, dtype=np.int64)
        self.labels = np.array([label for _, label in table], dtype=np.uint8)
        # Most label files are 8- or 16-bit: a table indexed by value.
        self._table: np.ndarray | None = None
        if table and values[0] >= 0 and values[-1] < 2**16:
            self._table = np.zeros(2**16, dtype=np.uint8)
            self._table[self.values] = self.labels

    def __call__(self, plane: np.ndarray) -> np.ndarray:
        """A slice's label values (uint8)."""
        if self._table is not None and plane.dtype in (np.uint8, np.uint16):
            return self._table[plane]
        if not self.values.size:
            return np.zeros(plane.shape, dtype=np.uint8)
        index = np.searchsorted(self.values, plane).clip(0, self.values.size - 1)
        found = self.values[index] == plane
        return np.where(found, self.labels[index], 0).astype(np.uint8)


def chunk_blocks(
    planes: Planes,
    lookup: Lookup,
    shape_zyx: Sequence[int],
    max_bytes: int = DEFAULT_BLOCK_BYTES,
    progress: Callable[[float], None] | None = None,
    check: Callable[[], None] | None = None,
) -> Iterator[tuple[ChunkKey, tuple[int, int, int], np.ndarray]]:
    """
    The labels as label chunks: (key, origin, block of label values) for
    each 64³ chunk that labels any voxel, a chunk deep at a time. Each slice
    is read once, unless a chunk-deep slab of label values wouldn't fit in
    `max_bytes`; then the slab is read in bands of chunk rows, each reading
    its slices again. `check` is called before reading each slice (to stop a
    cancelled job), and `progress` with the fraction of chunks looked at.
    """
    check_size(planes, shape_zyx)
    depth, height, width = (int(n) for n in shape_zyx)
    size_z, size_y, size_x = LABEL_CHUNK_ZYX
    rows, columns = -(-height // size_y), -(-width // size_x)
    total = -(-depth // size_z) * rows * columns
    band = max(1, max_bytes // (size_z * size_y * width))
    done = 0
    for cz in range(-(-depth // size_z)):
        z0, z1 = cz * size_z, min(depth, (cz + 1) * size_z)
        for first_row in range(0, rows, band):
            last_row = min(rows, first_row + band)
            y0, y1 = first_row * size_y, min(height, last_row * size_y)
            slab = np.empty((z1 - z0, y1 - y0, width), dtype=np.uint8)
            for z in range(z0, z1):
                if check is not None:
                    check()
                slab[z - z0] = lookup(planes.plane(z)[y0:y1])
            for cy in range(first_row, last_row):
                for cx in range(columns):
                    done += 1
                    block = slab[
                        :,
                        cy * size_y - y0 : min(height, (cy + 1) * size_y) - y0,
                        cx * size_x : min(width, (cx + 1) * size_x),
                    ]
                    if block.any():
                        yield (cz, cy, cx), (z0, cy * size_y, cx * size_x), block
                    if progress is not None:
                        progress(done / total)


def chunk_delta(
    block: np.ndarray, origin: tuple[int, int, int], only_if: str = "unlabeled"
) -> ChunkDelta:
    """
    The edit that writes a chunk's labels (its nonzero voxels), from
    `chunk_blocks`: one value for the whole delta when there's just one.
    """
    mask = block != 0
    labels = np.unique(block[mask])
    if labels.size == 1:
        deltas = split_into_deltas(mask, origin, value=int(labels[0]), only_if=only_if)
    else:
        deltas = split_into_deltas(mask, origin, values=block, only_if=only_if)
    [delta] = deltas
    return delta
