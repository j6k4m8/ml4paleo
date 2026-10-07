"""
Ingest: turn an uploaded archive into an OME-Zarr image.

An upload is one zip archive holding either a stack of 2D images (one per z
slice, ordered by name, with numbers compared as numbers) or a DICOM series
(ordered along the slice normal). Archives are read in place from storage
(`ml4paleo.storage.open_object`) one member at a time, so ingest never
downloads or unpacks the whole file, and parallel jobs each read only their
own slices.

`probe` looks at an archive once and records what it found, including the
slices in stacking order, as a `SourceIndex`; `slab_provider` then gives the
jobs that copy slabs of slices a volume provider without looking again.

Problems with the upload itself raise `IngestError`, with a message for the
person who uploaded it.
"""

import io
import json
import re
import struct
import zipfile
import zlib
from dataclasses import asdict, dataclass
from typing import BinaryIO, Literal

import numpy as np

from .volume_providers.imagevp import ImageStackVolumeProvider
from .volume_providers.volume_provider import VolumeProvider, normalize_key

MAX_MEMBERS = 200_000
# Archives that expand by more than this look like zip bombs.
MAX_EXPANSION = 1000
# Python decompresses bzip2 and LZMA members without bounding the output, so
# only these methods are read.
SUPPORTED_COMPRESSION = {zipfile.ZIP_STORED: "stored", zipfile.ZIP_DEFLATED: "deflate"}
READ_CHUNK = 1024 * 1024
SourceKind = Literal["images", "dicom"]


@dataclass(frozen=True)
class SliceLimits:
    """
    How large one slice may be: its bytes in the archive (held in memory
    while it is decoded) and its decoded pixels. Reading a slice takes about
    the member plus twice the decoded size at the peak.
    """

    max_member_bytes: int
    max_decoded_bytes: int

    @classmethod
    def for_memory(cls, budget_bytes: int) -> "SliceLimits":
        """
        Limits that keep one slice within a job's memory budget, between
        64 MiB and 2 GiB each.
        """
        share = min(2 * 1024**3, max(64 * 1024**2, budget_bytes // 4))
        return cls(max_member_bytes=share, max_decoded_bytes=share)


DEFAULT_LIMITS = SliceLimits.for_memory(4 * 1024**3)


def _megabytes(size: int) -> str:
    return f"{size / 1024**2:,.0f} MB"


class IngestError(ValueError):
    """
    Something is wrong with the uploaded file; retrying won't help.
    """


def ignored(name: str) -> bool:
    """
    Files that come along in archives but aren't slices: macOS resource forks
    and hidden files.
    """
    parts = [p for p in re.split(r"[/\\]", name) if p]
    return (
        "__MACOSX" in parts
        or any(part.startswith(".") for part in parts)
        or parts[-1:] == ["Thumbs.db"]
    )


def natural_key(name: str) -> list:
    """
    Sort "slice_2" before "slice_10".
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


@dataclass(frozen=True)
class ZipMember:
    """
    One archive member, as a `SliceSource` for the volume providers.
    """

    archive: zipfile.ZipFile
    info: zipfile.ZipInfo

    @property
    def name(self) -> str:
        return self.info.filename

    def open(self) -> BinaryIO:
        """
        Read the member into memory, in bounded chunks: zipfile stops at the
        declared size, and small reads keep each decompression step small.
        """
        data = io.BytesIO()
        try:
            with self.archive.open(self.info) as member:
                while chunk := member.read(READ_CHUNK):
                    data.write(chunk)
        except (
            zipfile.BadZipFile,
            zlib.error,
            EOFError,
            RuntimeError,
            ValueError,
            NotImplementedError,
            struct.error,
        ) as exc:
            raise IngestError(f"The archive is damaged ({self.name}: {exc}).") from None
        data.seek(0)
        return data

    def __str__(self) -> str:
        return self.name


NOT_A_ZIP = (
    "The upload is not a zip archive. Put the slices (image files or a DICOM "
    "series) in one zip file."
)


def entry_count(fileobj: BinaryIO) -> int | None:
    """
    The number of entries an archive says it has, from its end-of-directory
    record, without reading the directory itself (which could be huge).
    None if there is no such record.
    """
    fileobj.seek(0, io.SEEK_END)
    size = fileobj.tell()
    fileobj.seek(max(0, size - (65536 + 22)))
    tail = fileobj.read()
    end = tail.rfind(b"PK\x05\x06")
    if end < 0 or len(tail) < end + 22:
        return None
    count = struct.unpack("<H", tail[end + 10 : end + 12])[0]
    if count != 0xFFFF:
        return count
    # A ZIP64 archive: the real count is in the ZIP64 end record.
    locator = tail.rfind(b"PK\x06\x07", 0, end)
    if locator < 0 or len(tail) < locator + 20:
        return None
    record = struct.unpack("<Q", tail[locator + 8 : locator + 16])[0]
    fileobj.seek(record)
    header = fileobj.read(56)
    if len(header) < 40 or header[:4] != b"PK\x06\x06":
        return None
    return struct.unpack("<Q", header[32:40])[0]


def open_archive(fileobj: BinaryIO) -> zipfile.ZipFile:
    count = entry_count(fileobj)
    if count is None:
        raise IngestError(NOT_A_ZIP)
    if count > MAX_MEMBERS:
        raise IngestError(f"The archive has more than {MAX_MEMBERS} files.")
    fileobj.seek(0)
    try:
        return zipfile.ZipFile(fileobj)
    except zipfile.BadZipFile:
        raise IngestError(NOT_A_ZIP) from None
    except (ValueError, EOFError, NotImplementedError, struct.error) as exc:
        # For example a file name that isn't the UTF-8 it says it is.
        raise IngestError(f"The archive is damaged ({exc}).") from None


def slice_members(
    archive: zipfile.ZipFile, limits: SliceLimits = DEFAULT_LIMITS
) -> list[ZipMember]:
    """
    The archive's slices in name order, after checking the archive is safe
    to read.
    """
    files = [info for info in archive.infolist() if not info.is_dir()]
    # Check every name, including ones skipped below: a name that climbs out
    # of the archive means it wasn't made innocently.
    for info in files:
        parts = re.split(r"[/\\]", info.filename)
        if info.filename.startswith(("/", "\\")) or ".." in parts:
            raise IngestError(f"The archive has an unsafe file name: {info.filename!r}")
    infos = [info for info in files if not ignored(info.filename)]
    if not infos:
        raise IngestError("The archive has no slices in it.")
    if len(infos) > MAX_MEMBERS:
        raise IngestError(f"The archive has more than {MAX_MEMBERS} files.")
    for info in infos:
        if info.file_size > limits.max_member_bytes:
            raise IngestError(
                f"{info.filename} is {_megabytes(info.file_size)}; this server "
                f"reads slices of up to {_megabytes(limits.max_member_bytes)}."
            )
        if info.flag_bits & 0x1:
            raise IngestError("The archive is encrypted; upload it without a password.")
        if info.compress_type not in SUPPORTED_COMPRESSION:
            raise IngestError(
                f"{info.filename} uses a compression method ml4paleo doesn't read; "
                "make the zip with ordinary (deflate) compression."
            )
    expanded = sum(info.file_size for info in infos)
    compressed = sum(info.compress_size for info in infos) or 1
    if expanded / compressed > MAX_EXPANSION:
        raise IngestError("The archive expands too much to be a scan.")
    infos.sort(key=lambda info: natural_key(info.filename))
    return [ZipMember(archive, info) for info in infos]


@dataclass(frozen=True)
class SourceIndex:
    """
    What `probe` found: the kind of slices, their stacking order, and the
    volume they make.
    """

    kind: SourceKind
    members: list[str]
    shape_xyz: tuple[int, int, int]
    dtype: str
    voxel_size_zyx: tuple[float, float, float] | None
    unit: str | None

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, data: bytes) -> "SourceIndex":
        raw = json.loads(data)
        return cls(
            kind=raw["kind"],
            members=list(raw["members"]),
            shape_xyz=tuple(raw["shape_xyz"]),
            dtype=raw["dtype"],
            voxel_size_zyx=tuple(raw["voxel_size_zyx"])
            if raw["voxel_size_zyx"]
            else None,
            unit=raw["unit"],
        )


def _decoded_dicom_bytes(dataset) -> int:
    """
    How much memory a DICOM dataset's pixels take once decoded, from its
    header.
    """
    rows, columns = (
        int(getattr(dataset, "Rows", 0)),
        int(getattr(dataset, "Columns", 0)),
    )
    frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    per_sample = -(-int(getattr(dataset, "BitsAllocated", 16) or 16) // 8)
    return rows * columns * frames * samples * per_sample


def _check_dicom_size(dataset, name: str, limits: SliceLimits) -> None:
    decoded = _decoded_dicom_bytes(dataset)
    if decoded > limits.max_decoded_bytes:
        raise IngestError(
            f"{name} decodes to {_megabytes(decoded)}; this server reads slices "
            f"of up to {_megabytes(limits.max_decoded_bytes)}."
        )


def _is_dicom(member: ZipMember) -> bool:
    import pydicom
    from pydicom.errors import InvalidDicomError

    data = member.open()  # IngestError if the archive is damaged
    try:
        pydicom.dcmread(data, stop_before_pixels=True)
    except (InvalidDicomError, EOFError, ValueError, TypeError, OSError):
        return False
    return True


def probe(fileobj: BinaryIO, limits: SliceLimits = DEFAULT_LIMITS) -> SourceIndex:
    """
    Look at an archive and work out the volume in it, refusing slices too
    large for `limits` before decoding them.
    """
    members = slice_members(open_archive(fileobj), limits)
    if _is_dicom(members[0]):
        from .volume_providers.dicomvp import DicomVolumeProvider

        if not all(_is_dicom(member) for member in members[1:]):
            raise IngestError(
                "The archive mixes DICOM files with other files. Upload one "
                "kind of slice per archive."
            )
        import pydicom

        for member in members:
            header = pydicom.dcmread(member.open(), stop_before_pixels=True)
            _check_dicom_size(header, member.name, limits)
        try:
            provider = DicomVolumeProvider(members)  # type: ignore[arg-type]
        except ValueError as exc:
            raise IngestError(str(exc)) from exc
        spacing = provider.voxel_size_xyz_mm
        return SourceIndex(
            kind="dicom",
            members=[member.name for member in provider.files],
            shape_xyz=_shape(provider),
            dtype=np.dtype(provider.dtype).str,
            voxel_size_zyx=tuple(spacing[::-1]) if spacing else None,  # type: ignore[arg-type]
            unit="millimeter" if spacing else None,
        )
    try:
        provider = ImageStackVolumeProvider(
            members,  # type: ignore[arg-type]
            max_decoded_bytes=limits.max_decoded_bytes,
        )
    except ValueError as exc:
        raise IngestError(str(exc)) from exc
    return SourceIndex(
        kind="images",
        members=[member.name for member in members],
        shape_xyz=_shape(provider),
        dtype=np.dtype(provider.dtype).str,
        voxel_size_zyx=None,
        unit=None,
    )


def _shape(provider: VolumeProvider) -> tuple[int, int, int]:
    x, y, z = (int(s) for s in provider.shape)
    return (x, y, z)


def slab_provider(
    fileobj: BinaryIO, index: SourceIndex, limits: SliceLimits = DEFAULT_LIMITS
) -> VolumeProvider:
    """
    A provider over the archive's slices in stacking order, reading each
    slice only when it is asked for.
    """
    archive = open_archive(fileobj)
    by_name = {info.filename: info for info in archive.infolist()}
    try:
        members = [ZipMember(archive, by_name[name]) for name in index.members]
    except KeyError as exc:
        raise IngestError(f"The archive changed: {exc.args[0]} is missing.") from None
    for name, member in zip(index.members, members, strict=True):
        if member.info.file_size > limits.max_member_bytes:
            raise IngestError(
                f"{name} is {_megabytes(member.info.file_size)}; this server reads "
                f"slices of up to {_megabytes(limits.max_member_bytes)}."
            )
    if index.kind == "images":
        return ImageStackVolumeProvider(
            members,  # type: ignore[arg-type]
            max_decoded_bytes=limits.max_decoded_bytes,
        )
    if len(members) == 1 and index.shape_xyz[2] > 1:
        return _DicomFrames(members[0], index.shape_xyz, np.dtype(index.dtype), limits)
    return _DicomSlices(members, index.shape_xyz, np.dtype(index.dtype), limits)


class _DicomSlices(VolumeProvider):
    """
    DICOM slices in a known order, read one at a time (without reading every
    header again, as sorting a series needs).
    """

    def __init__(
        self,
        members: list[ZipMember],
        shape_xyz: tuple[int, int, int],
        dtype: np.dtype,
        limits: SliceLimits = DEFAULT_LIMITS,
    ):
        self._members = members
        self._shape_xyz = shape_xyz
        self._dtype = dtype
        self._limits = limits

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape_xyz

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, key) -> np.ndarray:
        import pydicom

        zs, ys, xs = normalize_key(key, self._shape_xyz[::-1])
        volume = np.empty(
            (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]), dtype=self._dtype
        )
        for i, z in enumerate(range(zs[0], zs[1])):
            member = self._members[z]
            dataset = pydicom.dcmread(member.open())
            _check_dicom_size(dataset, member.name, self._limits)
            pixels = dataset.pixel_array.T
            if pixels.shape != self._shape_xyz[:2] or pixels.dtype != self._dtype:
                raise IngestError(
                    f"DICOM slice {member.name} is {pixels.shape} {pixels.dtype}, "
                    f"unlike the rest of the series."
                )
            volume[:, :, i] = pixels[xs[0] : xs[1], ys[0] : ys[1]]
        return volume


class _DicomFrames(VolumeProvider):
    """
    One multi-frame DICOM file, read once (within the slice limits).
    """

    def __init__(
        self,
        member: ZipMember,
        shape_xyz: tuple[int, int, int],
        dtype: np.dtype,
        limits: SliceLimits = DEFAULT_LIMITS,
    ):
        self._limits = limits
        self._member = member
        self._shape_xyz = shape_xyz
        self._dtype = dtype
        self._volume_xyz: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape_xyz

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, key) -> np.ndarray:
        import pydicom

        if self._volume_xyz is None:
            dataset = pydicom.dcmread(self._member.open())
            _check_dicom_size(dataset, self._member.name, self._limits)
            frames = dataset.pixel_array
            volume = np.transpose(frames, (2, 1, 0))
            if volume.shape != self._shape_xyz or volume.dtype != self._dtype:
                raise IngestError(f"The DICOM file {self._member.name} changed.")
            self._volume_xyz = volume
        zs, ys, xs = normalize_key(key, self._shape_xyz[::-1])
        return self._volume_xyz[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]]


def intensity_summary(values: np.ndarray, bins: int = 256) -> dict:
    """
    A histogram and a default display window (the 0.5th to 99.5th
    percentiles) for an image, from a sample of its values (for example its
    coarsest pyramid level).
    """
    values = np.asarray(values).ravel()
    if values.dtype == bool:
        values = values.astype(np.uint8)
    if np.issubdtype(values.dtype, np.floating):
        # NaN and infinity (for example outside a specimen) have no place on
        # a display scale.
        values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": 0, "max": 0, "window": [0, 0], "histogram": []}
    low, high = float(values.min()), float(values.max())
    window = [float(v) for v in np.percentile(values, [0.5, 99.5])]
    counts, edges = np.histogram(values, bins=bins, range=(low, high or 1))
    return {
        "min": low,
        "max": high,
        "window": window,
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }
