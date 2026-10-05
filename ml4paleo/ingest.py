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
import zipfile
from dataclasses import asdict, dataclass
from typing import BinaryIO, Literal

import numpy as np

from .volume_providers.imagevp import ImageStackVolumeProvider
from .volume_providers.volume_provider import VolumeProvider, normalize_key

MAX_MEMBERS = 200_000
MAX_SLICE_BYTES = 4 * 1024**3
# Archives that expand by more than this look like zip bombs.
MAX_EXPANSION = 1000
SourceKind = Literal["images", "dicom"]


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
        # zipfile stops at the member's declared size and checks its CRC.
        with self.archive.open(self.info) as member:
            return io.BytesIO(member.read())

    def __str__(self) -> str:
        return self.name


def open_archive(fileobj: BinaryIO) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(fileobj)
    except zipfile.BadZipFile:
        raise IngestError(
            "The upload is not a zip archive. Put the slices (image files or a "
            "DICOM series) in one zip file."
        ) from None


def slice_members(archive: zipfile.ZipFile) -> list[ZipMember]:
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
        if info.file_size > MAX_SLICE_BYTES:
            raise IngestError(
                f"{info.filename} is larger than {MAX_SLICE_BYTES} bytes."
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


def _is_dicom(member: ZipMember) -> bool:
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        pydicom.dcmread(member.open(), stop_before_pixels=True)
    except (InvalidDicomError, EOFError, ValueError, TypeError, OSError):
        return False
    return True


def probe(fileobj: BinaryIO) -> SourceIndex:
    """
    Look at an archive and work out the volume in it.
    """
    members = slice_members(open_archive(fileobj))
    if _is_dicom(members[0]):
        from .volume_providers.dicomvp import DicomVolumeProvider

        if not all(_is_dicom(member) for member in members[1:]):
            raise IngestError(
                "The archive mixes DICOM files with other files. Upload one "
                "kind of slice per archive."
            )
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
        provider = ImageStackVolumeProvider(members)  # type: ignore[arg-type]
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


def slab_provider(fileobj: BinaryIO, index: SourceIndex) -> VolumeProvider:
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
    if index.kind == "images":
        return ImageStackVolumeProvider(members)  # type: ignore[arg-type]
    return _DicomSlices(members, index.shape_xyz, np.dtype(index.dtype))


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
    ):
        self._members = members
        self._shape_xyz = shape_xyz
        self._dtype = dtype

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
            pixels = pydicom.dcmread(member.open()).pixel_array.T
            if pixels.shape != self._shape_xyz[:2] or pixels.dtype != self._dtype:
                raise IngestError(
                    f"DICOM slice {member.name} is {pixels.shape} {pixels.dtype}, "
                    f"unlike the rest of the series."
                )
            volume[:, :, i] = pixels[xs[0] : xs[1], ys[0] : ys[1]]
        return volume


def intensity_summary(values: np.ndarray, bins: int = 256) -> dict:
    """
    A histogram and a default display window (the 0.5th to 99.5th
    percentiles) for an image, from a sample of its values (for example its
    coarsest pyramid level).
    """
    values = np.asarray(values).ravel()
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
