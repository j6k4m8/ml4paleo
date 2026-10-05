import logging
import pathlib

import numpy as np

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError as e:
    raise ImportError(
        "pydicom was not found. Install pydicom or sync the dicom dependency group."
    ) from e

from .sources import is_source
from .volume_provider import VolumeProvider, normalize_key

log = logging.getLogger(__name__)


def _readable(path):
    """
    What pydicom should read: a path, or an open `SliceSource`.
    """
    return path.open() if is_source(path) else str(path)


class DicomVolumeProvider(VolumeProvider):
    """
    A VolumeProvider backed by one multi-frame DICOM or a series of DICOM files.
    """

    def __init__(
        self,
        path_to_dcms: pathlib.Path | str | list[pathlib.Path],
        dcm_glob: str = "*",
    ):
        self._path: pathlib.Path | None = None
        self._glob = dcm_glob
        self._files: list[pathlib.Path] = []
        self._volume_xyz: np.ndarray | None = None
        self._voxel_size_xyz_mm: tuple[float, float, float] | None = None

        if isinstance(path_to_dcms, list):
            self._load_file_list(path_to_dcms)
        else:
            self._path = pathlib.Path(path_to_dcms)
            if self._path.is_file():
                self._load_single_file(self._path)
            elif self._path.is_dir():
                self._load_directory(self._path)
            else:
                raise ValueError(f"Path does not exist: {self._path}")

    @staticmethod
    def _read_header(path: pathlib.Path):
        return pydicom.dcmread(_readable(path), stop_before_pixels=True)

    @staticmethod
    def _slice_normal(dataset) -> np.ndarray | None:
        """
        Return the unit normal of a slice's plane, from ImageOrientationPatient.
        """
        try:
            orientation = [float(v) for v in dataset.ImageOrientationPatient]
            normal = np.cross(orientation[:3], orientation[3:6])
        except (AttributeError, TypeError, ValueError):
            return None
        length = float(np.linalg.norm(normal))
        return normal / length if length > 0 else None

    @staticmethod
    def _sort_key(
        path: pathlib.Path, dataset, normal: np.ndarray | None = None
    ) -> tuple:
        image_position = getattr(dataset, "ImagePositionPatient", None)
        if image_position is not None and len(image_position) >= 3:
            try:
                position = [float(v) for v in image_position[:3]]
            except (TypeError, ValueError):
                position = None
            if position is not None:
                # Sort by distance along the slice normal, so coronal,
                # sagittal, and oblique series stack in order too. Without an
                # orientation, fall back to the patient z coordinate.
                if normal is None:
                    return (0, position[2], path.name)
                return (0, float(np.dot(position, normal)), path.name)

        instance_number = getattr(dataset, "InstanceNumber", None)
        if instance_number is not None:
            try:
                return (1, int(instance_number), path.name)
            except (TypeError, ValueError):
                pass

        return (2, path.name)

    @staticmethod
    def _pixel_measures(dataset):
        """
        Return the dataset that holds the pixel spacing: the dataset itself,
        or the shared functional group of an enhanced multi-frame DICOM.
        """
        if getattr(dataset, "PixelSpacing", None) is not None:
            return dataset
        shared_groups = getattr(dataset, "SharedFunctionalGroupsSequence", None)
        if shared_groups:
            pixel_measures = getattr(shared_groups[0], "PixelMeasuresSequence", None)
            if pixel_measures:
                return pixel_measures[0]
        return dataset

    @classmethod
    def _in_plane_spacing_xy(cls, dataset) -> tuple[float, float] | None:
        # PixelSpacing is (row spacing, column spacing). Rows are stacked
        # along Y and columns along X, so X spacing is the second value.
        pixel_spacing = getattr(cls._pixel_measures(dataset), "PixelSpacing", None)
        try:
            row_spacing, column_spacing = (float(v) for v in pixel_spacing)
        except (TypeError, ValueError):
            return None
        if row_spacing <= 0 or column_spacing <= 0:
            return None
        return column_spacing, row_spacing

    @classmethod
    def _nominal_slice_spacing(cls, dataset) -> float | None:
        for source in (dataset, cls._pixel_measures(dataset)):
            for attribute in ("SpacingBetweenSlices", "SliceThickness"):
                try:
                    spacing = float(getattr(source, attribute, None))
                except (TypeError, ValueError):
                    continue
                if spacing > 0:
                    return spacing
        return None

    @staticmethod
    def _slice_spacing_from_positions(headers) -> float | None:
        """
        Return the median distance between sorted slices along the slice
        normal, or None if any slice lacks a position.
        """
        try:
            positions = np.array(
                [
                    [float(v) for v in header.ImagePositionPatient[:3]]
                    for _, header in headers
                ]
            )
            orientation = [float(v) for v in headers[0][1].ImageOrientationPatient]
            normal = np.cross(orientation[:3], orientation[3:6])
        except (AttributeError, TypeError, ValueError):
            return None
        if len(positions) < 2:
            return None
        gaps = np.abs(np.diff(positions @ normal))
        gaps = gaps[gaps > 0]
        return float(np.median(gaps)) if gaps.size else None

    @classmethod
    def _voxel_size(
        cls, dataset, slice_spacing: float | None = None
    ) -> tuple[float, float, float] | None:
        in_plane_spacing = cls._in_plane_spacing_xy(dataset)
        if slice_spacing is None:
            slice_spacing = cls._nominal_slice_spacing(dataset)
        if in_plane_spacing is None or slice_spacing is None:
            return None
        return (*in_plane_spacing, slice_spacing)

    def _load_single_file(self, dicom_path: pathlib.Path) -> None:
        dataset = pydicom.dcmread(_readable(dicom_path))
        pixel_array = dataset.pixel_array

        if pixel_array.ndim == 2:
            volume_xyz = pixel_array.T[:, :, np.newaxis]
        elif pixel_array.ndim == 3 and getattr(dataset, "SamplesPerPixel", 1) == 1:
            # Multi-frame grayscale DICOMs are typically (frames, rows, cols).
            volume_xyz = np.transpose(pixel_array, (2, 1, 0))
        else:
            raise ValueError(
                f"Unsupported DICOM pixel array shape {pixel_array.shape} for {dicom_path}."
            )

        self._ds = dataset
        self._dtype = np.dtype(volume_xyz.dtype)
        self._shape_xyz = volume_xyz.shape
        self._files = [dicom_path]
        self._volume_xyz = volume_xyz
        self._voxel_size_xyz_mm = self._voxel_size(dataset)

    def _load_file_list(self, dicom_files: list[pathlib.Path]) -> None:
        if len(dicom_files) == 0:
            raise ValueError("No DICOM files were provided.")

        self._files = [
            path if is_source(path) else pathlib.Path(path) for path in dicom_files
        ]
        if len(self._files) == 1:
            self._load_single_file(self._files[0])
            return

        self._load_headers_for_paths(self._files, source_label="provided file list")

    def _load_directory(self, dicom_dir: pathlib.Path) -> None:
        headers = []
        for path in sorted(dicom_dir.glob(self._glob)):
            if not path.is_file():
                continue
            try:
                dataset = self._read_header(path)
            except InvalidDicomError:
                continue
            except Exception:
                log.debug(
                    "Skipping unreadable file while scanning DICOM input: %s", path
                )
                continue
            headers.append((path, dataset))

        if len(headers) == 0:
            raise ValueError(f"No DICOM files found in {dicom_dir}.")

        # If the directory contains a single multi-frame DICOM, treat it like a
        # single-file upload instead of a one-slice series.
        if len(headers) == 1 and int(getattr(headers[0][1], "NumberOfFrames", 1)) > 1:
            self._load_single_file(headers[0][0])
            return

        self._load_headers(headers, source_label=str(dicom_dir))

    def _load_headers_for_paths(
        self, dicom_paths: list[pathlib.Path], source_label: str
    ) -> None:
        headers = []
        for path in dicom_paths:
            try:
                dataset = self._read_header(path)
            except InvalidDicomError as exc:
                raise ValueError(
                    f"File {path.name} is not a valid DICOM file."
                ) from exc
            headers.append((path, dataset))

        if len(headers) == 1 and int(getattr(headers[0][1], "NumberOfFrames", 1)) > 1:
            self._load_single_file(headers[0][0])
            return

        self._load_headers(headers, source_label=source_label)

    def _load_headers(self, headers, source_label: str) -> None:
        grouped_headers = {}
        for path, dataset in headers:
            series_uid = getattr(dataset, "SeriesInstanceUID", None) or "__missing__"
            grouped_headers.setdefault(series_uid, []).append((path, dataset))

        if len(grouped_headers) > 1:
            largest_series_size = max(len(items) for items in grouped_headers.values())
            largest_series = [
                (series_uid, items)
                for series_uid, items in grouped_headers.items()
                if len(items) == largest_series_size
            ]
            if len(largest_series) > 1:
                raise ValueError(
                    f"Found multiple DICOM series of equal size in {source_label}; upload one series per job."
                )
            selected_series_uid, headers = largest_series[0]
            log.warning(
                "Found multiple DICOM series in %s; using the largest series %s (%d files).",
                source_label,
                selected_series_uid,
                len(headers),
            )
        else:
            headers = next(iter(grouped_headers.values()))

        normal = self._slice_normal(headers[0][1])
        headers = sorted(
            headers, key=lambda item: self._sort_key(item[0], item[1], normal)
        )
        self._files = [path for path, _ in headers]

        dataset = pydicom.dcmread(_readable(self._files[0]))
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)

        for _, header in headers[1:]:
            if (
                int(getattr(header, "Rows", rows)) != rows
                or int(getattr(header, "Columns", cols)) != cols
            ):
                raise ValueError(
                    f"DICOM series in {source_label} has inconsistent slice dimensions."
                )
            if int(getattr(header, "NumberOfFrames", 1)) > 1:
                raise ValueError(
                    f"DICOM source {source_label} contains multi-frame files mixed into a series upload."
                )

        self._ds = dataset
        self._dtype = np.dtype(dataset.pixel_array.dtype)
        self._shape_xyz = (cols, rows, len(self._files))
        self._voxel_size_xyz_mm = self._voxel_size(
            dataset, self._slice_spacing_from_positions(headers)
        )

    def __getitem__(self, key):
        zs, ys, xs = normalize_key(key, self.shape[::-1])
        return self._get_subvolume(xs, ys, zs)

    def _read_slice_xyz(self, z_index: int) -> np.ndarray:
        if self._volume_xyz is not None:
            return self._volume_xyz[:, :, z_index]
        pixels = pydicom.dcmread(_readable(self._files[z_index])).pixel_array
        if pixels.dtype != self._dtype:
            # Reading into a preallocated array would cast silently.
            raise ValueError(
                f"DICOM slice {self._files[z_index]} has pixel type {pixels.dtype}, "
                f"but the first slice has {self._dtype}."
            )
        return pixels.T

    def _get_subvolume(self, xs, ys, zs):
        if self._volume_xyz is not None:
            return self._volume_xyz[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]]

        vol = np.empty((xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]), dtype=self.dtype)
        for i, z in enumerate(range(zs[0], zs[1])):
            vol[:, :, i] = self._read_slice_xyz(z)[xs[0] : xs[1], ys[0] : ys[1]]
        return vol

    @property
    def files(self) -> list:
        """
        The slices in stacking order (paths, or `SliceSource`s).
        """
        return list(self._files)

    @property
    def shape(self):
        return self._shape_xyz

    @property
    def dtype(self):
        return self._dtype

    @property
    def voxel_size_xyz_mm(self) -> tuple[float, float, float] | None:
        return self._voxel_size_xyz_mm
