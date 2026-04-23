import logging
import pathlib
from typing import Optional, Union

import numpy as np

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError as e:
    raise ImportError(
        "pydicom was not found. Install pydicom or sync the dicom dependency group."
    ) from e

from .volume_provider import VolumeProvider, normalize_key

log = logging.getLogger(__name__)


class DicomVolumeProvider(VolumeProvider):
    """
    A VolumeProvider backed by one multi-frame DICOM or a series of DICOM files.
    """

    def __init__(
        self,
        path_to_dcms: Union[pathlib.Path, str, list[pathlib.Path]],
        dcm_glob: str = "*",
    ):
        self._path: Optional[pathlib.Path] = None
        self._glob = dcm_glob
        self._files: list[pathlib.Path] = []
        self._volume_xyz: Optional[np.ndarray] = None

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
        return pydicom.dcmread(str(path), stop_before_pixels=True)

    @staticmethod
    def _sort_key(path: pathlib.Path, dataset) -> tuple:
        image_position = getattr(dataset, "ImagePositionPatient", None)
        if image_position is not None and len(image_position) >= 3:
            try:
                return (0, float(image_position[2]), path.name)
            except (TypeError, ValueError):
                pass

        instance_number = getattr(dataset, "InstanceNumber", None)
        if instance_number is not None:
            try:
                return (1, int(instance_number), path.name)
            except (TypeError, ValueError):
                pass

        return (2, path.name)

    def _load_single_file(self, dicom_path: pathlib.Path) -> None:
        dataset = pydicom.dcmread(str(dicom_path))
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

    def _load_file_list(self, dicom_files: list[pathlib.Path]) -> None:
        if len(dicom_files) == 0:
            raise ValueError("No DICOM files were provided.")

        self._files = [pathlib.Path(path) for path in dicom_files]
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
                log.debug("Skipping unreadable file while scanning DICOM input: %s", path)
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
                raise ValueError(f"File {path.name} is not a valid DICOM file.") from exc
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

        headers = sorted(headers, key=lambda item: self._sort_key(item[0], item[1]))
        self._files = [path for path, _ in headers]

        dataset = pydicom.dcmread(str(self._files[0]))
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)

        for _, header in headers[1:]:
            if int(getattr(header, "Rows", rows)) != rows or int(
                getattr(header, "Columns", cols)
            ) != cols:
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

    def __getitem__(self, key):
        zs, ys, xs = normalize_key(key, self.shape[::-1])
        return self._get_subvolume(xs, ys, zs)

    def _read_slice_xyz(self, z_index: int) -> np.ndarray:
        if self._volume_xyz is not None:
            return self._volume_xyz[:, :, z_index]
        return pydicom.dcmread(str(self._files[z_index])).pixel_array.T

    def _get_subvolume(self, xs, ys, zs):
        if self._volume_xyz is not None:
            return self._volume_xyz[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]]

        slices = []
        for z in range(zs[0], zs[1]):
            slices.append(self._read_slice_xyz(z)[xs[0] : xs[1], ys[0] : ys[1]])
        return np.stack(slices, axis=-1)

    @property
    def shape(self):
        return self._shape_xyz

    @property
    def dtype(self):
        return self._dtype
