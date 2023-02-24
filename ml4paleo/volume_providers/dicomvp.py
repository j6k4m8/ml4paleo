from typing import Union

import numpy as np

try:
    import pydicom
except ImportError as e:
    raise ImportError("pydicom was not found. Try `pip install ml4paleo[dicom]`") from e
import pathlib

from .volume_provider import VolumeProvider, normalize_key


class DicomVolumeProvider(VolumeProvider):
    """
    A VolumeProvider that proxies subvolume requests to DCM files on disk.

    """

    def __init__(self, path_to_dcms: Union[pathlib.Path, str], dcm_glob: str = "*"):
        """
        Create a new DICOM-proxy volume provider.

        Note that unlike reading the DICOMs directly, this will read them and
        proxy cutout queries in XYZ order, so this is NOT the same as reading
        the volume directly with something like PyDicom.

        Arguments:
            path_to_dcms (Union[pathlib.Path, str]): The path to the .dcm files
                on disk (a directory).
            dcm_glob (str): An optional file glob pattern, in case you have
                multiple DICOM volumes in the same directory (eek).

        """
        self._path = path_to_dcms
        self._glob = dcm_glob

        self._files = list(pathlib.Path(self._path).glob(self._glob))
        self._files.sort()
        self._files = [str(f) for f in self._files]
        self._ds = pydicom.dcmread(self._files[0])
        self._shape_xyz = (self._ds.Rows, self._ds.Columns, len(self._files))

    def __getitem__(self, key):
        zs, ys, xs = normalize_key(key, self.shape[::-1])
        return self._get_subvolume(xs, ys, zs)

    def _get_subvolume(self, xs, ys, zs):
        # Return an XYZ-subvolume.
        slices = []
        for z in range(zs[0], zs[1]):
            slices.append(
                pydicom.dcmread(self._files[z]).pixel_array[
                    xs[0] : xs[1], ys[0] : ys[1]
                ]
            )
        return np.stack(slices, axis=-1)

    @property
    def shape(self):
        return self._shape_xyz

    @property
    def dtype(self):
        return np.dtype(self._ds.pixel_array.dtype)
