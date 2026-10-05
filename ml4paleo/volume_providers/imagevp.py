import pathlib

import numpy as np
from PIL import Image

from .volume_provider import VolumeProvider, normalize_key


class ImageStackVolumeProvider(VolumeProvider):
    """
    A VolumeProvider that provides a 3D volume of data from a stack of 2D
    images on disk.

    Each image is one z-slice. Image columns run along X and rows along Y, so
    `provider[x, y, z]` is the pixel at column x, row y of image z.
    """

    def __init__(
        self,
        path_or_list_of_images: pathlib.Path | list[pathlib.Path],
        image_glob: str = "*",
        cache_size: int | str = 0,
    ):
        """
        Create a new ImageStackVolumeProvider.

        If a directory is provided, images will be loaded from the directory
        using the provided glob pattern. The images will be sorted by filename.
        If a list of paths is provided, the images will be loaded in the order
        they are provided.

        Arguments:
            path (pathlib.Path | List[pathlib.Path]): The path to the directory
                containing the images, or a list of paths to the images.
            image_glob (str): A glob pattern to match the image files against,
                if path is a directory. Defaults to "*".
            cache_size: Unused; kept so existing callers keep working.

        Raises:
            ValueError: If the path is not a directory, the list is empty, or
                the first image cannot be read.

        """
        if isinstance(path_or_list_of_images, pathlib.Path):
            if not path_or_list_of_images.is_dir():
                raise ValueError(
                    f"Path must be a directory, but got '{path_or_list_of_images}'."
                )
            self.paths = list(path_or_list_of_images.glob(image_glob))
            self.paths.sort()
        else:
            self.paths = list(path_or_list_of_images)
        if len(self.paths) == 0:
            raise ValueError("No images found.")

        # Read the first slice once, so shape and dtype never need to reopen
        # files (and a bad first file fails here, not deep inside a job).
        first = _read_slice(self.paths[0])
        self._shape_xy: tuple[int, int] = (int(first.shape[0]), int(first.shape[1]))
        self._dtype = first.dtype

    def _read_image(self, path: pathlib.Path) -> np.ndarray:
        """
        Read one slice as an (x, y) array and check it matches the first slice.

        v1 silently replaced unreadable or mismatched slices with zeros, which
        corrupted volumes without any error.
        """
        res = _read_slice(path)
        if res.shape != self._shape_xy:
            raise ValueError(
                f"Image slice {path} has size {res.shape} (x, y), but the first "
                f"slice has size {self._shape_xy}."
            )
        if res.dtype != self._dtype:
            raise ValueError(
                f"Image slice {path} has pixel type {res.dtype}, but the first "
                f"slice has {self._dtype}."
            )
        return res

    @property
    def shape(self) -> tuple[int, int, int]:
        return (*self._shape_xy, len(self.paths))

    def __getitem__(self, key):
        """
        Get a 3D subvolume of the data from the given indices.

        Note that this method can be quite slow if the slice is "deep" in Z
        and small in XY, since it reads whole slices.

        Arguments:
            key (tuple): The indices to slice.

        """
        # Normalize the indices
        zs, ys, xs = normalize_key(key, self.shape[::-1])

        # Read the images.
        images = [self._read_image(self.paths[z]) for z in range(zs[0], zs[1])]

        # Return the subvolume.
        vol = np.stack(images, axis=-1)
        return vol[xs[0] : xs[1], ys[0] : ys[1], :]

    @property
    def dtype(self) -> np.dtype:
        return self._dtype


def _read_slice(path: pathlib.Path) -> np.ndarray:
    """
    Read one image file as an (x, y) array, keeping the first channel of
    multichannel images. Palette images keep their palette indices, which is
    what label images saved with a palette mean.

    Raises:
        ValueError: If the file is not a readable image.
    """
    try:
        with Image.open(path) as image:
            res = np.array(image).T
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read image slice {path}: {exc}") from exc
    # If dim is CHW (an RGB or RGBA image), keep the first channel.
    if len(res.shape) == 3:
        res = res[0]
    return res
