import pathlib
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import psutil
from PIL import Image

from .volume_provider import VolumeProvider, normalize_key


class ImageStackVolumeProvider(VolumeProvider):
    """
    A VolumeProvider that provides a 3D volume of data from a stack of 2D
    images on disk.
    """

    def __init__(
        self,
        path_or_list_of_images: Union[pathlib.Path, List[pathlib.Path]],
        image_glob: str = "*",
        cache_size: Union[int, str] = "guess",
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
            cache_size (int): The size of the cache to use, in number of stored
                images. If 0, no cache will be used. If "guess", a reasonable
                default will be chosen based upon the size of the images and
                available memory (will use 50% of available memory). Defaults
                to "guess".

        Raises:
            ValueError: If the path is not a directory or list is empty.

        """
        if isinstance(path_or_list_of_images, pathlib.Path):
            if not path_or_list_of_images.is_dir():
                raise ValueError(
                    f"Path must be a directory, but got '{path_or_list_of_images}'."
                )
            self.paths = list(path_or_list_of_images.glob(image_glob))
            self.paths.sort()
        else:
            self.paths = path_or_list_of_images
        if len(self.paths) == 0:
            raise ValueError("No images found.")

        # Calculate the cache size.
        if cache_size == "guess":
            # Calculate the size of the images.
            image_size = Image.open(self.paths[0]).size
            image_size = image_size[0] * image_size[1] * 4
            # Multiply by datatype size;
            dtype = np.dtype(np.float32)
            image_size *= dtype.itemsize
            # Calculate the size of the cache.
            cache_size = int(psutil.virtual_memory().available / 2 / image_size)
        elif isinstance(cache_size, str):
            raise ValueError(
                f"Invalid cache size: {cache_size}. Must be an integer or 'guess'."
            )
        self._cache_size = cache_size

        # Decorate the read function with a cache.
        # self._read_image = lru_cache(maxsize=self._cache_size)(self._read_image)

    def _read_image(self, path: pathlib.Path) -> np.ndarray:
        """
        Read an image from disk.

        Arguments:
            path (pathlib.Path): The path to the image to read.

        Returns:
            np.ndarray: The image data.

        """
        try:
            res = np.array(Image.open(path)).T
        except:
            res = np.zeros(self.shape[:2], dtype=self.dtype)
        # If dim is CHW, chop off the C dimension.
        if len(res.shape) == 3:
            res = res[0]
        return res

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (*Image.open(self.paths[0]).size, len(self.paths))

    def __getitem__(self, key):
        """
        Get a 3D subvolume of the data from the given indices.

        Note that this method can be quite slow if the slice is "deep" in Z
        and small in XY. For better performance, this method will try to use
        the cache if `cache_size` is set to a value greater than 0 in the
        constructor of this class.

        Arguments:
            key (tuple): The indices to slice.

        """
        # Normalize the indices
        zs, ys, xs = normalize_key(key, self.shape[::-1])

        # Read the images.
        images = [self._read_image(self.paths[z]) for z in range(zs[0], zs[1])]

        # Return the subvolume.
        vol = np.stack(images, axis=-1)
        # Return the subvolume.
        return vol[xs[0] : xs[1], ys[0] : ys[1], :]

    @property
    def dtype(self) -> np.dtype:
        return self[0:1, 0:1, 0].dtype
