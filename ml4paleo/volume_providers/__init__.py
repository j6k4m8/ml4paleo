"""
This module contains the VolumeProvider class and its subclasses.

VolumeProvider is an abstract base class that defines the interface for
volume providers. Subclasses of VolumeProvider implement the interface for the
numpy slicing protocol. This allows the user to slice the volume provider
object as if it were a numpy array.

"""

import abc
import logging
import pathlib
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import psutil
from PIL import Image

logger = logging.getLogger(__name__)


def _normalize_key(
    key: Tuple,
    self_shape: Tuple,
    permit_single_int: bool = True,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    NOTE: This function is copied from the BossDB "intern" package.
    https://github.com/jhuapl-boss/intern/blob/master/intern/convenience/array.py#L1293

    Given indexing tuple, return (start, stop) for each dimension XYZ.


    Arguments:
        key (Tuple): An array of three values in one of the following formats
            1. Start/Stop (`int:int`)
            2. Single index (`int`)
        self_shape (Tuple): Shape of the array being indexed. Used to
            determine the bounds of the array if no endpoint is provided.
            For example, if the user asks for `my_array[60:, 60:, 60:]` then
            the endpoint is assumed to be the full extent of the array.
        permit_single_int (bool): Default True. Permit a single integer.
            This integer is assumed to be a single z slice
            (e.g. `my_array[500]`).

    Returns:
        Tuple[Tuple]: Set of three tuples with (start, stop) integers
            for each dimension in XYZ

    """
    # There is a wide variety of indexing options available, including
    # single-integer indexing,
    # tuple-of-slices indexing,
    # tuple-of-int indexing...

    # First we'll address if the user presents a single integer.
    # ```
    # my_array[500]
    # ```
    # In this case, the user is asking for a single Z slice (or single X
    # slice if in XYZ order... But that's a far less common use case.)
    # We will get the full XY extents and download a single 2D array:
    if isinstance(key, int) and permit_single_int:
        # Get the full Z slice:
        xs = (0, self_shape[2])
        ys = (0, self_shape[1])
        zs = (key, key + 1)
    else:
        _normalize_units = (1, 1, 1)

        # We will now do the following codeblock three times, for X,Y,Z:
        # First, we check to see if this index is a single integer. If so,
        # the user is requesting a 2D array with zero depth along this
        # dimension. For example, if the user asks for
        # ```
        # my_data[0:120, 0:120, 150]
        # ```
        # Then "150" suggests that the user just wants one single X slice.
        if isinstance(key[2], int):
            xs = (key[2], key[2] + 1)
        else:
            # If the key is a Slice, then it has .start and .stop attrs.
            # (The user is requesting an array with more than one slice
            # in this dimension.)
            start = key[2].start if key[2].start else 0
            stop = key[2].stop if key[2].stop else self_shape[0]

            start = int(start / _normalize_units[0])
            stop = int(stop / _normalize_units[0])

            # Cast the coords to integers (since Boss needs int coords)
            xs = (int(start), int(stop))

        # Do the same thing again for the next dimension: Either a single
        # integer, or a slice...
        if isinstance(key[1], int):
            ys = (key[1], key[1] + 1)
        else:
            start = key[1].start if key[1].start else 0
            stop = key[1].stop if key[1].stop else self_shape[1]

            start = start / _normalize_units[1]
            stop = stop / _normalize_units[1]

            ys = (int(start), int(stop))

        # Do the same thing again for the last dimension: Either a single
        # integer, or a slice...
        if isinstance(key[0], int):
            zs = (key[0], key[0] + 1)
        else:
            start = key[0].start if key[0].start else 0
            stop = key[0].stop if key[0].stop else self_shape[2]

            start = start / _normalize_units[2]
            stop = stop / _normalize_units[2]

            zs = (int(start), int(stop))

    return xs, ys, zs


class VolumeProvider(abc.ABC):
    """
    Abstract base class for volume providers.

    A VolumeProvider is a class that implements the numpy-style slicing
    protocol, and provides a 3D volume of data backed by a file or other
    storage mechanism.
    """

    @abc.abstractproperty
    def shape(self) -> Tuple[int, int, int]:
        """
        Return the shape of the volume.

        Returns:
            The shape of the volume.

        """
        raise NotImplementedError

    @abc.abstractproperty
    def dtype(self) -> np.dtype:
        """
        Return the dtype of the volume.

        Returns:
            The dtype of the volume.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key) -> np.ndarray:
        """
        Return a slice of the volume.

        Arguments:
            key: The slice to return.

        Returns:
            The slice of the volume.

        """
        raise NotImplementedError


class NumpyVolumeProvider(VolumeProvider):
    """
    A VolumeProvider that provides a 3D volume of data from a numpy array.
    """

    def __init__(self, data: np.ndarray):
        """
        Create a new NumpyVolumeProvider.

        Arguments:
            data: The 3D numpy array to provide.

        Raises:
            ValueError: If the data is not 3D.

        """
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D, but has shape {data.shape}.")
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype


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
        self._read_image = lru_cache(maxsize=self._cache_size)(self._read_image)

    def _read_image(self, path: pathlib.Path) -> np.ndarray:
        """
        Read an image from disk.

        Arguments:
            path (pathlib.Path): The path to the image to read.

        Returns:
            np.ndarray: The image data.

        """
        logger.debug(f"Reading image {path}")
        return np.array(Image.open(path))

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (*Image.open(self.paths[0]).size[::-1], len(self.paths))

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
        zs, ys, xs = _normalize_key(key, self.shape[::-1])

        # Read the images.
        images = [self._read_image(self.paths[z]) for z in range(zs[0], zs[1])]

        # Return the subvolume.
        vol = np.stack(images, axis=-1)
        # Return the subvolume.
        return vol[xs[0] : xs[1], ys[0] : ys[1], :]

    @property
    def dtype(self) -> np.dtype:
        return self[0:1, 0:1, 0].dtype


__all__ = ["VolumeProvider", "NumpyVolumeProvider", "ImageStackVolumeProvider"]
