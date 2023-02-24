from typing import Tuple
import abc
import numpy as np


def normalize_key(
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


__all__ = ["VolumeProvider", "normalize_key"]
