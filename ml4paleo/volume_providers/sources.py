"""
Slices that aren't files on disk, such as members of an uploaded archive read
straight from object storage. The image stack and DICOM providers accept these
wherever they accept paths.
"""

import os
from typing import BinaryIO, Protocol


class SliceSource(Protocol):
    """
    One slice's bytes. `name` is used for sorting and in error messages.
    """

    name: str

    def open(self) -> BinaryIO: ...


def is_source(item: object) -> bool:
    return not isinstance(item, (str, os.PathLike)) and hasattr(item, "open")


def open_binary(item) -> BinaryIO:
    """
    Open a path or a `SliceSource` for reading bytes.
    """
    return item.open() if is_source(item) else open(item, "rb")
