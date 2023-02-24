"""
This module contains the VolumeProvider class and its subclasses.

VolumeProvider is an abstract base class that defines the interface for
volume providers. Subclasses of VolumeProvider implement the interface for the
numpy slicing protocol. This allows the user to slice the volume provider
object as if it were a numpy array.

"""
from .volume_provider import VolumeProvider
from .numpyvp import NumpyVolumeProvider
from .imagevp import ImageStackVolumeProvider
from .zarrvp import ZarrVolumeProvider

__all__ = [
    "VolumeProvider",
    "NumpyVolumeProvider",
    "ImageStackVolumeProvider",
    "ZarrVolumeProvider",
]
