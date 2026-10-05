"""
Volume providers must fail loudly on bad input instead of producing a
silently corrupted volume.
"""

import numpy as np
import pytest
from PIL import Image

from ml4paleo.volume_providers import ImageStackVolumeProvider


def _save(path, pixels, mode=None):
    image = Image.fromarray(pixels)
    if mode is not None:
        image = image.convert(mode)
    image.save(path)
    return path


def test_unreadable_slices_raise(tmp_path):
    good = _save(tmp_path / "0.png", np.zeros((3, 5), dtype=np.uint8))
    bad = tmp_path / "1.png"
    bad.write_bytes(b"not an image")
    provider = ImageStackVolumeProvider([good, bad])
    with pytest.raises(ValueError, match="Could not read"):
        provider[:, :, 0:2]
    with pytest.raises(ValueError, match="Could not read"):
        ImageStackVolumeProvider([bad, good])


def test_mismatched_slice_sizes_raise(tmp_path):
    first = _save(tmp_path / "0.png", np.zeros((3, 5), dtype=np.uint8))
    second = _save(tmp_path / "1.png", np.zeros((5, 3), dtype=np.uint8))
    provider = ImageStackVolumeProvider([first, second])
    with pytest.raises(ValueError, match="first slice"):
        provider[:, :, 0:2]


def test_rgb_slices_keep_the_first_channel(tmp_path):
    pixels = np.arange(15, dtype=np.uint8).reshape(3, 5) * 10
    rgb = _save(tmp_path / "rgb.png", np.stack([pixels] * 3, axis=-1))
    provider = ImageStackVolumeProvider([rgb])
    np.testing.assert_array_equal(provider[:, :, 0:1][:, :, 0], pixels.T)
    assert provider.shape == (5, 3, 1)
    assert provider.dtype == np.uint8
    palette = _save(tmp_path / "p.png", pixels, mode="P")
    assert ImageStackVolumeProvider([palette])[:, :, 0:1].shape == (5, 3, 1)


def test_palette_indices_are_kept(tmp_path):
    indices = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    image = Image.fromarray(indices, mode="P")
    image.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (256 * 3 - 12))
    image.save(tmp_path / "labels.png")
    provider = ImageStackVolumeProvider([tmp_path / "labels.png"])
    assert provider[:, :, 0:1][:, 0, 0].tolist() == [0, 1, 2, 3]


def test_mixed_pixel_types_raise(tmp_path):
    first = _save(tmp_path / "0.tif", np.zeros((3, 5), dtype=np.uint8))
    second = _save(tmp_path / "1.tif", np.zeros((3, 5), dtype=np.uint16))
    provider = ImageStackVolumeProvider([first, second])
    with pytest.raises(ValueError, match="pixel type"):
        provider[:, :, 0:2]
