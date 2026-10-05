"""
How label chunks are stored.

Every version of a label chunk is stored once, under a key derived from its
content (the sha256 of its raw bytes). Snapshots, history, and undo therefore
never copy chunk data: they only remember which hashes were current.

The stored bytes are exactly a zarr v3 chunk of a uint8 array with the codecs
in `ZARR_CODECS`. The data gateway can serve them as-is to any zarr reader.
"""

import hashlib

import numpy as np
import zstandard
from numcodecs import Zstd

from . import LABEL_CHUNK_ZYX

ZSTD_LEVEL = 3
ZARR_CODECS = [
    {"name": "bytes"},
    {"name": "zstd", "configuration": {"level": ZSTD_LEVEL, "checksum": False}},
]

_zstd = Zstd(level=ZSTD_LEVEL)


def content_hash(chunk: np.ndarray) -> str | None:
    """
    Return the sha256 hex digest of a chunk's raw C-order bytes, or None for
    an all-zero chunk. All-zero chunks are never stored; a missing chunk reads
    as zeros.
    """
    chunk = _check_chunk(chunk)
    if not chunk.any():
        return None
    return hashlib.sha256(chunk.tobytes(order="C")).hexdigest()


def encode_chunk(chunk: np.ndarray) -> bytes:
    """
    Encode a full chunk as zarr v3 chunk bytes.
    """
    return bytes(_zstd.encode(np.ascontiguousarray(_check_chunk(chunk))))


def decode_chunk(data: bytes | None) -> np.ndarray:
    """
    Decode zarr v3 chunk bytes into a full chunk. None decodes to zeros.
    Decompression never produces more than one chunk's worth of bytes.
    """
    if data is None:
        return np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    size = int(np.prod(LABEL_CHUNK_ZYX))
    try:
        raw = zstandard.ZstdDecompressor().decompress(data, max_output_size=size)
    except zstandard.ZstdError as exc:
        raise ValueError(f"Invalid label chunk: {exc}") from exc
    if len(raw) != size:
        raise ValueError(f"Decoded label chunk has {len(raw)} bytes, expected {size}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(LABEL_CHUNK_ZYX).copy()


def blob_key(sha256_hex: str) -> str:
    """
    Return the storage key of a chunk blob, relative to its layer's prefix.
    The two-character fan-out keeps directory listings small.
    """
    if len(sha256_hex) != 64 or any(c not in "0123456789abcdef" for c in sha256_hex):
        raise ValueError(f"Not a sha256 hex digest: {sha256_hex!r}")
    return f"blobs/{sha256_hex[:2]}/{sha256_hex}"


def _check_chunk(chunk: np.ndarray) -> np.ndarray:
    if chunk.shape != LABEL_CHUNK_ZYX or chunk.dtype != np.uint8:
        raise ValueError(
            f"Label chunks are uint8 {LABEL_CHUNK_ZYX}, got {chunk.dtype} {chunk.shape}"
        )
    return chunk
