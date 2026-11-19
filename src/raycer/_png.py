"""Minimal PNG writer without Pillow: 8-bit greyscale or RGB, no compression filters."""

import os
import struct
import zlib

import numpy as np
from numpy.typing import NDArray


def _chunk(kind: bytes, data: bytes) -> bytes:
    """Return one PNG chunk with length and CRC."""
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def write_png(path: str | os.PathLike[str], array: NDArray[np.uint8]) -> None:
    """Write a (H,W) uint8 array as greyscale or a (H,W,3) uint8 array as RGB PNG."""
    if array.dtype != np.uint8 or not (
        array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 3)
    ):
        raise ValueError(f"expected (H,W) or (H,W,3) uint8, got {array.shape} {array.dtype}")
    height, width = array.shape[:2]
    colour_type = 0 if array.ndim == 2 else 2
    rows = np.ascontiguousarray(array).reshape(height, -1)
    raw = np.hstack([np.zeros((height, 1), np.uint8), rows]).tobytes()  # filter byte 0 per row
    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", zlib.compress(raw)))
        f.write(_chunk(b"IEND", b""))
