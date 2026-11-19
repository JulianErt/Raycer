import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from raycer._png import write_png


def read_chunks(path: Path) -> list[tuple[bytes, bytes, int]]:
    """Return (type, data, crc) for every chunk after the signature."""
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, pos = [], 8
    while pos < len(raw):
        (length,) = struct.unpack(">I", raw[pos : pos + 4])
        kind = raw[pos + 4 : pos + 8]
        data = raw[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", raw[pos + 8 + length : pos + 12 + length])
        chunks.append((kind, data, crc))
        pos += 12 + length
    return chunks


def decode_png(path: Path) -> NDArray[np.uint8]:
    """Decode an unfiltered 8-bit greyscale or RGB PNG."""
    chunks = read_chunks(path)
    width, height, depth, colour, *_ = struct.unpack(">IIBBBBB", chunks[0][1])
    assert depth == 8
    channels = {0: 1, 2: 3}[colour]
    raw = zlib.decompress(b"".join(d for k, d, _ in chunks if k == b"IDAT"))
    rows = np.frombuffer(raw, np.uint8).reshape(height, 1 + width * channels)
    assert (rows[:, 0] == 0).all()
    pixels = rows[:, 1:].reshape(height, width, channels)
    return pixels[:, :, 0] if colour == 0 else pixels


@pytest.mark.parametrize("shape", [(5, 7), (4, 6, 3)])
def test_roundtrip(tmp_path: Path, shape: tuple[int, ...]) -> None:
    array = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    path = tmp_path / "a.png"
    write_png(path, array)
    chunks = read_chunks(path)
    assert [k for k, _, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    width, height, _, colour, *_ = struct.unpack(">IIBBBBB", chunks[0][1])
    assert (width, height, colour) == (shape[1], shape[0], 0 if len(shape) == 2 else 2)
    for kind, data, crc in chunks:
        assert crc == zlib.crc32(kind + data)
    np.testing.assert_array_equal(decode_png(path), array)


@pytest.mark.parametrize(
    "array", [np.zeros((4, 4), np.float32), np.zeros((4, 4, 4), np.uint8), np.zeros(4, np.uint8)]
)
def test_rejects_bad_arrays(tmp_path: Path, array: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError, match="uint8"):
        write_png(tmp_path / "a.png", array)  # type: ignore[arg-type]
