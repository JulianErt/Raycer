"""Track model: wall segments, centerline, checkpoints, culling grid and builtin tracks."""

import math
from collections.abc import Sequence
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

CELL_SIZE = 20.0
CENTER_STEP = 1.0
WALL_STEP = 2.0
OBSTACLE_SIZE = 1.5
_SPAN_SAMPLES = 100


class Material(IntEnum):
    """Segment material; values equal the label IDs."""

    LEFT_BARRIER = 2
    RIGHT_BARRIER = 3
    OBSTACLE = 5


class Track:
    """A closed track: wall/obstacle segments, centerline and a 20 m culling grid."""

    def __init__(
        self,
        seg_a: Array,
        seg_b: Array,
        seg_material: NDArray[np.uint8],
        seg_u0: Array,
        center: Array,
        normal: Array,
        width: Array,
    ) -> None:
        """Store the arrays and derive length, checkpoints, finish quad and grid."""
        self.seg_a = seg_a
        self.seg_b = seg_b
        self.seg_material = seg_material
        self.seg_u0 = seg_u0
        self.center = center
        self.normal = normal
        self.width = width
        self.length = float(np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1).sum())
        n_cp = max(8, round(self.length / 25))
        self.checkpoint_s = np.arange(n_cp, dtype=np.float64) * self.length / n_cp
        n0 = normal[0]
        t0 = np.array([n0[1], -n0[0]])
        half = width[0] / 2
        self.finish_quad = np.array(
            [
                center[0] - 0.5 * t0 - half * n0,
                center[0] + 0.5 * t0 - half * n0,
                center[0] + 0.5 * t0 + half * n0,
                center[0] - 0.5 * t0 + half * n0,
            ]
        )
        lo = np.floor(np.minimum(seg_a, seg_b) / CELL_SIZE).astype(np.intp)
        hi = np.floor(np.maximum(seg_a, seg_b) / CELL_SIZE).astype(np.intp)
        cells: dict[tuple[int, int], list[int]] = {}
        for i in range(len(seg_a)):
            for cx in range(lo[i, 0], hi[i, 0] + 1):
                for cy in range(lo[i, 1], hi[i, 1] + 1):
                    cells.setdefault((cx, cy), []).append(i)
        self._cells = {k: np.array(v, dtype=np.intp) for k, v in cells.items()}

    def _collect(self, keys: list[tuple[int, int]]) -> NDArray[np.intp]:
        found = [self._cells[k] for k in keys if k in self._cells]
        if not found:
            return np.empty(0, dtype=np.intp)
        return np.unique(np.concatenate(found))

    def segments_in_radius(
        self, pos: Sequence[float] | Array, radius: float, cull: bool = True
    ) -> NDArray[np.intp]:
        """Return sorted indices of segments in grid cells intersecting the circle."""
        if not cull:
            return np.arange(len(self.seg_a), dtype=np.intp)
        x, y = float(pos[0]), float(pos[1])
        keys = []
        for cx in range(
            math.floor((x - radius) / CELL_SIZE), math.floor((x + radius) / CELL_SIZE) + 1
        ):
            for cy in range(
                math.floor((y - radius) / CELL_SIZE), math.floor((y + radius) / CELL_SIZE) + 1
            ):
                dx = max(cx * CELL_SIZE - x, 0.0, x - (cx + 1) * CELL_SIZE)
                dy = max(cy * CELL_SIZE - y, 0.0, y - (cy + 1) * CELL_SIZE)
                if dx * dx + dy * dy <= radius * radius:
                    keys.append((cx, cy))
        return self._collect(keys)

    def segments_near(self, pos: Sequence[float] | Array, cull: bool = True) -> NDArray[np.intp]:
        """Return sorted indices of segments in the 3x3 grid cells around pos."""
        if not cull:
            return np.arange(len(self.seg_a), dtype=np.intp)
        cx = math.floor(float(pos[0]) / CELL_SIZE)
        cy = math.floor(float(pos[1]) / CELL_SIZE)
        return self._collect([(cx + i, cy + j) for i in (-1, 0, 1) for j in (-1, 0, 1)])


def _resample_closed(points: Array, step: float) -> Array:
    """Resample a closed polyline by arc length into round(L / step) equal steps."""
    closed = np.vstack([points, points[:1]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    n = round(cum[-1] / step)
    s = np.arange(n) * (cum[-1] / n)
    return np.column_stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])])


def _catmull_rom_closed(control: Array) -> Array:
    """Densely sample a closed centripetal Catmull-Rom spline (alpha = 0.5)."""
    k = len(control)
    u = np.linspace(0.0, 1.0, _SPAN_SAMPLES, endpoint=False)[:, None]
    spans = []
    for i in range(k):
        p0, p1, p2, p3 = (control[(i + j) % k] for j in (-1, 0, 1, 2))
        t0 = 0.0
        t1 = t0 + float(np.linalg.norm(p1 - p0)) ** 0.5
        t2 = t1 + float(np.linalg.norm(p2 - p1)) ** 0.5
        t3 = t2 + float(np.linalg.norm(p3 - p2)) ** 0.5
        t = t1 + u * (t2 - t1)
        a1 = ((t1 - t) * p0 + (t - t0) * p1) / (t1 - t0)
        a2 = ((t2 - t) * p1 + (t - t1) * p2) / (t2 - t1)
        a3 = ((t3 - t) * p2 + (t - t2) * p3) / (t3 - t2)
        b1 = ((t2 - t) * a1 + (t - t0) * a2) / (t2 - t0)
        b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1)
        spans.append(((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1))
    return np.vstack(spans)


def _wall(points: Array) -> tuple[Array, Array, Array]:
    """Resample a closed wall into 2 m segments; return seg_a, seg_b, seg_u0."""
    a = _resample_closed(points, WALL_STEP)
    b = np.roll(a, -1, axis=0)
    lengths = np.linalg.norm(b - a, axis=1)
    return a, b, np.concatenate([[0.0], np.cumsum(lengths[:-1])])


def _build_track(center: Array, width: Array, obstacles: Sequence[tuple[float, float]]) -> Track:
    """Build a Track from a 1 m centerline, widths and (s, lateral offset) obstacles."""
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offset = normal * (width / 2)[:, None]
    parts = [_wall(center + offset), _wall(center - offset)]
    materials = [
        np.full(len(parts[0][0]), Material.LEFT_BARRIER, dtype=np.uint8),
        np.full(len(parts[1][0]), Material.RIGHT_BARRIER, dtype=np.uint8),
    ]
    step = float(np.linalg.norm(center[1] - center[0]))
    h = OBSTACLE_SIZE / 2
    for s, lateral in obstacles:
        i = round(s / step) % len(center)
        mid, t, n = center[i] + lateral * normal[i], tangent[i], normal[i]
        a: Array = np.array(
            [mid - h * t - h * n, mid + h * t - h * n, mid + h * t + h * n, mid - h * t + h * n]
        )
        parts.append((a, np.roll(a, -1, axis=0), np.arange(4.0) * OBSTACLE_SIZE))
        materials.append(np.full(4, Material.OBSTACLE, dtype=np.uint8))
    return Track(
        np.vstack([p[0] for p in parts]),
        np.vstack([p[1] for p in parts]),
        np.concatenate(materials),
        np.concatenate([p[2] for p in parts]),
        center,
        normal,
        width,
    )


def _oval_center() -> Array:
    """Analytic oval centerline: 120 m straights, r = 35 m, s = 0 at (0, -35) heading +x."""
    r, half = 35.0, 60.0
    arc = math.pi * r
    length = 4 * half + 2 * arc
    n = round(length / CENTER_STEP)
    s = np.arange(n) * (length / n)
    # Shift so s = 0 is the start of the right semicircle; the bottom straight's second half
    # becomes the last section.
    q = (s - half) % length
    b1, b2, b3 = arc, arc + 2 * half, 2 * arc + 2 * half
    th_r = -math.pi / 2 + q / r
    th_l = math.pi / 2 + (q - b2) / r
    x = np.select(
        [q < b1, q < b2, q < b3],
        [half + r * np.cos(th_r), half - (q - b1), -half + r * np.cos(th_l)],
        -half + (q - b3),
    )
    y = np.select([q < b1, q < b2, q < b3], [r * np.sin(th_r), np.full(n, r), r * np.sin(th_l)], -r)
    return np.column_stack([x, y])


_KIDNEY = np.array(
    [
        (90.0, 5.0),
        (75.0, 45.0),
        (30.0, 65.0),
        (-20.0, 60.0),
        (-65.0, 40.0),
        (-85.0, 0.0),
        (-70.0, -40.0),
        (-30.0, -55.0),
        (5.0, -45.0),
        (55.0, -45.0),
    ]
)

_TECHNICAL = np.array(
    [
        (117.0, 0.0),
        (110.0, 52.0),
        (72.0, 78.0),
        (32.0, 68.0),
        (0.0, 91.0),
        (-39.0, 104.0),
        (-78.0, 78.0),
        (-104.0, 32.0),
        (-94.0, 0.0),
        (-107.0, -32.0),
        (-78.0, -78.0),
        (-20.0, -84.0),
        (32.0, -58.0),
        (84.0, -72.0),
    ]
)

# (s, lateral offset) of the technical track's obstacles.
_TECHNICAL_OBSTACLES = ((60.0, 2.5), (150.0, -2.5), (260.0, 2.5), (380.0, -2.5))


def load_builtin(name: str) -> Track:
    """Return the builtin track `oval`, `kidney` or `technical`."""
    obstacles: Sequence[tuple[float, float]] = ()
    if name == "oval":
        center, w = _oval_center(), 14.0
    elif name == "kidney":
        center, w = _resample_closed(_catmull_rom_closed(_KIDNEY), CENTER_STEP), 12.0
    elif name == "technical":
        center = _resample_closed(_catmull_rom_closed(_TECHNICAL), CENTER_STEP)
        w, obstacles = 10.0, _TECHNICAL_OBSTACLES
    else:
        raise ValueError(f"unknown builtin track: {name!r}")
    return _build_track(center, np.full(len(center), w), obstacles)
