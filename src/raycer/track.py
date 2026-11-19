"""Track model: wall segments, centerline, checkpoints, culling grid and builtin tracks."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

CELL_SIZE = 20.0
CENTER_STEP = 1.0
WALL_STEP = 2.0
OBSTACLE_SIZE = 1.5
_SPAN_SAMPLES = 100
_KEY = 1 << 32
_MAX_ATTEMPTS = 100


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


@dataclass(frozen=True)
class TrackParams:
    """Generator parameters (WORLD-05); the defaults are the `medium` preset."""

    k_min: int = 10
    k_max: int = 12
    r_min: float = 40.0
    r_max: float = 90.0
    w_min: float = 10.0
    w_max: float = 14.0
    min_curvature_radius: float = 12.0
    n_obstacles: int = 3


PRESETS: dict[str, TrackParams] = {
    "easy": TrackParams(8, 10, 50.0, 80.0, 12.0, 16.0, 12.0, 0),
    "medium": TrackParams(),
    "hard": TrackParams(12, 16, 30.0, 100.0, 8.0, 12.0, 12.0, 6),
}

SPLITS: dict[str, range] = {
    "train": range(0, 1_000_000),
    "val": range(1_000_000, 1_010_000),
    "test": range(1_010_000, 1_020_000),
}


def split_of(seed: int) -> str:
    """Return the name of the split containing seed; raise ValueError outside all splits."""
    for name, seeds in SPLITS.items():
        if seed in seeds:
            return name
    raise ValueError(f"seed {seed} is outside all splits")


def _smooth_radii(u: Array) -> Array:
    """Apply the cyclic (1, 2, 1) / 4 smoothing S twice."""
    for _ in range(2):
        u = (np.roll(u, 1) + 2 * u + np.roll(u, -1)) / 4
    return u


def _cross(u: Array, v: Array) -> Array:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _near_pairs(points: Array, radius: float) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return all index pairs i < j with |p_i - p_j| < radius, found via a uniform grid."""
    cell = np.floor(points / radius).astype(np.int64)
    key = cell[:, 0] * _KEY + cell[:, 1]
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    rows = np.arange(len(points))
    found_i, found_j = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            target = key + dx * _KEY + dy
            lo = np.searchsorted(sorted_key, target, "left")
            counts = np.searchsorted(sorted_key, target, "right") - lo
            i = np.repeat(rows, counts)
            j = order[np.arange(len(i)) + np.repeat(lo - np.cumsum(counts) + counts, counts)]
            keep = i < j
            found_i.append(i[keep])
            found_j.append(j[keep])
    i, j = np.concatenate(found_i), np.concatenate(found_j)
    close = np.linalg.norm(points[i] - points[j], axis=1) < radius
    return i[close], j[close]


def _center_errors(center: Array, width: Array, min_curvature_radius: float) -> list[str]:
    """Check length, curvature and separation of a centerline (WORLD-04 a, b, d)."""
    errors = []
    chords = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    length = float(chords.sum())
    if not 250.0 <= length <= 1200.0:
        errors.append(f"length {length:.1f} m is outside [250, 1200]")
    w_max = float(width.max())
    p, q, r = np.roll(center, 3, axis=0), center, np.roll(center, -3, axis=0)
    a, b, c = (np.linalg.norm(x - y, axis=1) for x, y in ((p, q), (q, r), (r, p)))
    with np.errstate(divide="ignore"):
        radius = float((a * b * c / (2 * np.abs(_cross(q - p, r - p)))).min())
    limit = max(min_curvature_radius, w_max / 2 + 2)
    if radius < limit:
        errors.append(f"curvature radius {radius:.1f} m is below {limit:.1f} m")
    i, j = _near_pairs(center, w_max + 6)
    ds = np.abs(i - j) * (length / len(center))
    far = np.minimum(ds, length - ds) > 3 * w_max
    i, j = i[far], j[far]
    gap = np.linalg.norm(center[i] - center[j], axis=1) - (width[i] + width[j]) / 2
    if (gap < 6).any():
        errors.append(f"centerline separation {gap.min() + 6:.1f} m is too small")
    return errors


def _segment_errors(track: Track) -> list[str]:
    """Check wall simplicity, wall crossings and obstacle containment (WORLD-04 c, e)."""
    errors = []
    a, b, mat = track.seg_a, track.seg_b, track.seg_material
    # Crossing segments have midpoints at most the longest segment length apart.
    i, j = _near_pairs((a + b) / 2, float(np.linalg.norm(b - a, axis=1).max()) + 1e-9)
    d1, d2 = b[i] - a[i], b[j] - a[j]
    proper = (_cross(d1, a[j] - a[i]) * _cross(d1, b[j] - a[i]) < 0) & (
        _cross(d2, a[i] - a[j]) * _cross(d2, b[i] - a[j]) < 0
    )
    mi, mj = mat[i[proper]], mat[j[proper]]
    for side, wall in (("left", Material.LEFT_BARRIER), ("right", Material.RIGHT_BARRIER)):
        if ((mi == wall) & (mj == wall)).any():
            errors.append(f"{side} wall polygon self-intersects")
    if ((mi != mj) & (mi != Material.OBSTACLE) & (mj != Material.OBSTACLE)).any():
        errors.append("left and right walls cross")
    # Every non-obstacle segment is a wall.
    if ((mi == Material.OBSTACLE) != (mj == Material.OBSTACLE)).any():
        errors.append("an obstacle crosses a wall")
    corners = a[mat == Material.OBSTACLE]
    if len(corners):
        k = np.argmin(np.linalg.norm(corners[:, None] - track.center[None], axis=2), axis=1)
        lateral = np.abs(np.einsum("ij,ij->i", corners - track.center[k], track.normal[k]))
        if (lateral >= track.width[k] / 2).any():
            errors.append("an obstacle is not inside the corridor")
    return errors


def validate_track(track: Track, min_curvature_radius: float = 12.0) -> list[str]:
    """Return one message per violated WORLD-04 check; empty for a valid track."""
    return _center_errors(track.center, track.width, min_curvature_radius) + _segment_errors(track)


def _draw_obstacles(
    rng: np.random.Generator, params: TrackParams, width: Array, length: float
) -> list[tuple[float, float]] | None:
    """Draw (s, lateral offset) obstacles snapped to centerline points; None if spacing fails."""
    n = len(width)
    step = length / n
    lo, hi = math.ceil(30.0 / step), math.floor((length - 30.0) / step)
    for _ in range(_MAX_ATTEMPTS):
        s = rng.uniform(30.0, length - 30.0, params.n_obstacles)
        idx = np.clip(np.round(s / step).astype(np.intp), lo, hi)
        gaps = np.abs(idx[:, None] - idx[None]) * step
        if (gaps[~np.eye(len(idx), dtype=bool)] >= 15.0).all():
            break
    else:
        return None
    half = width[idx] / 2
    offset = rng.uniform(np.maximum(0.0, 4.75 - half), half - 1.25)
    sign = rng.choice((-1.0, 1.0), params.n_obstacles)
    return [(float(i), float(x)) for i, x in zip(idx, sign * offset, strict=True)]


def generate_track(seed: int, params: TrackParams = TrackParams()) -> Track:
    """Deterministically generate a valid closed track from seed (WORLD-03)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(_MAX_ATTEMPTS):
        k = int(rng.integers(params.k_min, params.k_max + 1))
        phi = 2 * np.pi * np.arange(k) / k + rng.uniform(-0.3, 0.3, k) * 2 * np.pi / k
        r = _smooth_radii(rng.uniform(params.r_min, params.r_max, k))
        m = int(rng.integers(1, 4))
        psi = rng.uniform(0.0, 2 * np.pi)
        control = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
        center = _resample_closed(_catmull_rom_closed(control), CENTER_STEP)
        chords = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
        s = np.concatenate([[0.0], np.cumsum(chords[:-1])])
        length = float(chords.sum())
        span = params.w_max - params.w_min
        width = params.w_min + span * (0.5 + 0.5 * np.sin(2 * np.pi * m * s / length + psi))
        if _center_errors(center, width, params.min_curvature_radius):
            continue
        obstacles = _draw_obstacles(rng, params, width, length)
        if obstacles is None:
            continue
        # _build_track maps s to the point round(s / step); pass indices times its step.
        step = float(np.linalg.norm(center[1] - center[0]))
        track = _build_track(center, width, [(i * step, x) for i, x in obstacles])
        if not _segment_errors(track):
            return track
    raise ValueError(f"no valid track for seed {seed} after {_MAX_ATTEMPTS} attempts")
