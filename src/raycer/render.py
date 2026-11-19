"""Column raycaster (REND-01..09): renders RGB, depth and labels buffers from a car pose."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from raycer import palette
from raycer.physics import CarState
from raycer.track import Material, Track

Array = NDArray[np.float64]

CAMERA_HEIGHT = 1.0
# Stripe colours A/B and stripe width per material, indexed by material ID.
_STRIPE_A = np.zeros((6, 3))
_STRIPE_B = np.zeros((6, 3))
_STRIPE_W = np.ones(6)
for _m, _a, _b, _w in (
    (Material.LEFT_BARRIER, palette.LEFT_A, palette.LEFT_B, 2.0),
    (Material.RIGHT_BARRIER, palette.RIGHT_A, palette.RIGHT_B, 2.0),
    (Material.OBSTACLE, palette.OBSTACLE_A, palette.OBSTACLE_B, 0.5),
):
    _STRIPE_A[_m], _STRIPE_B[_m], _STRIPE_W[_m] = _a, _b, _w
_FOG = np.array(palette.FOG, dtype=np.float64)
# Label IDs of the labels buffer (REND-05); wall IDs equal track.Material.
LABELS: dict[int, str] = {
    0: "SKY",
    1: "ROAD",
    2: "LEFT_BARRIER",
    3: "RIGHT_BARRIER",
    4: "FINISH",
    5: "OBSTACLE",
}


@dataclass(frozen=True)
class RenderOutput:
    """Rendered buffers: rgb (H, W, 3) uint8, depth (H, W) float32, labels (H, W) uint8."""

    rgb: NDArray[np.uint8] | None
    depth: NDArray[np.float32] | None
    labels: NDArray[np.uint8] | None


def _checker(px: Array, py: Array, cell: float, a: tuple[int, ...], b: tuple[int, ...]) -> Array:
    """World-space checkerboard colour: even cell index uses a, odd uses b."""
    odd = (np.floor(px / cell) + np.floor(py / cell)) % 2 == 1
    return np.where(odd[..., None], np.array(b, dtype=np.float64), np.array(a, dtype=np.float64))


def _fog(colour: Array, depth: Array, max_view_distance: float) -> Array:
    """Blend colour towards FOG with factor min(1, depth/max)^1.5."""
    f = np.minimum(1.0, depth / max_view_distance) ** 1.5
    return colour + (_FOG - colour) * f[..., None]


class Renderer:
    """Column raycaster with precomputed per-column rays and per-row floor depths."""

    def __init__(
        self,
        width: int = 96,
        height: int = 72,
        fov_deg: float = 90.0,
        max_view_distance: float = 60.0,
        *,
        rgb: bool = True,
        depth: bool = False,
        labels: bool = False,
    ) -> None:
        """Validate the camera (REND-01/04) and precompute rays, rows and the sky gradient."""
        if width % 2 or not 32 <= width <= 640:
            raise ValueError(f"width must be even and in 32..640, got {width}")
        if height % 2 or not 24 <= height <= 480:
            raise ValueError(f"height must be even and in 24..480, got {height}")
        if not 60.0 <= fov_deg <= 120.0:
            raise ValueError(f"fov_deg must be in 60..120, got {fov_deg}")
        if not 10.0 <= max_view_distance <= 200.0:
            raise ValueError(f"max_view_distance must be in 10..200, got {max_view_distance}")
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.max_view_distance = max_view_distance
        self.rgb_enabled = rgb
        self.depth_enabled = depth
        self.labels_enabled = labels
        self.focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
        half = height // 2
        self._k = (np.arange(width) + 0.5 - width / 2) / self.focal
        self._dy = np.arange(height) + 0.5 - height / 2
        # Floor depth of rows H/2.. (dy > 0); rows above are sky or wall.
        self._floor_depth = self.focal * CAMERA_HEIGHT / self._dy[half:]
        t = (np.arange(half) / (half - 1))[:, None]
        top, horizon = np.array(palette.SKY_TOP), np.array(palette.SKY_HORIZON)
        self._sky = top + (horizon - top) * t

    def _cast(
        self, track: Track, car_state: CarState
    ) -> tuple[Array, Array, NDArray[np.intp], Array, Array]:
        """Return per-column ray (W,2), depth, segment index, segment parameter and |cos θ|."""
        h = car_state.heading
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([math.sin(h), -math.cos(h)])
        rays = fwd + self._k[:, None] * right
        w = self.width
        idx = track.segments_in_radius((car_state.x, car_state.y), self.max_view_distance)
        depth = np.full(w, self.max_view_distance)
        if idx.size == 0:
            return rays, depth, np.zeros(w, dtype=np.intp), np.zeros(w), np.ones(w)
        a = track.seg_a[idx] - np.array([car_state.x, car_state.y])
        e = track.seg_b[idx] - track.seg_a[idx]
        rx, ry = rays[:, 0:1], rays[:, 1:2]
        denom = rx * e[:, 1] - ry * e[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (a[:, 0] * e[:, 1] - a[:, 1] * e[:, 0]) / denom
            s = (a[:, 0] * ry - a[:, 1] * rx) / denom
        t = np.where((s >= 0.0) & (s <= 1.0) & (t > 0.0), t, np.inf)
        cols = np.arange(w)
        j = np.argmin(t, axis=1)
        nearest = t[cols, j]
        hit = nearest < self.max_view_distance
        depth[hit] = nearest[hit]
        cos = np.abs(denom[cols, j]) / (np.hypot(rx[:, 0], ry[:, 0]) * np.linalg.norm(e[j], axis=1))
        return rays, depth, idx[j], s[cols, j], cos

    def wall_depth(self, track: Track, car_state: CarState) -> Array:
        """Return per-column perpendicular wall distance, max_view_distance where nothing is hit."""
        return self._cast(track, car_state)[1]

    def render(self, track: Track, car_state: CarState) -> RenderOutput:
        """Render the enabled buffers: sky, checkered floor and striped, shaded walls."""
        if not (self.rgb_enabled or self.depth_enabled or self.labels_enabled):
            return RenderOutput(None, None, None)
        rays, depth, seg, s, cos = self._cast(track, car_state)
        half = self.height // 2
        mvd = self.max_view_distance
        mask = (np.abs(self._dy)[:, None] <= self.focal * CAMERA_HEIGHT / depth) & (depth < mvd)
        fd = self._floor_depth[:, None]
        mat = track.seg_material[seg]
        rgb = dbuf = lbuf = None
        if self.rgb_enabled or self.labels_enabled:
            px = car_state.x + fd * rays[:, 0]
            py = car_state.y + fd * rays[:, 1]
            q = track.finish_quad
            edge = q[[1, 2, 3, 0]] - q
            e4, q4 = edge[:, :, None, None], q[:, :, None, None]
            side = e4[:, 0] * (py - q4[:, 1]) - e4[:, 1] * (px - q4[:, 0])
            inside = np.all(side >= 0, axis=0) | np.all(side <= 0, axis=0)
        if self.depth_enabled or self.labels_enabled:
            far = np.float32(mvd)
            dbuf = np.full((self.height, self.width), far, dtype=np.float32)
            dbuf[half:] = fd.astype(np.float32)
            dbuf = np.where(mask, depth.astype(np.float32)[None], dbuf)
            dbuf = np.ascontiguousarray(np.minimum(dbuf, far))
        if self.labels_enabled:
            lbuf = np.zeros((self.height, self.width), dtype=np.uint8)
            lbuf[half:] = np.where(inside, 4, 1)
            lbuf = np.where(mask, mat[None], lbuf).astype(np.uint8)
            lbuf[dbuf == far] = 0
        if self.rgb_enabled:
            img = np.empty((self.height, self.width, 3))
            img[:half] = self._sky[:, None, :]
            floor = np.where(
                inside[..., None],
                _checker(px, py, 0.5, palette.FINISH_A, palette.FINISH_B),
                _checker(px, py, 4.0, palette.ROAD_A, palette.ROAD_B),
            )
            img[half:] = _fog(floor, np.broadcast_to(fd, px.shape), mvd)
            u = track.seg_u0[seg] + s * np.linalg.norm(track.seg_b[seg] - track.seg_a[seg], axis=1)
            odd = np.floor(u / _STRIPE_W[mat]) % 2 == 1
            wall = (
                np.where(odd[:, None], _STRIPE_B[mat], _STRIPE_A[mat]) * (0.6 + 0.4 * cos)[:, None]
            )
            wall = _fog(wall, depth, mvd)
            img = np.where(mask[..., None], wall[None], img)
            rgb = np.ascontiguousarray(np.clip(np.rint(img), 0, 255).astype(np.uint8))
        return RenderOutput(rgb=rgb, depth=dbuf if self.depth_enabled else None, labels=lbuf)
