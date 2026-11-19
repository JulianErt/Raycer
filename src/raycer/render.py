"""Column raycaster (REND-01..09): renders RGB, depth and labels buffers from a car pose.

The RGB image adds a two-layer skyline, sun-lit walls and curb and centre-line road markings.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from raycer import palette
from raycer.physics import CarState
from raycer.track import Material, Track

Array = NDArray[np.float64]

CAMERA_HEIGHT = 1.0
# Striped wall materials and their stripe width per material ID; colours come from the palette.
_WALLS = [Material.LEFT_BARRIER, Material.RIGHT_BARRIER, Material.OBSTACLE]
_STRIPE_W = np.ones(6)
_STRIPE_W[_WALLS] = 2.0, 2.0, 0.5
# Skyline sine terms (amplitude, integer frequency, phase); frequency 0 is the base height.
_SKYLINE_FAR = np.array([[0.12, 0, math.pi / 2], [0.06, 2, 0.5], [0.04, 5, 1.3], [0.02, 11, 2.1]])
_SKYLINE_NEAR = np.array([[0.06, 0, math.pi / 2], [0.05, 3, 2.0], [0.03, 7, 0.4], [0.015, 13, 4.0]])
# Horizontal world direction towards the sun, azimuth 35° measured like the heading (STYLE-04).
_SUN = np.array([math.cos(math.radians(35.0)), math.sin(math.radians(35.0))])
_CURB_WIDTH = 0.6
# Floor colour table: rows 0..5 stripe A and 6..11 stripe B per material, then road, finish, line.
_ROAD, _FINISH, _CENTER = 12, 14, 16
_CENTER_HALF_WIDTH = 0.075
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


def _skyline(phi: Array, terms: Array) -> Array:
    """Skyline layer height as a fraction of the sky half for absolute ray angles phi."""
    h = (terms[:, 0] * np.sin(terms[:, 1] * phi[:, None] + terms[:, 2])).sum(axis=1)
    return np.clip(h, 0.0, 0.25)


def _fog(colour: Array, depth: Array, max_view_distance: float, fog: Array) -> Array:
    """Blend colour towards the fog colour with factor min(1, depth/max)^1.5."""
    f = np.minimum(1.0, depth / max_view_distance) ** 1.5
    return colour + (fog - colour) * f[..., None]


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
        self._angle = np.arctan(self._k)
        # Pixel colour index into the fogged floor table is row offset + table row.
        self._floor_row = (np.arange(half) * (_CENTER + 1))[:, None]
        # Rows above the horizon counted from it: row H/2 - 1 is 1.
        self._rise = (half - np.arange(half))[:, None]
        self.set_appearance(None)

    def set_appearance(self, params: dict[str, float] | None) -> None:
        """Rebuild colours, periods and skyline phase from the palette, randomized by params."""
        c = palette.appearance_colors(params)
        self._period = 1.0 if params is None else params["period_scale"]
        self._skyline_phase = 0.0 if params is None else params["skyline_phase"]
        self._fog_colour = np.array(c["FOG"], dtype=np.float64)
        self._skyline_colours = c["SKYLINE_FAR"], c["SKYLINE_NEAR"]
        half = self.height // 2
        t = (np.arange(half) / (half - 1))[:, None]
        top, horizon = np.array(c["SKY_TOP"]), np.array(c["SKY_HORIZON"])
        self._sky = top + (horizon - top) * t
        # Every colour a sky-labelled RGB pixel can have: gradient rows and both skyline layers.
        self.sky_colors = np.vstack([np.rint(self._sky), *self._skyline_colours]).astype(np.uint8)
        self._stripe_a, self._stripe_b = np.zeros((6, 3)), np.zeros((6, 3))
        self._stripe_a[_WALLS] = c["LEFT_A"], c["RIGHT_A"], c["OBSTACLE_A"]
        self._stripe_b[_WALLS] = c["LEFT_B"], c["RIGHT_B"], c["OBSTACLE_B"]
        floor = np.vstack(
            [
                self._stripe_a,
                self._stripe_b,
                [c["ROAD_A"], c["ROAD_B"], c["FINISH_A"], c["FINISH_B"], c["CENTER_LINE"]],
            ]
        )
        # Fogged floor colours per row, flattened.
        fd = np.asarray(self._floor_depth[:, None], dtype=np.float64)
        self._floor_table = _fog(floor[None], fd, self.max_view_distance, self._fog_colour).reshape(
            -1, 3
        )

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
            # STYLE-03: far, then near mountain layer over the sky gradient.
            phi = car_state.heading - self._angle + self._skyline_phase
            far_layer = self._rise <= half * _skyline(phi, _SKYLINE_FAR)
            near_layer = self._rise <= half * _skyline(phi, _SKYLINE_NEAR)
            img[:half] = self._sky[:, None, :]
            img[:half][far_layer], img[:half][near_layer] = self._skyline_colours
            a = track.seg_a[seg]
            e = track.seg_b[seg] - a
            elen = np.linalg.norm(e, axis=1)
            eh = e / elen[:, None]
            # Unit wall normal facing the camera.
            n = np.stack([-eh[:, 1], eh[:, 0]], axis=1)
            ndr = (n * rays).sum(axis=1)
            n = np.where((ndr > 0)[:, None], -n, n)
            # STYLE-05: floor colour index into _FLOOR_COLORS, markings from the hit wall.
            road = 4.0 * self._period
            ci = _ROAD + ((np.floor(px / road) + np.floor(py / road)).astype(np.intp) & 1)
            barrier = (depth < mvd) & (
                (mat == Material.LEFT_BARRIER) | (mat == Material.RIGHT_BARRIER)
            )
            front = barrier & (fd < depth)
            dist = (depth - fd) * np.abs(ndr)
            uf = track.seg_u0[seg] + (px - a[:, 0]) * eh[:, 0] + (py - a[:, 1]) * eh[:, 1]
            stripe = mat + 6 * (np.floor(uf / self._period).astype(np.intp) & 1)
            ci = np.where(front & (dist <= _CURB_WIDTH), stripe, ci)
            pos = np.array([car_state.x, car_state.y])
            reach = mvd + float(track.width.max())
            local = np.flatnonzero(((track.center - pos) ** 2).sum(axis=1) <= reach * reach)
            if local.size:
                hx, hy = pos[:, None] + depth * rays.T
                cx, cy = track.center[local].T
                d2 = (hx[:, None] - cx) ** 2 + (hy[:, None] - cy) ** 2
                hw = track.width[local[np.argmin(d2, axis=1)]] / 2
                dash = (np.floor(uf / 3.0).astype(np.intp) & 1) == 0
                line = front & dash & (np.abs(dist - hw) <= _CENTER_HALF_WIDTH)
                ci = np.where(line, _CENTER, ci)
            cell = 0.5 * self._period
            fin = _FINISH + ((np.floor(px / cell) + np.floor(py / cell)).astype(np.intp) & 1)
            ci = np.where(inside, fin, ci)
            img[half:] = self._floor_table[self._floor_row + ci]
            u = track.seg_u0[seg] + s * elen
            odd = np.floor(u / (_STRIPE_W[mat] * self._period)) % 2 == 1
            # STYLE-04: ambient, view angle and sun term.
            shade = 0.45 + 0.25 * cos + 0.30 * np.maximum(0.0, n @ _SUN)
            wall = np.where(odd[:, None], self._stripe_b[mat], self._stripe_a[mat]) * shade[:, None]
            img = np.where(mask[..., None], _fog(wall, depth, mvd, self._fog_colour)[None], img)
            # STYLE-06: the top 8 % of the wall block is darker; it lies above the horizon.
            edge = mask[:half] & (
                self._dy[:half, None] < -0.84 * self.focal * CAMERA_HEIGHT / depth
            )
            rows, cols = np.nonzero(edge)
            img[rows, cols] = _fog(0.7 * wall[cols], depth[cols], mvd, self._fog_colour)
            rgb = np.ascontiguousarray(np.clip(np.rint(img), 0, 255).astype(np.uint8))
        return RenderOutput(rgb=rgb, depth=dbuf if self.depth_enabled else None, labels=lbuf)
