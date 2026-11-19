"""Kinematic bicycle car physics (PHYS-01..07) with circle-vs-segment wall collision."""

import math
import weakref
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from raycer.track import CELL_SIZE, Track

Array = NDArray[np.float64]

DT: float = 1 / 60
_COLLISION_ITERATIONS = 3
_PUSH_EPS = 1e-4
# Per track and grid cell: nearby segment starts, directions and squared lengths.
_NEAR: weakref.WeakKeyDictionary[Track, dict[tuple[int, int], tuple[Array, Array, Array]]] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True)
class CarParams:
    """Car constants of PHYS-03."""

    L: float = 2.5
    r_car: float = 1.0
    v_max: float = 30.0
    v_rev_max: float = 5.0
    a_engine: float = 8.0
    a_brake: float = 16.0
    c_roll: float = 1.0
    delta_0: float = 0.5
    v_ref: float = 15.0


@dataclass(frozen=True)
class CarState:
    """Car pose and signed speed: x, y in m, heading in rad, v in m/s."""

    x: float
    y: float
    heading: float
    v: float


class Action(NamedTuple):
    """Internal action: steer in [-1, 1] (> 0 is left), throttle and brake in [0, 1]."""

    steer: float
    throttle: float
    brake: float


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


def _sign(value: float) -> float:
    return float((value > 0) - (value < 0))


def _resolve(x: float, y: float, track: Track, r_car: float) -> tuple[float, float, bool]:
    """Push the centre out of all segments closer than r_car; return x, y and whether it moved."""
    cells = _NEAR.setdefault(track, {})
    key = (math.floor(x / CELL_SIZE), math.floor(y / CELL_SIZE))
    if key not in cells:
        idx = track.segments_near((x, y))
        a, b = track.seg_a[idx], track.seg_b[idx]
        d = b - a
        cells[key] = (a, d, np.einsum("nk,nk->n", d, d))
    a, d, dd = cells[key]
    ax, ay, dx, dy = a[:, 0], a[:, 1], d[:, 0], d[:, 1]
    pushed = False
    for _ in range(_COLLISION_ITERATIONS):
        t = np.minimum(np.maximum(((x - ax) * dx + (y - ay) * dy) / dd, 0.0), 1.0)
        dist = np.hypot(x - (ax + t * dx), y - (ay + t * dy))
        hits = np.flatnonzero(dist < r_car)
        if len(hits) == 0:
            break
        pushed = True
        # Sequential pushes, each against the current position, so corners resolve correctly.
        for i in hits.tolist():
            sx, sy, ex, ey = float(ax[i]), float(ay[i]), float(dx[i]), float(dy[i])
            ti = _clip(((x - sx) * ex + (y - sy) * ey) / float(dd[i]), 0.0, 1.0)
            cx, cy = sx + ti * ex, sy + ti * ey
            ox, oy = x - cx, y - cy
            norm = math.hypot(ox, oy)
            if norm >= r_car:
                continue
            if norm == 0.0:
                # Centre exactly on the segment: push along the segment's left normal.
                seg_len = math.hypot(ex, ey)
                ox, oy, norm = -ey / seg_len, ex / seg_len, 1.0
            x = cx + ox / norm * (r_car + _PUSH_EPS)
            y = cy + oy / norm * (r_car + _PUSH_EPS)
    return x, y, pushed


def step(state: CarState, action: Action, params: CarParams, track: Track) -> tuple[CarState, bool]:
    """Advance the car by one tick of DT; return the new state and whether it collided."""
    p = params
    steer = _clip(action[0], -1.0, 1.0)
    throttle = _clip(action[1], 0.0, 1.0)
    brake = _clip(action[2], 0.0, 1.0)
    v = state.v

    a = p.a_engine * throttle - p.c_roll * _sign(v)
    if v > 0.1:
        a -= p.a_brake * brake
    elif brake > 0:
        a = -p.a_engine * brake
    if v < 0 and throttle > 0:
        a += p.a_brake * throttle

    v_new = _clip(v + a * DT, -p.v_rev_max, p.v_max)
    if v != 0 and throttle == 0 and brake == 0 and _sign(v_new) == -_sign(v):
        v_new = 0.0

    delta = steer * p.delta_0 / (1 + abs(v_new) / p.v_ref)
    heading = state.heading + v_new / p.L * math.tan(delta) * DT
    if heading > math.pi:
        heading -= 2 * math.pi
    elif heading <= -math.pi:
        heading += 2 * math.pi

    x = state.x + v_new * math.cos(heading) * DT
    y = state.y + v_new * math.sin(heading) * DT

    x, y, collided = _resolve(x, y, track, p.r_car)
    if collided:
        v_new = 0.5 * v_new
    return CarState(x, y, heading, v_new), collided


def start_state(track: Track, lateral_offset: float = 0.0) -> CarState:
    """Return the PHYS-06 start: centre[0] shifted left by lateral_offset, along the tangent."""
    c0, n0 = track.center[0], track.normal[0]
    heading = math.atan2(float(-n0[0]), float(n0[1]))
    return CarState(
        float(c0[0] + lateral_offset * n0[0]),
        float(c0[1] + lateral_offset * n0[1]),
        math.pi if heading == -math.pi else heading,
        0.0,
    )
