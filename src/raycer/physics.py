"""Kinematic bicycle car physics (PHYS-01..07) with circle-vs-segment wall collision."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from raycer.track import Track

DT: float = 1 / 60
_COLLISION_ITERATIONS = 3
_PUSH_EPS = 1e-4


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
    idx = track.segments_near((x, y))
    a, b = track.seg_a[idx], track.seg_b[idx]
    d = b - a
    dd = np.einsum("nk,nk->n", d, d)
    pushed = False
    for _ in range(_COLLISION_ITERATIONS):
        t = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / dd, 0.0, 1.0)
        dist = np.hypot(x - (a[:, 0] + t * d[:, 0]), y - (a[:, 1] + t * d[:, 1]))
        hits = np.flatnonzero(dist < r_car)
        if len(hits) == 0:
            break
        pushed = True
        # Sequential pushes, each against the current position, so corners resolve correctly.
        for i in hits.tolist():
            ax, ay, dx, dy = float(a[i, 0]), float(a[i, 1]), float(d[i, 0]), float(d[i, 1])
            ti = _clip(((x - ax) * dx + (y - ay) * dy) / float(dd[i]), 0.0, 1.0)
            cx, cy = ax + ti * dx, ay + ti * dy
            ox, oy = x - cx, y - cy
            norm = math.hypot(ox, oy)
            if norm >= r_car:
                continue
            if norm == 0.0:
                # Centre exactly on the segment: push along the segment's left normal.
                seg_len = math.hypot(dx, dy)
                ox, oy, norm = -dy / seg_len, dx / seg_len, 1.0
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
