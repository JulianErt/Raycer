"""Per-tick progress tracking: arc length, checkpoints, laps, wrong-way, stuck (PROG-01..05)."""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from raycer.track import Track

SEARCH_WINDOW = 20.0
SEARCH_MARGIN = 2.0
WRONG_WAY_TICKS = 60


@dataclass(frozen=True)
class ProgressState:
    """Progress values after one tick (PROG-05)."""

    s: float
    total_progress: float
    lap: int
    next_checkpoint: int
    speed: float
    collided: bool
    collision_count: int
    wrong_way: bool
    stuck: bool
    checkpoint_timeout: bool
    lateral_offset: float
    heading_error: float


def _wrap_angle(a: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    a = math.remainder(a, math.tau)
    return math.pi if a == -math.pi else a


class ProgressTracker:
    """Track arc length, checkpoints, laps and progress flags of one car on a track."""

    def __init__(
        self,
        track: Track,
        dt: float = 1 / 60,
        checkpoint_timeout_s: float = 30.0,
        stuck_window_s: float = 5.0,
        stuck_min_progress: float = 1.0,
    ) -> None:
        """Precompute centerline segments and convert thresholds to tick counts."""
        self.track = track
        self._a = track.center
        self._d = np.roll(track.center, -1, axis=0) - track.center
        self._seg_len = np.linalg.norm(self._d, axis=1)
        self._s0 = np.concatenate([[0.0], np.cumsum(self._seg_len[:-1])])
        self._seg_len2 = self._seg_len**2
        self._everything = np.arange(len(self._a))
        # A segment starting up to one step before the window still reaches into it.
        self._window = SEARCH_WINDOW + self._seg_len.max()
        self._timeout_ticks = round(checkpoint_timeout_s / dt)
        self._stuck_ticks = round(stuck_window_s / dt)
        self._stuck_min = stuck_min_progress
        self._history: deque[float] = deque(maxlen=self._stuck_ticks + 1)
        self._state: ProgressState
        self._ticks_since_cp = 0
        self._wrong_ticks = 0

    def _project(
        self, x: float, y: float, idx: NDArray[np.intp]
    ) -> tuple[float, int, float, float]:
        """Project (x, y) onto the segments idx; return s, segment, t and distance."""
        p = np.array([x, y])
        a, d = self._a[idx], self._d[idx]
        t = np.minimum(np.maximum(np.einsum("nk,nk->n", p - a, d) / self._seg_len2[idx], 0.0), 1.0)
        off = a + t[:, None] * d - p
        dist = np.sqrt(np.add.reduce(off * off, axis=1))
        j = int(np.argmin(dist))
        i, tj = int(idx[j]), float(t[j])
        s = (self._s0[i] + tj * self._seg_len[i]) % self.track.length
        return s, i, tj, float(dist[j])

    def _locate(self, x: float, y: float, s_old: float | None) -> tuple[float, int, float]:
        """Find s, segment and t; local window around s_old with global fallback."""
        if s_old is not None:
            length = self.track.length
            gap = np.abs((self._s0 - s_old + length / 2) % length - length / 2)
            local = np.flatnonzero(gap <= self._window)
            s, i, t, dist = self._project(x, y, local)
            w = (1 - t) * self.track.width[i] + t * self.track.width[(i + 1) % len(self._a)]
            if dist <= w / 2 + SEARCH_MARGIN:
                return s, i, t
        s, i, t, _ = self._project(x, y, self._everything)
        return s, i, t

    def _state_at(
        self, x: float, y: float, heading: float, i: int, t: float
    ) -> tuple[float, float, float]:
        """Return lateral offset, heading error and tangent angle at the projection."""
        j = (i + 1) % len(self._a)
        n = (1 - t) * self.track.normal[i] + t * self.track.normal[j]
        n /= math.sqrt(n.dot(n))
        c = self._a[i] + t * self._d[i]
        lateral = float(np.dot(np.array([x, y]) - c, n))
        tangent = math.atan2(-float(n[0]), float(n[1]))
        return lateral, _wrap_angle(heading - tangent), tangent

    def reset(self, x: float, y: float, heading: float) -> ProgressState:
        """Start a new episode at the given pose with a global search for s."""
        s, i, t = self._locate(x, y, None)
        lateral, heading_error, _ = self._state_at(x, y, heading, i, t)
        self._ticks_since_cp = 0
        self._wrong_ticks = 0
        self._history.clear()
        self._history.append(0.0)
        self._state = ProgressState(
            s=s,
            total_progress=0.0,
            lap=0,
            next_checkpoint=1,
            speed=0.0,
            collided=False,
            collision_count=0,
            wrong_way=False,
            stuck=False,
            checkpoint_timeout=False,
            lateral_offset=lateral,
            heading_error=heading_error,
        )
        return self._state

    def update(
        self, x: float, y: float, heading: float, speed: float, collided: bool
    ) -> ProgressState:
        """Advance one physics tick (after reset) and return the new state."""
        old = self._state
        length = self.track.length
        s, i, t = self._locate(x, y, old.s)
        lateral, heading_error, tangent = self._state_at(x, y, heading, i, t)
        ds = length / 2 - (length / 2 - (s - old.s)) % length
        total = old.total_progress + ds

        lap, k = old.lap, old.next_checkpoint
        self._ticks_since_cp += 1
        cps = self.track.checkpoint_s
        if ds > 0 and (cps[k] - old.s) % length < ds:
            if k == 0:
                lap += 1
            k = (k + 1) % len(cps)
            self._ticks_since_cp = 0

        wrong = math.cos(heading - tangent) < -0.5 and abs(speed) > 1.0
        self._wrong_ticks = self._wrong_ticks + 1 if wrong else 0

        self._history.append(total)
        stuck = (
            len(self._history) > self._stuck_ticks and total - self._history[0] < self._stuck_min
        )

        self._state = ProgressState(
            s=s,
            total_progress=total,
            lap=lap,
            next_checkpoint=k,
            speed=speed,
            collided=collided,
            collision_count=old.collision_count + collided,
            wrong_way=self._wrong_ticks >= WRONG_WAY_TICKS,
            stuck=stuck,
            checkpoint_timeout=self._ticks_since_cp >= self._timeout_ticks,
            lateral_offset=lateral,
            heading_error=heading_error,
        )
        return self._state
