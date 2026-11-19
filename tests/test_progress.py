import dataclasses
import itertools
import math

import numpy as np
import pytest

from raycer.progress import ProgressState, ProgressTracker
from raycer.track import Track, load_builtin

STEP = 0.5


@pytest.fixture(scope="module")
def oval() -> Track:
    return load_builtin("oval")


def _pose(track: Track, s: float) -> tuple[float, float, float]:
    """Centerline point and tangent heading at arc length s (interpolated)."""
    closed = np.vstack([track.center, track.center[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s %= track.length
    i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
    t = (s - cum[i]) / seg[i]
    d = closed[i + 1] - closed[i]
    p = closed[i] + t * d
    return float(p[0]), float(p[1]), math.atan2(float(d[1]), float(d[0]))


def _drive(
    tracker: ProgressTracker, track: Track, s_from: float, s_to: float, speed: float = 5.0
) -> list[ProgressState]:
    """Update along the centerline from s_from to s_to in steps of at most STEP."""
    n = max(1, math.ceil(abs(s_to - s_from) / STEP))
    states = []
    for s in np.linspace(s_from, s_to, n + 1)[1:]:
        x, y, h = _pose(track, float(s))
        states.append(tracker.update(x, y, h, speed, False))
    return states


def _reset(tracker: ProgressTracker, track: Track, s: float) -> ProgressState:
    return tracker.reset(*_pose(track, s))


def test_one_lap_progress_equals_length(oval: Track) -> None:
    tr = ProgressTracker(oval)
    _reset(tr, oval, 0.0)
    states = _drive(tr, oval, 0.0, oval.length)
    assert abs(states[-1].total_progress - oval.length) <= 0.5
    assert all(0.0 <= st.s < oval.length for st in states)


def test_reverse_lap_decreases_progress(oval: Track) -> None:
    tr = ProgressTracker(oval)
    _reset(tr, oval, 0.0)
    states = _drive(tr, oval, oval.length, 0.0)
    assert states[0].total_progress < 0
    assert all(b.total_progress < a.total_progress for a, b in itertools.pairwise(states))
    assert abs(states[-1].total_progress + oval.length) <= 0.5
    assert all(0.0 <= st.s < oval.length for st in states)


def test_reset_global_search_and_far_jump() -> None:
    kidney = load_builtin("kidney")
    tr = ProgressTracker(kidney)
    assert _reset(tr, kidney, 100.0).s == pytest.approx(100.0, abs=0.05)
    far = 100.0 + kidney.length / 2
    st = tr.update(*_pose(kidney, far), 0.0, False)
    assert st.s == pytest.approx(far % kidney.length, abs=0.05)


@pytest.mark.parametrize("name", ["oval", "kidney", "technical"])
def test_three_laps(name: str) -> None:
    track = load_builtin(name)
    tr = ProgressTracker(track)
    _reset(tr, track, 0.0)
    states = _drive(tr, track, 0.0, 3 * track.length + 1.0)
    assert states[-1].lap == 3
    assert states[-1].next_checkpoint == 1
    assert all(0.0 <= st.s < track.length for st in states)


def test_checkpoint_order(oval: Track) -> None:
    tr = ProgressTracker(oval)
    assert _reset(tr, oval, 0.0).next_checkpoint == 1
    cps = oval.checkpoint_s
    c = len(cps)
    assert _drive(tr, oval, 0.0, cps[1] + 1.0)[-1].next_checkpoint == 2
    st = _drive(tr, oval, cps[1] + 1.0, cps[c - 1] + 1.0)[-1]
    assert st.next_checkpoint == 0
    assert st.lap == 0


def test_backwards_over_finish_does_not_count(oval: Track) -> None:
    tr = ProgressTracker(oval)
    _reset(tr, oval, 0.0)
    _drive(tr, oval, 0.0, -5.0)
    st = _drive(tr, oval, -5.0, 5.0)[-1]
    assert st.lap == 0
    assert st.next_checkpoint == 1


def test_repeated_crossing_counts_once(oval: Track) -> None:
    tr = ProgressTracker(oval)
    _reset(tr, oval, 0.0)
    cp = float(oval.checkpoint_s[1])
    seen = [st.next_checkpoint for st in _drive(tr, oval, 0.0, cp + 3.0)]
    seen += [st.next_checkpoint for st in _drive(tr, oval, cp + 3.0, cp - 3.0)]
    seen += [st.next_checkpoint for st in _drive(tr, oval, cp - 3.0, cp + 3.0)]
    assert seen[-1] == 2
    assert sum(b != a for a, b in itertools.pairwise(seen)) == 1


def test_wrong_way_after_60_ticks(oval: Track) -> None:
    tr = ProgressTracker(oval)
    _reset(tr, oval, 100.0)
    s = 100.0
    for tick in range(1, 61):
        s -= 5.0 / 60
        x, y, h = _pose(oval, s)
        st = tr.update(x, y, h + math.pi, 5.0, False)
        assert st.wrong_way == (tick >= 60)
    x, y, h = _pose(oval, s)
    assert not tr.update(x, y, h + math.pi, 0.0, False).wrong_way
    for _ in range(60):
        tr.update(x, y, h + math.pi, 5.0, False)
    assert tr.update(x, y, h + math.pi, 5.0, False).wrong_way
    assert not tr.update(x, y, h, 5.0, False).wrong_way


def _stand(tr: ProgressTracker, track: Track, n: int) -> list[ProgressState]:
    x, y, h = _pose(track, 0.0)
    tr.reset(x, y, h)
    return [tr.update(x, y, h, 0.0, False) for _ in range(n)]


def test_stuck_after_exactly_300_ticks(oval: Track) -> None:
    states = _stand(ProgressTracker(oval), oval, 300)
    assert not states[298].stuck
    assert states[299].stuck


def test_stuck_not_set_while_progressing(oval: Track) -> None:
    tr = ProgressTracker(oval)
    _reset(tr, oval, 0.0)
    assert not any(st.stuck for st in _drive(tr, oval, 0.0, 200.0))


def test_checkpoint_timeout_after_1800_ticks(oval: Track) -> None:
    states = _stand(ProgressTracker(oval), oval, 1800)
    assert not states[1798].checkpoint_timeout
    assert states[1799].checkpoint_timeout


def test_thresholds_configurable(oval: Track) -> None:
    tr = ProgressTracker(oval, checkpoint_timeout_s=1.0, stuck_window_s=2.0)
    states = _stand(tr, oval, 120)
    assert [st.checkpoint_timeout for st in states[58:60]] == [False, True]
    assert [st.stuck for st in states[118:120]] == [False, True]


def test_checkpoint_resets_timeout(oval: Track) -> None:
    tr = ProgressTracker(oval, checkpoint_timeout_s=1.0)
    _reset(tr, oval, 0.0)
    cp = float(oval.checkpoint_s[1])
    states = _drive(tr, oval, 0.0, cp + 0.25)
    assert states[-1].next_checkpoint == 2
    # 100 ticks in total: past the 60-tick timeout unless the pass reset it.
    states = _drive(tr, oval, cp + 0.25, 50.0)
    assert len(states) < 60
    assert not states[-1].checkpoint_timeout


def test_state_fields() -> None:
    assert [f.name for f in dataclasses.fields(ProgressState)] == [
        "s",
        "total_progress",
        "lap",
        "next_checkpoint",
        "speed",
        "collided",
        "collision_count",
        "wrong_way",
        "stuck",
        "checkpoint_timeout",
        "lateral_offset",
        "heading_error",
    ]


def test_lateral_offset_and_heading_error(oval: Track) -> None:
    tr = ProgressTracker(oval)
    assert tr.reset(0.0, -33.0, 0.0).lateral_offset == pytest.approx(2.0, abs=0.02)
    assert tr.reset(0.0, -37.0, 0.0).lateral_offset == pytest.approx(-2.0, abs=0.02)
    assert tr.reset(0.0, -35.0, 0.3).heading_error == pytest.approx(0.3, abs=0.02)
    assert tr.reset(0.0, -35.0, -0.3).heading_error == pytest.approx(-0.3, abs=0.02)
    err = tr.reset(0.0, -35.0, math.pi).heading_error
    assert -math.pi < err <= math.pi
    assert abs(err) == pytest.approx(math.pi, abs=0.02)
    assert tr.update(0.0, -33.0, 0.3, 4.0, False).lateral_offset == pytest.approx(2.0, abs=0.02)


def test_collision_count_and_speed(oval: Track) -> None:
    tr = ProgressTracker(oval)
    x, y, h = _pose(oval, 0.0)
    tr.reset(x, y, h)
    assert tr.update(x, y, h, -2.5, True).collided
    tr.update(x, y, h, 1.0, True)
    st = tr.update(x, y, h, 3.0, False)
    assert st.collision_count == 2
    assert not st.collided
    assert st.speed == 3.0
