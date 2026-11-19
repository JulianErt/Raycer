import dataclasses
import hashlib
import math
import struct

import numpy as np
import pytest
from numpy.typing import NDArray

from raycer.physics import DT, Action, CarParams, CarState, start_state, step
from raycer.track import Material, Track, load_builtin

NAMES = ["oval", "kidney", "technical"]
P = CarParams()

Array = NDArray[np.float64]


@pytest.fixture(scope="module")
def oval() -> Track:
    return load_builtin("oval")


def _seg_dist(p: Array, a: Array, b: Array) -> Array:
    """Distance from each point p (P,2) to each segment (N,2) -> (P,N)."""
    d = b - a
    t = np.einsum("pnk,nk->pn", p[:, None, :] - a[None], d) / np.einsum("nk,nk->n", d, d)
    closest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * d[None]
    return np.linalg.norm(p[:, None, :] - closest, axis=2)


def _inside(points: Array, poly: Array) -> NDArray[np.bool_]:
    """Even-odd point-in-polygon test of points (P,2) against a closed polygon (N,2)."""
    a, b = poly[None], np.roll(poly, -1, axis=0)[None]
    px, py = points[:, None, 0], points[:, None, 1]
    straddle = (a[..., 1] > py) != (b[..., 1] > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = a[..., 0] + (py - a[..., 1]) * (b[..., 0] - a[..., 0]) / (b[..., 1] - a[..., 1])
    return np.asarray((straddle & (px < x_cross)).sum(axis=1) % 2 == 1)


def _run(state: CarState, action: Action, track: Track, n: int) -> tuple[CarState, list[bool]]:
    hits = []
    for _ in range(n):
        state, collided = step(state, action, P, track)
        hits.append(collided)
    return state, hits


def _episode(track: Track, rng: np.random.Generator, n: int) -> tuple[Array, bool]:
    """Run n ticks with a new uniform random action every 4 ticks; return centres and any hit."""
    state, any_hit = start_state(track), False
    points = np.empty((n, 2))
    for i in range(n):
        if i % 4 == 0:
            # brake is drawn from [-1, 1] so clipping leaves it 0 half the time and the car
            # gains speed; with brake from [0, 1] it barely moves and never reaches a wall.
            action = Action(rng.uniform(-1, 1), rng.uniform(0, 1), rng.uniform(-1, 1))
        state, collided = step(state, action, P, track)
        any_hit |= collided
        points[i] = state.x, state.y
    return points, any_hit


def _check_contained(track: Track, points: Array) -> None:
    mat = track.seg_material
    assert _inside(points, track.seg_a[mat == Material.RIGHT_BARRIER]).all()
    assert not _inside(points, track.seg_a[mat == Material.LEFT_BARRIER]).any()
    obstacles = track.seg_a[mat == Material.OBSTACLE]
    for k in range(0, len(obstacles), 4):
        assert not _inside(points, obstacles[k : k + 4]).any()


def test_defaults() -> None:
    assert dataclasses.asdict(CarParams()) == {
        "L": 2.5,
        "r_car": 1.0,
        "v_max": 30.0,
        "v_rev_max": 5.0,
        "a_engine": 8.0,
        "a_brake": 16.0,
        "c_roll": 1.0,
        "delta_0": 0.5,
        "v_ref": 15.0,
    }
    assert DT == 1 / 60


def test_action_clipping(oval: Track) -> None:
    s = CarState(0.0, -35.0, 0.3, 4.0)
    assert step(s, Action(5, 3, 0), P, oval) == step(s, Action(1, 1, 0), P, oval)
    assert step(s, Action(-5, -1, 7), P, oval) == step(s, Action(-1, 0, 1), P, oval)


def test_full_throttle_reaches_v_max(oval: Track) -> None:
    state, hits = _run(CarState(-1000.0, -35.0, 0.0, 0.0), Action(0, 1, 0), oval, 600)
    assert not any(hits)
    assert abs(state.v - 30) <= 1e-9


def test_rolling_stops_exactly(oval: Track) -> None:
    state = dataclasses.replace(start_state(oval), v=10.0)
    hits = []
    for _ in range(math.ceil(10.1 / DT)):
        state, collided = step(state, Action(0, 0, 0), P, oval)
        hits.append(collided)
        if state.v == 0.0:
            break
    assert state.v == 0.0
    state, more = _run(state, Action(0, 0, 0), oval, 600)
    assert state.v == 0.0
    assert not any(hits + more)


def test_turn_radius(oval: Track) -> None:
    delta = 0.5 / (1 + 5 / 15)
    expected = P.L / math.tan(delta)
    n = math.ceil(2 * math.pi * expected / 5 / DT)
    state, total = CarState(0.0, -1000.0, 0.0, 5.0), 0.0
    for _ in range(n):
        new, collided = step(state, Action(1, 0.125, 0), P, oval)
        assert not collided
        assert new.v == 5.0
        total += math.remainder(new.heading - state.heading, 2 * math.pi)
        state = new
    assert total >= 2 * math.pi
    radius = 5.0 * n * DT / total
    assert radius == pytest.approx(expected, rel=0.01)


def test_heading_normalised(oval: Track) -> None:
    state = CarState(0.0, -1000.0, math.pi - 1e-3, 10.0)
    wrapped = False
    for _ in range(60):
        state, _ = step(state, Action(1, 0.5, 0), P, oval)
        assert -math.pi < state.heading <= math.pi
        wrapped |= state.heading < 0
    assert wrapped


def test_zero_action_is_identity(oval: Track) -> None:
    s = start_state(oval)
    new, collided = step(s, Action(0, 0, 0), P, oval)
    assert new == s
    assert collided is False


def test_steer_zero_keeps_heading(oval: Track) -> None:
    rng = np.random.default_rng(1)
    state = CarState(0.0, -1000.0, 0.7, 12.0)
    for _ in range(120):
        state, _ = step(state, Action(0, rng.uniform(0, 1), rng.uniform(0, 1)), P, oval)
        assert state.heading == 0.7


def test_wall_collision(oval: Track) -> None:
    # The bottom outer wall lies at y = -42; the car starts 1.2 m above it.
    s = CarState(0.0, -40.8, -math.pi / 2, 30.0)
    new, collided = step(s, Action(0, 1, 0), P, oval)
    assert collided is True
    assert new.v == 15.0
    assert new.heading == s.heading
    assert _seg_dist(np.array([[new.x, new.y]]), oval.seg_a, oval.seg_b).min() >= P.r_car


def test_contained_random_episodes() -> None:
    any_hit = False
    for name in NAMES:
        track = load_builtin(name)
        for seed in range(5):
            points, hit = _episode(track, np.random.default_rng(seed), 2000)
            _check_contained(track, points)
            any_hit |= hit
    assert any_hit


@pytest.mark.slow
def test_contained_random_episodes_full() -> None:
    # generate_track is not on main yet, so this cycles over the builtin tracks.
    tracks = [load_builtin(name) for name in NAMES]
    for episode in range(200):
        track = tracks[episode % len(tracks)]
        rng = np.random.default_rng([episode % 20, episode])
        points, _ = _episode(track, rng, 2000)
        _check_contained(track, points)


def test_step_is_pure(oval: Track) -> None:
    s, a, p = CarState(1.0, -36.0, 0.2, 7.0), Action(0.4, 0.6, 0.1), CarParams()
    seg_a = oval.seg_a.copy()
    assert step(s, a, p, oval) == step(s, a, p, oval)
    assert s == CarState(1.0, -36.0, 0.2, 7.0)
    assert a == Action(0.4, 0.6, 0.1)
    assert p == CarParams()
    np.testing.assert_array_equal(oval.seg_a, seg_a)


def test_determinism_hash() -> None:
    track = load_builtin("technical")
    actions = np.random.default_rng(7).uniform([-1, 0, 0], [1, 1, 1], size=(1000, 3))
    digests = []
    for _ in range(2):
        h, state = hashlib.sha256(), start_state(track)
        for row in actions:
            state, _ = step(state, Action(*row.tolist()), P, track)
            h.update(struct.pack("<4d", state.x, state.y, state.heading, state.v))
        digests.append(h.hexdigest())
    assert digests[0] == digests[1]


def test_start_state(oval: Track) -> None:
    s = start_state(oval)
    assert (s.x, s.y, s.heading, s.v) == pytest.approx((0.0, -35.0, 0.0, 0.0), abs=1e-12)
    assert start_state(oval, 2.0).y == pytest.approx(-33.0, abs=1e-12)
