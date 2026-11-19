import dataclasses
import math
import time

import numpy as np
import pytest

from raycer.track import (
    PRESETS,
    SPLITS,
    Material,
    Track,
    TrackParams,
    _build_track,
    _resample_closed,
    _smooth_radii,
    generate_track,
    load_builtin,
    split_of,
    validate_track,
)
from test_track import NAMES, _cross, _seg_dist, _walls

ARRAYS = ["seg_a", "seg_b", "seg_material", "seg_u0", "center", "normal", "width"]


def _generate(name: str, n: int) -> tuple[list[Track], float]:
    t0 = time.perf_counter()
    tracks = [generate_track(seed, PRESETS[name]) for seed in range(n)]
    return tracks, time.perf_counter() - t0


@pytest.fixture(scope="module", params=list(PRESETS))
def generated(request: pytest.FixtureRequest) -> tuple[str, list[Track], float]:
    tracks, elapsed = _generate(request.param, 500)
    return request.param, tracks, elapsed


@pytest.fixture(scope="module")
def hard_tracks() -> list[Track]:
    return _generate("hard", 500)[0]


def _circle(radius: float) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, 2000, endpoint=False)
    return _resample_closed(np.column_stack([radius * np.cos(t), radius * np.sin(t)]), 1.0)


def _check_obstacles(t: Track, n: int) -> None:
    obstacle = t.seg_material == Material.OBSTACLE
    assert obstacle.sum() == 4 * n
    a, b = t.seg_a[obstacle], t.seg_b[obstacle]
    np.testing.assert_allclose(np.linalg.norm(b - a, axis=1), 1.5)
    left_a, left_b, right_a, right_b = _walls(t)
    step = t.length / len(t.center)
    s_values = []
    for k in range(n):
        corners = a[4 * k : 4 * k + 4]
        np.testing.assert_array_equal(b[4 * k : 4 * k + 4], np.roll(corners, -1, axis=0))
        i = int(np.argmin(np.linalg.norm(t.center - corners.mean(axis=0), axis=1)))
        s_values.append(i * step)
        edge = (corners[1] - corners[0]) / 1.5
        assert abs(float(edge @ t.normal[i])) < 1e-9
        np.testing.assert_allclose(np.linalg.norm(corners[2] - corners[0]), 1.5 * math.sqrt(2))
        assert _seg_dist(corners, left_a, left_b).min() > 0
        assert _seg_dist(corners, right_a, right_b).min() > 0
        # Free space is measured across the corridor at the obstacle's centerline point.
        lateral = (corners - t.center[i]) @ t.normal[i]
        free = t.width[i] / 2 - np.array([lateral.max(), -lateral.min()])
        assert free.min() > 0
        assert free.max() >= 4.0 - 1e-9
    s = np.array(s_values)
    assert (s >= 30).all()
    assert (s <= t.length - 30).all()
    gaps = np.abs(s[:, None] - s[None])
    assert gaps[~np.eye(n, dtype=bool)].min() >= 15


def test_generation_time(generated: tuple[str, list[Track], float]) -> None:
    # QA-30 exception to QA-22: 500 seeds per preset in under 5 s.
    assert generated[2] < 5.0


def test_generated_tracks_valid(generated: tuple[str, list[Track], float]) -> None:
    name, tracks, _ = generated
    p = PRESETS[name]
    for t in tracks:
        assert validate_track(t, p.min_curvature_radius) == []
        assert 250 <= t.length <= 1200
        # Covers "easy tracks have no obstacles" too.
        assert (t.seg_material == Material.OBSTACLE).sum() == 4 * p.n_obstacles
        assert p.w_min <= t.width.min()
        assert t.width.max() <= p.w_max
        spacing = np.linalg.norm(np.roll(t.center, -1, axis=0) - t.center, axis=1)
        np.testing.assert_allclose(spacing, 1.0, atol=0.02)
        tangent = np.roll(t.center, -1, axis=0) - np.roll(t.center, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        np.testing.assert_allclose(t.normal, np.column_stack([-tangent[:, 1], tangent[:, 0]]))
        assert _cross(tangent, t.normal).min() > 0


@pytest.mark.parametrize("name", list(PRESETS))
def test_deterministic(name: str) -> None:
    for seed in range(50):
        a, b = generate_track(seed, PRESETS[name]), generate_track(seed, PRESETS[name])
        for attr in ARRAYS:
            assert np.array_equal(getattr(a, attr), getattr(b, attr))


def test_seed_17_medium() -> None:
    assert validate_track(generate_track(17, PRESETS["medium"])) == []


def test_smooth_radii() -> None:
    def s(x: np.ndarray) -> np.ndarray:
        n = len(x)
        return np.array([(x[k - 1] + 2 * x[k] + x[(k + 1) % n]) / 4 for k in range(n)])

    u = np.array([40.0, 90.0, 40.0, 90.0])
    np.testing.assert_allclose(_smooth_radii(u), [65.0] * 4)
    rng = np.random.default_rng(0)
    for p in PRESETS.values():
        for k in range(p.k_min, p.k_max + 1):
            u = rng.uniform(p.r_min, p.r_max, k)
            r = _smooth_radii(u)
            np.testing.assert_allclose(r, s(s(u)))
            assert (r >= p.r_min).all()
            assert (r <= p.r_max).all()


def test_presets() -> None:
    assert PRESETS["easy"] == TrackParams(8, 10, 50.0, 80.0, 12.0, 16.0, 12.0, 0)
    assert PRESETS["medium"] == TrackParams(10, 12, 40.0, 90.0, 10.0, 14.0, 12.0, 3)
    assert PRESETS["hard"] == TrackParams(12, 16, 30.0, 100.0, 8.0, 12.0, 12.0, 6)
    assert TrackParams() == PRESETS["medium"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        PRESETS["medium"].k_min = 1  # type: ignore[misc]


def test_generate_value_error() -> None:
    with pytest.raises(ValueError, match="seed 0"):
        generate_track(0, TrackParams(r_min=10.0, r_max=10.0))


def test_figure_eight_self_intersects() -> None:
    t = np.linspace(0.0, 2 * np.pi, 4000, endpoint=False)
    eight = _resample_closed(np.column_stack([150 * np.sin(t), 75 * np.sin(2 * t)]), 1.0)
    errors = validate_track(_build_track(eight, np.full(len(eight), 10.0), ()))
    assert any("self-intersect" in e for e in errors)


def test_validate_length() -> None:
    center = _circle(30.0)
    errors = validate_track(_build_track(center, np.full(len(center), 10.0), ()))
    assert len(errors) == 1
    assert "length" in errors[0]


def test_validate_curvature() -> None:
    errors = validate_track(load_builtin("technical"), min_curvature_radius=20.0)
    assert len(errors) == 1
    assert "curvature" in errors[0]
    # Radius 30 m is below w/2 + 2 = 31 m.
    center = _circle(30.0)
    errors = validate_track(_build_track(center, np.full(len(center), 58.0), ()))
    assert any("curvature" in e for e in errors)


def test_validate_obstacle_outside() -> None:
    t = load_builtin("technical")
    errors = validate_track(_build_track(t.center, t.width, [(60.0, 5.0)]))
    assert errors == ["an obstacle crosses a wall", "an obstacle is not inside the corridor"]


def test_validate_separation() -> None:
    # A flat ellipse: the long sides are 12 m apart, less than 10/2 + 10/2 + 6.
    t = np.linspace(0.0, 2 * np.pi, 8000, endpoint=False)
    center = _resample_closed(np.column_stack([200 * np.cos(t), 6 * np.sin(t)]), 1.0)
    errors = validate_track(_build_track(center, np.full(len(center), 10.0), ()))
    assert any("separation" in e for e in errors)


@pytest.mark.parametrize("name", NAMES)
def test_builtins_valid(name: str) -> None:
    assert validate_track(load_builtin(name)) == []


def test_hard_obstacles(hard_tracks: list[Track]) -> None:
    for t in hard_tracks:
        _check_obstacles(t, 6)


def test_splits() -> None:
    expected_splits = {
        "train": range(0, 1_000_000),
        "val": range(1_000_000, 1_010_000),
        "test": range(1_010_000, 1_020_000),
    }
    assert expected_splits == SPLITS
    expected = {
        0: "train",
        999_999: "train",
        1_000_000: "val",
        1_009_999: "val",
        1_010_000: "test",
        1_019_999: "test",
    }
    for seed, name in expected.items():
        assert split_of(seed) == name
    for seed in (-1, 1_020_000):
        with pytest.raises(ValueError, match=str(seed)):
            split_of(seed)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(PRESETS))
def test_slow_1000_seeds(name: str) -> None:
    tracks, elapsed = _generate(name, 1000)
    assert elapsed / 1000 <= 0.1
    for t in tracks:
        assert validate_track(t, PRESETS[name].min_curvature_radius) == []
    if name == "hard":
        for t in tracks:
            _check_obstacles(t, 6)
