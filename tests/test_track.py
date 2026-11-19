import hashlib
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from raycer.track import Material, Track, load_builtin

NAMES = ["oval", "kidney", "technical"]
HASHES = {
    "oval": "ae98ca517e4121a6cfbd3d003fca29a43280d2eef5ff58222462e894ead4648b",
    "kidney": "37d7c4294f4750193d81e28f00f17b1d6f2462d3230c22b0a2bf7bc615a0462d",
    "technical": "9d4a4d2d78086e8e3deac7cb0a5f16607faf9b9ab5e95f1df315300b0ba18b20",
}

Array = NDArray[np.float64]


@pytest.fixture(scope="module", params=NAMES)
def track(request: pytest.FixtureRequest) -> Track:
    return load_builtin(request.param)


def _seg_dist(p: Array, a: Array, b: Array) -> Array:
    """Distance from each point p (P,2) to each segment (N,2) -> (P,N)."""
    d = b - a
    t = np.einsum("pnk,nk->pn", p[:, None, :] - a[None], d) / np.einsum("nk,nk->n", d, d)
    closest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * d[None]
    return np.linalg.norm(p[:, None, :] - closest, axis=2)


def _cross(u: Array, v: Array) -> Array:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _crossings(a1: Array, b1: Array, a2: Array, b2: Array) -> NDArray[np.bool_]:
    """Proper intersection matrix between segment sets 1 (N) and 2 (M)."""
    d1, d2 = (b1 - a1)[:, None], (b2 - a2)[None]
    o1 = _cross(d1, a2[None] - a1[:, None])
    o2 = _cross(d1, b2[None] - a1[:, None])
    o3 = _cross(d2, a1[:, None] - a2[None])
    o4 = _cross(d2, b1[:, None] - a2[None])
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def _radii(center: Array) -> Array:
    p, q, r = np.roll(center, 3, axis=0), center, np.roll(center, -3, axis=0)
    a, b, c = (np.linalg.norm(u - v, axis=1) for u, v in ((p, q), (q, r), (r, p)))
    area2 = np.abs(_cross(q - p, r - p))
    with np.errstate(divide="ignore"):
        return a * b * c / (2 * area2)


def _walls(t: Track) -> tuple[Array, Array, Array, Array]:
    left = t.seg_material == Material.LEFT_BARRIER
    right = t.seg_material == Material.RIGHT_BARRIER
    return t.seg_a[left], t.seg_b[left], t.seg_a[right], t.seg_b[right]


def _signed_area(p: Array) -> float:
    return 0.5 * float(_cross(p, np.roll(p, -1, axis=0)).sum())


def test_material_values(track: Track) -> None:
    assert Material.LEFT_BARRIER == 2
    assert Material.RIGHT_BARRIER == 3
    assert Material.OBSTACLE == 5
    assert set(np.unique(track.seg_material)) <= {2, 3, 5}


def test_shapes_dtypes_and_u0(track: Track) -> None:
    n, m, c = len(track.seg_a), len(track.center), len(track.checkpoint_s)
    for arr, shape, dtype in [
        (track.seg_a, (n, 2), np.float64),
        (track.seg_b, (n, 2), np.float64),
        (track.seg_material, (n,), np.uint8),
        (track.seg_u0, (n,), np.float64),
        (track.center, (m, 2), np.float64),
        (track.normal, (m, 2), np.float64),
        (track.width, (m,), np.float64),
        (track.checkpoint_s, (c,), np.float64),
        (track.finish_quad, (4, 2), np.float64),
    ]:
        assert arr.shape == shape
        assert arr.dtype == dtype
    assert isinstance(track.length, float)
    np.testing.assert_allclose(np.linalg.norm(track.normal, axis=1), 1.0)
    assert _signed_area(track.center) > 0
    lengths = np.linalg.norm(track.seg_b - track.seg_a, axis=1)
    wall = track.seg_material != Material.OBSTACLE
    assert lengths[wall].max() <= 2.1
    # Each polyline starts where seg_u0 is 0 and is chained (b of one = a of the next).
    starts = np.flatnonzero(track.seg_u0 == 0.0)
    assert starts[0] == 0
    assert len(starts) == 2 + int((track.seg_material == Material.OBSTACLE).sum()) // 4
    for lo, hi in zip(starts, [*starts[1:], n], strict=True):
        u = track.seg_u0[lo:hi]
        np.testing.assert_allclose(np.diff(u), lengths[lo : hi - 1])
        np.testing.assert_array_equal(track.seg_b[lo : hi - 1], track.seg_a[lo + 1 : hi])
        np.testing.assert_array_equal(track.seg_b[hi - 1], track.seg_a[lo])
        assert len(np.unique(track.seg_material[lo:hi])) == 1


def test_checkpoints_and_finish(track: Track) -> None:
    c = max(8, round(track.length / 25))
    assert len(track.checkpoint_s) == c
    assert track.checkpoint_s[0] == 0.0
    np.testing.assert_array_equal(track.checkpoint_s, np.arange(c) * track.length / c)
    quad, c0, w0 = track.finish_quad, track.center[0], track.width[0]
    assert (np.linalg.norm(quad - c0, axis=1) <= w0 / 2 + 0.6).all()
    left_a, left_b, right_a, right_b = _walls(track)
    # Right edge corners lie on the right wall, left edge corners on the left wall.
    assert _seg_dist(quad[:2], right_a, right_b).min(axis=1).max() < 0.05
    assert _seg_dist(quad[2:], left_a, left_b).min(axis=1).max() < 0.05
    n0 = track.normal[0]
    along = (quad - c0) @ np.array([n0[1], -n0[0]])
    assert along.max() - along.min() == pytest.approx(1.0, abs=1e-6)
    assert np.abs((quad - c0) @ n0).max() == pytest.approx(w0 / 2)


def test_culling_superset_of_brute_force(track: Track) -> None:
    rng = np.random.default_rng(0)
    lo, hi = track.center.min(axis=0) - 20, track.center.max(axis=0) + 20
    points = rng.uniform(lo, hi, size=(120, 2))
    dist = _seg_dist(points, track.seg_a, track.seg_b)
    everything = np.arange(len(track.seg_a))
    for p, d in zip(points, dist, strict=True):
        got = track.segments_in_radius(p, 60.0)
        assert got.dtype == np.intp
        assert (np.diff(got) > 0).all()
        assert set(np.flatnonzero(d <= 60.0)) <= set(got)
        near = track.segments_near(p)
        assert (np.diff(near) > 0).all()
        assert set(np.flatnonzero(d <= 20.0)) <= set(near)
        np.testing.assert_array_equal(track.segments_in_radius(p, 60.0, cull=False), everything)
        np.testing.assert_array_equal(track.segments_near(p, cull=False), everything)


def test_culling_far_point_is_empty() -> None:
    oval = load_builtin("oval")
    assert oval.segments_in_radius((1000.0, 1000.0), 60.0).size == 0
    assert oval.segments_near((1000.0, 1000.0)).size == 0


def test_load_builtin_unknown() -> None:
    with pytest.raises(ValueError, match="nope"):
        load_builtin("nope")


@pytest.mark.parametrize("name", NAMES)
def test_load_builtin_deterministic(name: str) -> None:
    a, b = load_builtin(name), load_builtin(name)
    for attr in ["seg_a", "seg_b", "seg_material", "seg_u0", "center", "normal", "width"]:
        assert getattr(a, attr).tobytes() == getattr(b, attr).tobytes()


@pytest.mark.parametrize("name", NAMES)
def test_pinned_hash(name: str) -> None:
    t = load_builtin(name)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.round(t.seg_a, 6) + 0.0, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(np.round(t.seg_b, 6) + 0.0, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(t.seg_material, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(np.round(t.seg_u0, 6) + 0.0, dtype=np.float64).tobytes())
    assert h.hexdigest() == HASHES[name]


def test_oval_geometry() -> None:
    t = load_builtin("oval")
    assert t.length == pytest.approx(240 + 70 * math.pi, abs=0.1)
    assert (t.width == 14.0).all()
    assert not (t.seg_material == Material.OBSTACLE).any()
    np.testing.assert_allclose(t.center[0], (0.0, -35.0), atol=1e-12)
    np.testing.assert_allclose(t.normal[0], (0.0, 1.0), atol=1e-12)


def test_kidney_geometry() -> None:
    t = load_builtin("kidney")
    assert (t.width == 12.0).all()
    assert not (t.seg_material == Material.OBSTACLE).any()
    tangent = np.roll(t.center, -1, axis=0) - t.center
    turn = _cross(tangent, np.roll(tangent, -1, axis=0))
    assert turn.max() > 0
    assert turn.min() < 0


def test_technical_geometry() -> None:
    t = load_builtin("technical")
    assert (t.width == 10.0).all()
    assert 12.0 <= _radii(t.center).min() <= 15.0
    obstacle = t.seg_material == Material.OBSTACLE
    assert obstacle.sum() == 16
    a, b = t.seg_a[obstacle], t.seg_b[obstacle]
    np.testing.assert_allclose(np.linalg.norm(b - a, axis=1), 1.5)
    left_a, left_b, right_a, right_b = _walls(t)
    step = t.length / len(t.center)
    s_values = []
    for k in range(4):
        corners = a[4 * k : 4 * k + 4]
        np.testing.assert_array_equal(b[4 * k : 4 * k + 4], np.roll(corners, -1, axis=0))
        mid = corners.mean(axis=0)
        i = int(np.argmin(np.linalg.norm(t.center - mid, axis=1)))
        s_values.append(i * step)
        # Square is aligned to the local tangent.
        edge = (corners[1] - corners[0]) / 1.5
        assert abs(float(edge @ t.normal[i])) < 1e-9
        # Fully inside the corridor, >= 4 m free on at least one side.
        lateral = (corners - t.center[i]) @ t.normal[i]
        assert (np.abs(lateral) < t.width[i] / 2).all()
        free_left = _seg_dist(corners, left_a, left_b).min()
        free_right = _seg_dist(corners, right_a, right_b).min()
        assert free_left > 0
        assert free_right > 0
        assert max(free_left, free_right) >= 4.0
        assert not _crossings(
            corners, np.roll(corners, -1, axis=0), t.seg_a[~obstacle], t.seg_b[~obstacle]
        ).any()
    s = np.array(s_values)
    assert (s >= 30).all()
    assert (s <= t.length - 30).all()
    gaps = np.abs(s[:, None] - s[None])
    assert gaps[~np.eye(4, dtype=bool)].min() >= 15


def test_world_04_precheck(track: Track) -> None:
    assert 250 <= track.length <= 1200
    w_max = float(track.width.max())
    assert _radii(track.center).min() >= max(12.0, w_max / 2 + 2)
    left_a, left_b, right_a, right_b = _walls(track)
    for a, b in [(left_a, left_b), (right_a, right_b)]:
        hits = _crossings(a, b, a, b)
        idx = np.arange(len(a))
        gap = np.abs(idx[:, None] - idx[None])
        adjacent = (gap <= 1) | (gap == len(a) - 1)
        assert not (hits & ~adjacent).any()
    assert not _crossings(left_a, left_b, right_a, right_b).any()
    c, w = track.center, track.width
    s = np.arange(len(c)) * (track.length / len(c))
    ds = np.abs(s[:, None] - s[None])
    arc = np.minimum(ds, track.length - ds)
    dist = np.linalg.norm(c[:, None] - c[None], axis=2)
    far = arc > 3 * w_max
    assert (dist[far] >= (w[:, None] / 2 + w[None] / 2 + 6)[far]).all()
