import dataclasses
import inspect
import itertools
import math
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from raycer import palette, render
from raycer.physics import CarState, start_state
from raycer.render import LABELS, Renderer, RenderOutput
from raycer.track import Material, Track, load_builtin

Array = NDArray[np.float64]
ORIGIN = CarState(0.0, 0.0, 0.0, 0.0)
D = 10.0


def _track(
    segs: list[tuple[float, float, float, float]],
    material: int = Material.LEFT_BARRIER,
    finish: tuple[float, float] = (1000.0, 1000.0),
) -> Track:
    """Hand-built track: segments (ax, ay, bx, by) and a 6 m wide finish line at `finish`."""
    s = np.array(segs, dtype=np.float64).reshape(-1, 4)
    center = np.array([finish, (finish[0] + 1.0, finish[1])])
    return Track(
        s[:, :2],
        s[:, 2:],
        np.full(len(s), material, dtype=np.uint8),
        np.zeros(len(s)),
        center,
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([6.0, 6.0]),
    )


FAR = _track([(100.0, -50.0, 100.0, 50.0)])
FRONTAL = [(D, -50.0, D, 50.0)]


def _rgb(
    track: Track, renderer: Renderer | None = None, state: CarState = ORIGIN
) -> NDArray[np.uint8]:
    out = (renderer or Renderer()).render(track, state).rgb
    assert out is not None
    return out


def _wall_rows(rgb: NDArray[np.uint8], ref: NDArray[np.uint8]) -> NDArray[np.intp]:
    """Number of rows per column whose colour differs from the reference render."""
    return np.any(rgb != ref, axis=2).sum(axis=0)


def _fog(colour: Array, depth: Array | float) -> Array:
    f = np.minimum(1.0, np.asarray(depth) / 60.0) ** 1.5
    return colour + (np.array(palette.FOG) - colour) * f[..., None]


def _ks(r: Renderer) -> Array:
    return (np.arange(r.width) + 0.5 - r.width / 2) / r.focal


@pytest.mark.parametrize("name", ["oval", "kidney", "technical"])
def test_render_shape_dtype(name: str) -> None:
    track = load_builtin(name)
    out = Renderer().render(track, start_state(track))
    assert isinstance(out, RenderOutput)
    assert out.rgb is not None
    assert out.rgb.dtype == np.uint8
    assert out.rgb.shape == (72, 96, 3)
    assert out.rgb.flags.c_contiguous
    assert _rgb(track, Renderer(160, 120), start_state(track)).shape == (120, 160, 3)


@pytest.mark.parametrize(
    "kwargs",
    [{"width": w} for w in (30, 33, 642)]
    + [{"height": h} for h in (22, 25, 482)]
    + [{"fov_deg": v} for v in (59, 121)],
)
def test_invalid_camera(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        Renderer(**kwargs)  # type: ignore[arg-type]


def test_camera_bounds_and_focal() -> None:
    for w, h, fov in ((32, 24, 60.0), (640, 480, 120.0)):
        r = Renderer(w, h, fov)
        assert (r.width, r.height, r.fov_deg, r.max_view_distance) == (w, h, fov, 60.0)
    assert Renderer().focal == pytest.approx(48.0)
    assert Renderer(fov_deg=60).focal == pytest.approx(48.0 / math.tan(math.radians(30)))


def test_frontal_wall() -> None:
    r = Renderer()
    depth = r.wall_depth(_track(FRONTAL), ORIGIN)
    assert depth[47:49] == pytest.approx(D, rel=1e-5)
    # Fisheye correction: every column sees the same perpendicular distance and height.
    assert depth == pytest.approx(np.full(96, D), rel=1e-5)
    rows = _wall_rows(_rgb(_track(FRONTAL)), _rgb(FAR))
    assert abs(rows[48] - round(2 * r.focal / D)) <= 1
    assert rows.max() - rows.min() <= 1


def test_diagonal_wall() -> None:
    r = Renderer()
    seg = [(D + 60.0, -60.0, D - 60.0, 60.0)]
    depth = r.wall_depth(_track(seg), ORIGIN)
    expected = D / (1 - _ks(r))
    hit = expected < 60.0
    assert hit.sum() > 60
    assert not hit.all()
    assert depth[hit] == pytest.approx(expected[hit], rel=1e-5)
    assert np.all(depth[~hit] == 60.0)
    # Two materials with distinct colours differ exactly on wall pixels; fog hides them near 60 m.
    rows = _wall_rows(_rgb(_track(seg)), _rgb(_track(seg, Material.OBSTACLE)))
    near = expected < 40.0
    assert np.all(np.abs(rows[near] - np.round(2 * r.focal / depth[near])) <= 1)


def test_parallel_rays_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        depth = Renderer().wall_depth(_track([(0.0, 0.0, 30.0, 0.0), (5.0, 0.0, 5.0, 0.0)]), ORIGIN)
    assert np.all(depth == 60.0)


def test_open_view_sky_and_floor() -> None:
    r = Renderer()
    assert np.all(r.wall_depth(FAR, ORIGIN) == 60.0)
    rgb = _rgb(FAR).astype(np.int16)
    half = r.height // 2
    assert np.all(np.abs(rgb[0] - palette.SKY_TOP) <= 1)
    t = np.arange(half) / (half - 1)
    sky = (
        np.array(palette.SKY_TOP) + (np.subtract(palette.SKY_HORIZON, palette.SKY_TOP)) * t[:, None]
    )
    # Above the highest possible skyline every sky row is the gradient.
    top = half - math.floor(0.25 * half)
    assert np.all(np.abs(rgb[:top] - sky[:top, None]) <= 1)
    # Every floor pixel is the fogged ROAD_A/ROAD_B checker at its world point (REND-03/08/09).
    depth = (r.focal / (np.arange(half, r.height) + 0.5 - half))[:, None]
    px, py = depth * np.ones(r.width), -depth * _ks(r)
    odd = (np.floor(px / 4) + np.floor(py / 4)) % 2 == 1
    expected = _fog(np.where(odd[..., None], palette.ROAD_B, palette.ROAD_A).astype(float), depth)
    assert np.all(np.abs(rgb[half:] - expected) <= 1)


def test_finish_line() -> None:
    r = Renderer()
    # Pixel (H/2+2, W/2) is the floor point at depth 48/2.5 = 19.2 m; the finish quad spans x in
    # [18.7, 19.7] and y in [-3, 3].
    rgb = _rgb(_track([(100.0, -50.0, 100.0, 50.0)], finish=(19.2, 0.0))).astype(np.int16)
    j, i = r.height // 2 + 2, r.width // 2
    depth = r.focal / 2.5
    x, y = depth, -depth * _ks(r)[i]
    odd = (math.floor(x / 0.5) + math.floor(y / 0.5)) % 2 == 1
    colour = np.array(palette.FINISH_B if odd else palette.FINISH_A, dtype=float)
    assert np.all(np.abs(rgb[j, i] - _fog(colour, depth)) <= 1)
    # One row up the floor point lies at 32 m, outside the quad: a road colour.
    road = [_fog(np.array(c, dtype=float), r.focal / 1.5) for c in (palette.ROAD_A, palette.ROAD_B)]
    assert min(np.abs(rgb[j - 1, i] - c).max() for c in road) <= 1


@pytest.mark.parametrize(
    ("material", "a", "b", "stripe"),
    [
        (Material.LEFT_BARRIER, palette.LEFT_A, palette.LEFT_B, 2.0),
        (Material.RIGHT_BARRIER, palette.RIGHT_A, palette.RIGHT_B, 2.0),
        (Material.OBSTACLE, palette.OBSTACLE_A, palette.OBSTACLE_B, 0.5),
    ],
)
def test_wall_stripes_and_shading(
    material: int, a: tuple[int, int, int], b: tuple[int, int, int], stripe: float
) -> None:
    r = Renderer()
    rgb = _rgb(_track(FRONTAL, material)).astype(np.int16)
    k = _ks(r)
    # Column i hits (D, -k·D); u = s·100 with s = (y + 50) / 100.
    u = 50.0 - k * D
    frac = (u / stripe) % 1
    clean = (frac > 1e-6) & (frac < 1 - 1e-6)
    odd = np.floor(u / stripe) % 2 == 1
    assert odd[clean].any()
    assert (~odd[clean]).any()
    # The wall normal (-1, 0) faces away from the sun, so only ambient and view terms remain.
    shade = 0.45 + 0.25 / np.hypot(1.0, k)
    expected = _fog(np.where(odd[:, None], b, a) * shade[:, None], D)
    assert np.all(np.abs(rgb[r.height // 2, clean] - expected[clean]) <= 1)


def test_culling_is_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    track = load_builtin("technical")
    s0 = start_state(track)
    poses = [s0] + [
        dataclasses.replace(s0, x=float(track.center[n, 0]), y=float(track.center[n, 1]), heading=h)
        for n, h in ((80, 1.0), (200, -2.0), (330, 3.0))
    ]
    culled = [_rgb(track, state=p) for p in poses]
    monkeypatch.setattr(track, "segments_in_radius", lambda *a, **k: np.arange(len(track.seg_a)))
    for p, rgb in zip(poses, culled, strict=True):
        assert np.array_equal(_rgb(track, state=p), rgb)


def _poses(n_per_track: int, seed: int = 7) -> list[tuple[Track, CarState]]:
    """Fixed-seed random poses inside the corridor of every builtin track."""
    rng = np.random.default_rng(seed)
    poses = []
    for name in ("oval", "kidney", "technical"):
        track = load_builtin(name)
        for _ in range(n_per_track):
            n = int(rng.integers(len(track.center)))
            lim = track.width[n] / 2 - 0.5
            x, y = track.center[n] + rng.uniform(-lim, lim) * track.normal[n]
            poses.append(
                (track, CarState(float(x), float(y), float(rng.uniform(-np.pi, np.pi)), 0.0))
            )
    return poses


def _check_invariants(r: Renderer, track: Track, state: CarState) -> None:
    """Assert the REND-06 invariants (QA-33) for one pose with all buffers enabled."""
    out = r.render(track, state)
    assert out.rgb is not None
    assert out.depth is not None
    assert out.labels is not None
    rgb, depth, labels = out.rgb, out.depth, out.labels
    h, half, far = r.height, r.height // 2, np.float32(r.max_view_distance)
    floor = np.float32(r.focal / (np.arange(half, h) + 0.5 - half))[:, None]
    # 1. Sky label exactly where the depth is the view distance.
    assert np.array_equal(labels == 0, depth == far)
    # 2. Road and finish only below the horizon, at the floor depth of their row.
    road = np.isin(labels, (1, 4))
    assert not road[:half].any()
    assert np.all(np.where(road[half:], depth[half:] == floor, True))
    # 3. One horizon-symmetric wall block per column with a single depth and label.
    wall = np.isin(labels, (2, 3, 5))
    assert np.array_equal(wall, wall[::-1])
    edges = np.diff(np.pad(wall.astype(np.int8), ((1, 1), (0, 0))), axis=0)
    assert np.all(np.abs(edges).sum(axis=0) <= 2)
    for buf in (depth, labels.astype(np.float32)):
        hi = np.where(wall, buf, -np.inf).max(axis=0)
        lo = np.where(wall, buf, np.inf).min(axis=0)
        assert np.all(hi[wall.any(axis=0)] == lo[wall.any(axis=0)])
    assert np.all(np.where(wall[half:], depth[half:] <= floor, True))
    # 4. Depth is positive and finite.
    assert np.all(np.isfinite(depth))
    assert np.all(depth > 0)
    # 6. Sky-labelled pixels show a colour of the sky gradient.
    sky = r.sky_colors
    assert np.all((rgb[labels == 0][:, None, :] == sky[None]).all(axis=2).any(axis=1))


def _check_toggles(r: Renderer, track: Track, state: CarState) -> None:
    """Assert that every toggle combination matches the all-on render (invariant 5)."""
    r.rgb_enabled = r.depth_enabled = r.labels_enabled = True
    full = r.render(track, state)
    for flags in itertools.product((False, True), repeat=3):
        r.rgb_enabled, r.depth_enabled, r.labels_enabled = flags
        out = r.render(track, state)
        for on, buf, ref in zip(
            flags,
            (out.rgb, out.depth, out.labels),
            (full.rgb, full.depth, full.labels),
            strict=True,
        ):
            if on:
                assert buf is not None
                assert ref is not None
                assert buf.dtype == ref.dtype
                assert np.array_equal(buf, ref)
            else:
                assert buf is None
    r.rgb_enabled = r.depth_enabled = r.labels_enabled = True


@pytest.mark.parametrize(("w", "h"), [(96, 72), (160, 120)])
def test_buffer_invariants(w: int, h: int) -> None:
    r = Renderer(w, h, rgb=True, depth=True, labels=True)
    for track, state in _poses(167):
        _check_invariants(r, track, state)
        _check_toggles(r, track, state)


@pytest.mark.parametrize("kwargs", [{"max_view_distance": 9.9}, {"max_view_distance": 200.1}])
def test_invalid_view_distance(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        Renderer(**kwargs)


def test_view_distance_bounds() -> None:
    assert Renderer(max_view_distance=10).max_view_distance == 10
    assert Renderer(max_view_distance=200).max_view_distance == 200
    assert Renderer().max_view_distance == 60.0
    r = Renderer(max_view_distance=10, depth=True, labels=True)
    out = r.render(FAR, ORIGIN)
    assert out.depth is not None
    assert out.labels is not None
    assert np.all(out.depth <= 10.0)
    assert np.all(out.depth[: r.height // 2] == 10.0)
    assert np.all(out.labels[: r.height // 2] == 0)


@pytest.mark.parametrize(("w", "h"), [(96, 72), (160, 120)])
def test_depth_labels_shape_dtype(w: int, h: int) -> None:
    track = load_builtin("oval")
    out = Renderer(w, h, depth=True, labels=True).render(track, start_state(track))
    assert out.depth is not None
    assert out.labels is not None
    assert (out.depth.dtype, out.depth.shape) == (np.float32, (h, w))
    assert (out.labels.dtype, out.labels.shape) == (np.uint8, (h, w))
    assert out.depth.flags.c_contiguous
    assert out.labels.flags.c_contiguous


def test_default_buffers() -> None:
    r = Renderer()
    assert (r.rgb_enabled, r.depth_enabled, r.labels_enabled) == (True, False, False)
    out = r.render(FAR, ORIGIN)
    assert out.rgb is not None
    assert out.depth is None
    assert out.labels is None


def test_labels_table() -> None:
    assert LABELS == {
        0: "SKY",
        1: "ROAD",
        2: "LEFT_BARRIER",
        3: "RIGHT_BARRIER",
        4: "FINISH",
        5: "OBSTACLE",
    }


@pytest.mark.parametrize(
    "material", [Material.LEFT_BARRIER, Material.RIGHT_BARRIER, Material.OBSTACLE]
)
def test_frontal_wall_depth_label(material: Material) -> None:
    r = Renderer(depth=True, labels=True)
    out = r.render(_track(FRONTAL, material), ORIGIN)
    assert out.depth is not None
    assert out.labels is not None
    cols = [r.width // 2 - 1, r.width // 2]
    assert out.depth[r.height // 2, cols] == pytest.approx(D, abs=1e-4)
    assert np.all(out.labels[r.height // 2, cols] == int(material))


def test_finish_line_label() -> None:
    r = Renderer(labels=True)
    out = r.render(_track([(100.0, -50.0, 100.0, 50.0)], finish=(19.2, 0.0)), ORIGIN)
    assert out.labels is not None
    j, i = r.height // 2 + 2, r.width // 2
    assert out.labels[j, i] == 4
    assert out.labels[j - 1, i] == 1


def test_public_docstrings() -> None:
    for obj in (render.Renderer, render.RenderOutput, Renderer.render, Renderer.wall_depth):
        doc = inspect.getdoc(obj)
        assert doc
        assert "\n" not in doc


def _skyline_rows(rgb: NDArray[np.uint8], r: Renderer) -> NDArray[np.bool_]:
    """Sky-half pixels that differ from their gradient row."""
    half = r.height // 2
    return np.any(rgb[:half] != r.sky_colors[:half, None], axis=2)


def test_sky_colors() -> None:
    r = Renderer()
    assert (r.sky_colors.dtype, r.sky_colors.shape) == (np.uint8, (38, 3))
    assert tuple(r.sky_colors[-2]) == palette.SKYLINE_FAR
    assert tuple(r.sky_colors[-1]) == palette.SKYLINE_NEAR


def test_skyline_rotation() -> None:
    r = Renderer()
    base = _rgb(FAR)
    turned = _rgb(FAR, state=CarState(0.0, 0.0, 0.3, 0.0))
    assert not np.array_equal(base[: r.height // 2], turned[: r.height // 2])
    counts = _skyline_rows(base, r).sum(axis=0)
    angle = np.arctan(_ks(r))
    for i, j in ((0, 95), (10, 50), (20, 70), (40, 41), (60, 5), (95, 30)):
        state = CarState(0.0, 0.0, float(angle[j] - angle[i]), 0.0)
        assert abs(_skyline_rows(_rgb(FAR, state=state), r).sum(axis=0)[j] - counts[i]) <= 1


def test_skyline_bounds() -> None:
    r = Renderer()
    half = r.height // 2
    seen: set[tuple[int, ...]] = set()
    for heading in np.linspace(-np.pi, np.pi, 36, endpoint=False):
        rgb = _rgb(FAR, state=CarState(0.0, 0.0, float(heading), 0.0))
        sky = rgb[:half]
        layer = _skyline_rows(rgb, r)
        rows = layer.sum(axis=0)
        # One contiguous run per column that ends at the horizon row.
        assert np.array_equal(layer, np.arange(half)[:, None] >= half - rows)
        assert rows.max() <= math.floor(0.25 * half) + 1
        assert np.all((sky[..., None, :] == r.sky_colors).all(axis=-1).any(axis=-1))
        seen |= {tuple(int(v) for v in c) for c in sky[layer]}
    assert seen == {palette.SKYLINE_FAR, palette.SKYLINE_NEAR}


@pytest.mark.parametrize(("x", "heading"), [(D, 0.0), (-D, math.pi)])
def test_sun_shading(x: float, heading: float) -> None:
    r = Renderer()
    state = CarState(0.0, 0.0, heading, 0.0)
    rgb = _rgb(_track([(x, -50.0, x, 50.0)]), state=state).astype(np.int16)
    k = _ks(r)
    hy = -k * D if heading == 0.0 else k * D
    u = hy + 50.0
    frac = (u / 2.0) % 1
    clean = (frac > 1e-6) & (frac < 1 - 1e-6)
    odd = np.floor(u / 2.0) % 2 == 1
    normal = np.array([-1.0, 0.0]) if x > 0 else np.array([1.0, 0.0])
    sun = max(0.0, float(normal @ render._SUN))
    shade = 0.45 + 0.25 / np.hypot(1.0, k) + 0.30 * sun
    expected = _fog(np.where(odd[:, None], palette.LEFT_B, palette.LEFT_A) * shade[:, None], D)
    assert np.all(np.abs(rgb[r.height // 2, clean] - expected[clean]) <= 1)
    assert (sun > 0) == (x < 0)


def test_sun_facing_wall_is_brighter() -> None:
    away = _rgb(_track([(D, -50.0, D, 50.0)])).astype(int)
    facing = _rgb(_track([(-D, -50.0, -D, 50.0)]), state=CarState(0.0, 0.0, math.pi, 0.0))
    # Mirrored wall: column i of one view sees the same wall point as column W-1-i of the other.
    j = Renderer().height // 2
    assert np.all(facing[j, ::-1].astype(int) >= away[j])
    assert facing[j].astype(int).sum() > away[j].sum()


def test_wall_edge() -> None:
    r = Renderer()
    rgb = _rgb(_track(FRONTAL)).astype(float)
    i = r.width // 2
    top = r.height // 2 - round(r.focal / D)
    assert r._dy[top] < -0.84 * r.focal / D <= r._dy[top + 1]
    f = (D / 60.0) ** 1.5
    fogless = (rgb[:, i] - f * np.array(palette.FOG)) / (1 - f)
    assert np.all(np.abs(fogless[top] - 0.7 * fogless[r.height // 2]) <= 1.5)
    assert np.array_equal(rgb[top + 1, i], rgb[r.height // 2, i])


def test_markings() -> None:
    r = Renderer(160, 120, labels=True)
    hw = 3.0
    track = _track(FRONTAL, finish=(5.0, 0.0))
    track.finish_quad = track.finish_quad + 1000.0  # move the finish line out of view
    out = r.render(track, ORIGIN)
    assert out.rgb is not None
    assert out.labels is not None
    rgb = out.rgb.astype(np.int16)
    i, half = r.width // 2, r.height // 2
    fd = r.focal / (np.arange(half, r.height) + 0.5 - half)
    curb_row = half + int(np.flatnonzero((fd > 9.4) & (fd < 10.0))[0])
    depth = r.focal / (curb_row + 0.5 - half)
    stripes = [_fog(np.array(c, dtype=float), depth) for c in (palette.LEFT_A, palette.LEFT_B)]
    assert min(np.abs(rgb[curb_row, i] - c).max() for c in stripes) <= 1
    assert out.labels[curb_row, i] == 1
    # The centre line sits hw from the wall at x = D - hw; u = y + 50 is on a dash near y = 0.
    line = np.flatnonzero(np.abs(D - fd - hw) <= 0.075)
    assert line.size
    row = half + int(line[0])
    py = -fd[line[0]] * _ks(r)[i]
    assert math.floor((py + 50.0) / 3.0) % 2 == 0
    assert np.all(
        np.abs(rgb[row, i] - _fog(np.array(palette.CENTER_LINE, float), fd[line[0]])) <= 1
    )
    assert out.labels[row, i] == 1


def _reference_cast(
    r: Renderer, track: Track, car_state: CarState
) -> tuple[Array, Array, NDArray[np.intp], Array, Array]:
    """Verbatim copy of Renderer._cast from v0.1 (main before STYLE-03..07)."""
    self = r
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


def _reference_depth_labels(
    r: Renderer, track: Track, car_state: CarState
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Verbatim copy of the v0.1 depth and labels blocks of Renderer.render."""
    self = r
    rays, depth, seg, _, _ = _reference_cast(r, track, car_state)
    half = self.height // 2
    mvd = self.max_view_distance
    mask = (np.abs(self._dy)[:, None] <= self.focal * render.CAMERA_HEIGHT / depth) & (depth < mvd)
    fd = self._floor_depth[:, None]
    mat = track.seg_material[seg]
    px = car_state.x + fd * rays[:, 0]
    py = car_state.y + fd * rays[:, 1]
    q = track.finish_quad
    edge = q[[1, 2, 3, 0]] - q
    e4, q4 = edge[:, :, None, None], q[:, :, None, None]
    side = e4[:, 0] * (py - q4[:, 1]) - e4[:, 1] * (px - q4[:, 0])
    inside = np.all(side >= 0, axis=0) | np.all(side <= 0, axis=0)
    far = np.float32(mvd)
    dbuf = np.full((self.height, self.width), far, dtype=np.float32)
    dbuf[half:] = fd.astype(np.float32)
    dbuf = np.where(mask, depth.astype(np.float32)[None], dbuf)
    dbuf = np.ascontiguousarray(np.minimum(dbuf, far))
    lbuf = np.zeros((self.height, self.width), dtype=np.uint8)
    lbuf[half:] = np.where(inside, 4, 1)
    lbuf = np.where(mask, mat[None], lbuf).astype(np.uint8)
    lbuf[dbuf == far] = 0
    return dbuf, lbuf


def test_ground_truth_matches_v01() -> None:
    r = Renderer(96, 72, rgb=True, depth=True, labels=True)
    poses = _poses(67)
    assert len(poses) == 201
    for track, state in poses:
        out = r.render(track, state)
        depth, labels = _reference_depth_labels(r, track, state)
        assert np.array_equal(out.depth, depth)
        assert np.array_equal(out.labels, labels)


def test_set_appearance_API21() -> None:
    track = load_builtin("technical")
    state = start_state(track)
    r = Renderer()
    base = _rgb(track, r, state)
    params = palette.draw_appearance(1, "test", 1.0)
    r.set_appearance(params)
    out = r.render(track, state)
    assert out.rgb is not None
    assert not np.array_equal(out.rgb, base)
    assert not np.array_equal(r.sky_colors, Renderer().sky_colors)
    r.set_appearance(None)
    assert np.array_equal(_rgb(track, r, state), base)
