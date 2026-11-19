import dataclasses
import inspect
import math
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from raycer import palette, render
from raycer.physics import CarState, start_state
from raycer.render import Renderer, RenderOutput
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
    assert np.all(np.abs(rgb[half - 1] - palette.SKY_HORIZON) <= 1)
    t = np.arange(half) / (half - 1)
    sky = (
        np.array(palette.SKY_TOP) + (np.subtract(palette.SKY_HORIZON, palette.SKY_TOP)) * t[:, None]
    )
    assert np.all(np.abs(rgb[:half] - sky[:, None]) <= 1)
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
    shade = 0.6 + 0.4 / np.hypot(1.0, k)
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


def test_public_docstrings() -> None:
    for obj in (render.Renderer, render.RenderOutput, Renderer.render, Renderer.wall_depth):
        doc = inspect.getdoc(obj)
        assert doc
        assert "\n" not in doc
