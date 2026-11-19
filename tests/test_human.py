import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

import raycer.human
from raycer import DISCRETE_ACTIONS, RaycerEnv
from raycer.cli import main
from raycer.human import (
    COUNTDOWN_FRAMES,
    Player,
    Viewer,
    hud_lines,
    keys_to_action,
    ramp_controls,
    replay,
    start_scale,
)
from raycer.recording import RecordEpisode, load_episode
from raycer.render import RenderOutput

W, H = 32, 24
HINT = 'uv add "raycer[human]"'


def _env(**kwargs: Any) -> RaycerEnv:
    return RaycerEnv(**{"render_mode": "human", "width": W, "height": H, **kwargs})


def _press(key: int, down: bool = True) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=key))


@pytest.fixture
def no_countdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(raycer.human, "COUNTDOWN_FRAMES", 0)


def _window(viewer: Viewer) -> np.ndarray[Any, Any]:
    return pygame.surfarray.array3d(viewer.surface).swapaxes(0, 1)


@pytest.mark.parametrize("bits", range(16))
def test_keys_to_action_API18(bits: int) -> None:
    up, down, left, right = (bool(bits >> i & 1) for i in range(4))
    expected = (float(left) - float(right), float(up) - float(down))
    assert DISCRETE_ACTIONS[keys_to_action(up, down, left, right)] == expected


@pytest.mark.usefixtures("no_countdown")
def test_player_key_events_API18(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    player = Player(env, seed=0)
    sent: list[Any] = []
    step = env.step
    monkeypatch.setattr(env, "step", lambda a: sent.append(a) or step(a))
    groups = {
        "up": (pygame.K_UP, pygame.K_w),
        "down": (pygame.K_DOWN, pygame.K_s),
        "left": (pygame.K_LEFT, pygame.K_a),
        "right": (pygame.K_RIGHT, pygame.K_d),
    }
    keys = [k for g in groups.values() for k in g]
    rng = np.random.default_rng(0)
    held: set[int] = set()
    for _ in range(100):
        key = int(rng.choice(keys))
        _press(key, key not in held)
        held ^= {key}
        assert player.frame()
        expected = keys_to_action(*(not held.isdisjoint(g) for g in groups.values()))
        assert sent[-1] == expected
    assert len(sent) == 100
    _press(pygame.K_ESCAPE)
    assert not player.frame()
    env.close()


def test_player_continuous_API18() -> None:
    env = _env(action_type="continuous")
    player = Player(env, seed=0)
    _press(pygame.K_UP)
    assert player.frame()
    env.close()


@pytest.mark.usefixtures("no_countdown")
def test_player_reset_and_pause_API18() -> None:
    env = _env()
    player = Player(env, seed=0)
    for _ in range(3):
        player.frame()
    assert player.info["step"] == 3
    _press(pygame.K_p)
    player.frame()
    player.frame()
    assert player.paused
    assert player.info["step"] == 3
    _press(pygame.K_r)
    player.frame()
    assert player.info["step"] == 0
    _press(pygame.K_p)
    player.frame()
    assert player.info["step"] == 1
    env.close()


def test_player_tab_and_f12_API18(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = _env(depth=True, labels=True)
    player = Player(env, seed=0)
    player.frame()
    views = []
    for _ in range(4):
        _press(pygame.K_TAB)
        player.frame()
        views.append(player.viewer.view)
    assert views == ["depth", "labels", "all", "rgb"]
    _press(pygame.K_F12)
    player.frame()
    (shot,) = tmp_path.glob("raycer_*.png")
    assert shot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    env.close()


def test_cycle_view_skips_missing_REND11() -> None:
    viewer = Viewer(W, H, 60.0)
    rgb = np.zeros((H, W, 3), np.uint8)
    viewer.show(RenderOutput(rgb, None, None), "")
    viewer.cycle_view()
    assert viewer.view == "all"
    viewer.show(RenderOutput(rgb, None, None), "hud")
    viewer.cycle_view()
    assert viewer.view == "rgb"
    viewer.close()


def test_viewer_draws_render_output_REND11() -> None:
    rgb = np.random.default_rng(0).integers(0, 256, (H, W, 3), np.uint8)
    depth = np.full((H, W), 30.0, np.float32)
    viewer = Viewer(W, H, 60.0)
    viewer.show(RenderOutput(rgb, None, None), "")
    assert np.array_equal(_window(viewer), np.repeat(np.repeat(rgb, 25, axis=0), 25, axis=1))
    viewer.view = "all"
    viewer.show(RenderOutput(rgb, depth, None), "")
    # 2W x 2H grid scaled by 12 into 800x600, centred at (16, 12)
    window = _window(viewer)[12:-12, 16:-16]
    assert np.array_equal(window[: 12 * H, : 12 * W], np.repeat(np.repeat(rgb, 12, 0), 12, 1))
    assert (window[: 12 * H, 12 * W :] == 128).all()
    assert (window[12 * H :] == 0).all()
    viewer.close()


def test_viewer_30_fps_API15() -> None:
    viewer = Viewer(W, H, 60.0)
    output = RenderOutput(np.zeros((H, W, 3), np.uint8), None, None)
    t0 = time.perf_counter()
    for _ in range(15):
        viewer.show(output, "")
    assert time.perf_counter() - t0 >= 0.4
    viewer.close()


def test_env_human_API15() -> None:
    env, ref = _env(), RaycerEnv(width=W, height=H)
    assert env.viewer is not None
    assert ref.viewer is None
    pairs = [(env.reset(seed=1)[0], ref.reset(seed=1)[0])]
    for a in (1, 1, 5, 6, 0):
        assert env.render() is None
        pairs.append((env.step(a)[0], ref.step(a)[0]))
    for obs, expected in pairs:
        assert obs.keys() == expected.keys()
        for key in obs:
            assert np.array_equal(obs[key], expected[key])
    env.close()


def test_replay_hashes_API18(tmp_path: Path) -> None:
    env = RecordEpisode(RaycerEnv(width=W, height=H, max_episode_steps=20), tmp_path)
    env.action_space.seed(0)
    env.reset(seed=7)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        done = term or trunc
    assert env.last_path is not None
    recorded = load_episode(env.last_path)["screen"]
    screens = replay(env.last_path)

    def sha(a: np.ndarray[Any, Any]) -> str:
        return hashlib.sha256(a.tobytes()).hexdigest()

    assert len(screens) == len(recorded) == 21
    assert [sha(s) for s in screens] == [sha(s) for s in recorded]


def test_replay_procedural_API18(tmp_path: Path) -> None:
    env = RecordEpisode(
        RaycerEnv("proc-train-v0", width=W, height=H, depth=True, max_episode_steps=5), tmp_path
    )
    env.reset(seed=3)
    for a in (1, 1, 5):
        env.step(a)
    env.close()
    assert env.last_path is not None
    assert np.array_equal(np.stack(replay(env.last_path)), load_episode(env.last_path)["screen"])


def test_replay_esc_API18(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = RecordEpisode(RaycerEnv(width=W, height=H), tmp_path)
    env.reset(seed=0)
    for _ in range(5):
        env.step(1)
    env.close()
    show = Viewer.show

    def show_then_esc(self: Viewer, output: RenderOutput, hud: str) -> None:
        show(self, output, hud)
        _press(pygame.K_ESCAPE)

    monkeypatch.setattr(Viewer, "show", show_then_esc)
    assert env.last_path is not None
    assert len(replay(env.last_path)) == 1


def test_cli_play_record_API18(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    show = Viewer.show
    calls = []

    def show_then_esc(self: Viewer, output: RenderOutput, hud: str) -> None:
        show(self, output, hud)
        calls.append(hud)
        _press(pygame.K_UP if len(calls) < 3 else pygame.K_ESCAPE)

    monkeypatch.setattr(Viewer, "show", show_then_esc)
    monkeypatch.setattr(raycer.human, "COUNTDOWN_FRAMES", 0)
    argv = ["play", "--record", str(tmp_path), "--width", "32", "--height", "24", "--depth"]
    assert main([*argv, "--discrete"]) == 0
    (path,) = tmp_path.glob("*.npz")
    ep = load_episode(path)
    assert list(ep["actions"]) == [0, 1, 1]
    assert ep["actions"].dtype == np.int64
    assert "depth" in ep
    assert "speed" in calls[0]

    monkeypatch.setattr(Viewer, "show", show)
    assert main(["play", "--replay", str(path)]) == 0


def test_missing_pygame_hint_API15(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "pygame", None)
    monkeypatch.delitem(sys.modules, "raycer.human")
    with pytest.raises(ImportError, match=r'uv add "raycer\[human\]"'):
        RaycerEnv(render_mode="human")
    assert main(["play"]) == 1
    assert HINT in capsys.readouterr().err


def test_start_scale_and_window_HUM01() -> None:
    assert start_scale(1024, 768, 320, 240) == 2
    assert start_scale(1024, 768, 32, 24) == 25
    assert start_scale(1024, 768, 640, 480) == 2
    viewer = Viewer(W, H, 60.0)
    assert viewer.surface.get_size() == (800, 600)
    viewer.close()


def test_resize_integer_centred_scaling_HUM01() -> None:
    env = _env()
    player = Player(env, seed=0)
    viewer = player.viewer
    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, size=(100, 100), w=100, h=100))
    player.frame()
    assert viewer.surface.get_size() == (100, 100)
    rgb = np.random.default_rng(0).integers(1, 256, (H, W, 3), np.uint8)
    viewer.hud, viewer.minimap, viewer.overlay = None, False, []
    viewer.show(RenderOutput(rgb, None, None), "")
    window = _window(viewer)
    # factor 3: 96x72 image at offset (2, 14)
    assert np.array_equal(window[14:86, 2:98], np.repeat(np.repeat(rgb, 3, 0), 3, 1))
    window[14:86, 2:98] = 0
    assert (window == 0).all()
    env.close()


def test_f11_fullscreen_HUM01() -> None:
    env = _env()
    player = Player(env, seed=0)
    _press(pygame.K_F11)
    player.frame()
    assert player.viewer.fullscreen
    _press(pygame.K_F11)
    player.frame()
    assert not player.viewer.fullscreen
    env.close()


def test_hud_lines_HUM02() -> None:
    gv = np.zeros(8, np.float32)
    gv[0] = 10.0
    lines = hud_lines(gv, {"lap_times": []}, 3)
    assert lines[0] == "36 km/h"
    assert "lap 1/3" in lines
    assert "last --" in lines
    assert "best --" in lines
    assert "WRONG WAY" not in lines
    gv[5] = 1
    assert "WRONG WAY" in hud_lines(gv, {"lap_times": []}, 3)
    gv[[2, 5, 7]] = 2, 0, 75.0
    lines = hud_lines(gv, {"lap_times": [40.0, 30.0]}, 3)
    assert "lap 3/3" in lines
    assert "time 0:05.00" in lines
    assert "last 0:30.00" in lines
    assert "best 0:30.00" in lines
    gv[2] = 3
    assert "lap 3/3" in hud_lines(gv, {"lap_times": [40.0, 30.0, 35.0]}, 3)


def test_player_draws_hud_HUM02() -> None:
    env = _env()
    player = Player(env, seed=0)
    player.frame()
    assert player.viewer.hud is not None
    assert player.viewer.hud[0] == "0 km/h"
    env.close()


def test_minimap_toggle_HUM03() -> None:
    env = _env()
    player = Player(env, seed=0)
    assert player.viewer.minimap
    player.frame()
    on = _window(player.viewer)[:200, -200:].copy()
    _press(pygame.K_m)
    player.frame()
    assert not player.viewer.minimap
    off = _window(player.viewer)[:200, -200:].copy()
    assert not np.array_equal(on, off)
    _press(pygame.K_m)
    player.frame()
    assert player.viewer.minimap
    env.close()


def test_countdown_freezes_steps_HUM04() -> None:
    assert COUNTDOWN_FRAMES == 90
    env = _env()
    player = Player(env, seed=0)
    for _ in range(2):
        for i in range(COUNTDOWN_FRAMES):
            player.frame()
            assert player.info["step"] == 0
            if i == 0:
                assert player.viewer.overlay == ["3"]
        player.frame()
        assert player.info["step"] == 1
        assert player.viewer.overlay == []
        _press(pygame.K_r)
    env.close()


@pytest.mark.usefixtures("no_countdown")
@pytest.mark.parametrize("key", [pygame.K_p, pygame.K_h, pygame.K_F1])
def test_pause_and_help_freeze_HUM04(key: int) -> None:
    env = _env()
    player = Player(env, seed=0)
    player.frame()
    _press(key)
    player.frame()
    player.frame()
    assert player.info["step"] == 1
    if key == pygame.K_p:
        assert player.paused
        assert player.viewer.overlay == ["PAUSE"]
    else:
        assert player.help
        text = " ".join(player.viewer.overlay)
        for name in ("W/A/S/D", "arrows", "R:", "P:", "M:", "H/F1", "Tab", "F11", "F12", "Esc"):
            assert name in text
    _press(key)
    player.frame()
    assert player.info["step"] == 2
    assert player.viewer.overlay == []
    env.close()


@pytest.mark.usefixtures("no_countdown")
def test_finish_screen_HUM04(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    player = Player(env, seed=0)
    step = env.step
    calls = []

    def finish(a: Any) -> Any:
        obs, r, _, trunc, info = step(a)
        calls.append(a)
        return obs, r, True, trunc, {**info, "lap_times": [31.5]}

    monkeypatch.setattr(env, "step", finish)
    player.frame()
    assert player.finished
    for _ in range(3):
        player.frame()
    assert len(calls) == 1
    assert player.viewer.overlay == ["FINISH", "lap 1: 0:31.50", "R: restart"]
    monkeypatch.setattr(env, "step", step)
    _press(pygame.K_r)
    player.frame()
    assert not player.finished
    assert player.info["step"] == 1
    env.close()


@pytest.mark.usefixtures("no_countdown")
def test_truncated_resets_HUM04() -> None:
    env = _env(max_episode_steps=3)
    player = Player(env, seed=0)
    for _ in range(3):
        player.frame()
    assert not player.finished
    assert player.info["step"] == 0
    env.close()


def test_ramp_timing_HUM05() -> None:
    steer = throttle = 0.0
    for frame in range(1, 9):
        steer, throttle = ramp_controls(steer, throttle, False, False, True, False, 1 / 30)
        if frame < 8:
            assert steer < 1.0
    assert steer == 1.0
    assert math.ceil(0.25 * 30) == 8
    for _ in range(5):
        steer, _ = ramp_controls(steer, 0.0, False, False, False, False, 1 / 30)
    assert steer == 0.0
    for up, down, target in ((True, False, 1.0), (False, True, -1.0)):
        throttle = 0.0
        for _ in range(6):
            _, throttle = ramp_controls(0.0, throttle, up, down, False, False, 1 / 30)
        assert throttle == target
    # towards the opposite side the steer rises at the 0.25 s rate and never overshoots
    steer, _ = ramp_controls(1.0, 0.0, False, False, False, True, 1 / 30)
    assert steer == pytest.approx(1 - 4 / 30)
    assert ramp_controls(-0.95, 0.0, False, False, False, True, 1 / 30) == (-1.0, 0.0)


@pytest.mark.usefixtures("no_countdown")
def test_player_continuous_ramp_HUM05(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env(action_type="continuous")
    player = Player(env, seed=0)
    sent: list[Any] = []
    step = env.step
    monkeypatch.setattr(env, "step", lambda a: sent.append(a) or step(a))
    _press(pygame.K_LEFT)
    _press(pygame.K_w)
    for _ in range(8):
        player.frame()
    assert all(isinstance(a, np.ndarray) and a.dtype == np.float32 for a in sent)
    assert sent[0][0] == pytest.approx(4 / 30)
    assert sent[0][1] == pytest.approx(5 / 30)
    assert sent[6][0] < 1.0
    assert tuple(sent[-1]) == (1.0, 1.0)
    env.close()


@pytest.mark.usefixtures("no_countdown")
def test_overlays_keep_obs_HUM02(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env(action_type="continuous", depth=True)
    ref = RaycerEnv(width=W, height=H, action_type="continuous", depth=True)
    pairs = []
    step = env.step

    def record(a: Any) -> Any:
        out = step(a)
        pairs.append((out[0], ref.step(a)[0]))
        return out

    monkeypatch.setattr(env, "step", record)
    player = Player(env, seed=4)
    ref.reset(seed=4)
    _press(pygame.K_UP)
    for i in range(12):
        if i == 3:
            _press(pygame.K_m)
        if i == 5:
            _press(pygame.K_TAB)
        if i == 7:
            _press(pygame.K_p)
        if i == 9:
            _press(pygame.K_p)
        player.frame()
    assert len(pairs) == 10
    for obs, expected in pairs:
        assert obs.keys() == expected.keys()
        for key in obs:
            assert np.array_equal(obs[key], expected[key])
    env.close()


def test_player_record_replay_hashes_HUM06(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(raycer.human, "COUNTDOWN_FRAMES", 2)
    env = RecordEpisode(_env(action_type="continuous"), tmp_path)
    player = Player(env, seed=2)
    _press(pygame.K_UP)
    _press(pygame.K_a)
    for i in range(14):
        if i == 6:
            _press(pygame.K_a, down=False)
        player.frame()
    env.close()
    assert env.last_path is not None
    recorded = load_episode(env.last_path)
    assert len(recorded["actions"]) == 12
    assert recorded["actions"].dtype == np.float32

    def sha(a: np.ndarray[Any, Any]) -> str:
        return hashlib.sha256(a.tobytes()).hexdigest()

    screens = replay(env.last_path)
    assert [sha(s) for s in screens] == [sha(s) for s in recorded["screen"]]


def test_get_track_HUM03() -> None:
    env = _env()
    assert env.game.get_track() is env.game._track
    env.close()
