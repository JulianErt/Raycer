import hashlib
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

from raycer import DISCRETE_ACTIONS, RaycerEnv
from raycer.cli import main
from raycer.human import Player, Viewer, keys_to_action, replay
from raycer.recording import RecordEpisode, load_episode
from raycer.render import RenderOutput

W, H = 32, 24
HINT = 'uv add "raycer[human]"'


def _env(**kwargs: Any) -> RaycerEnv:
    return RaycerEnv(**{"render_mode": "human", "width": W, "height": H, **kwargs})


def _press(key: int, down: bool = True) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=key))


@pytest.mark.parametrize("bits", range(16))
def test_keys_to_action_API18(bits: int) -> None:
    up, down, left, right = (bool(bits >> i & 1) for i in range(4))
    expected = (float(left) - float(right), float(up) - float(down))
    assert DISCRETE_ACTIONS[keys_to_action(up, down, left, right)] == expected


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
    window = pygame.surfarray.array3d(viewer.surface).swapaxes(0, 1)
    assert np.array_equal(window, np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1))
    viewer.view = "all"
    viewer.show(RenderOutput(rgb, depth, None), "")
    window = pygame.surfarray.array3d(viewer.surface).swapaxes(0, 1)
    assert np.array_equal(window[:H, :W], rgb)
    assert (window[:H, W:] == 128).all()
    assert (window[H:] == 0).all()
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
    argv = ["play", "--record", str(tmp_path), "--width", "32", "--height", "24", "--depth"]
    assert main(argv) == 0
    (path,) = tmp_path.glob("*.npz")
    ep = load_episode(path)
    assert list(ep["actions"]) == [0, 1, 1]
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
