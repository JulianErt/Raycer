"""QA-34: per-step SHA-256 determinism over 300 steps."""

import hashlib
from typing import Any

import numpy as np

from raycer import Game
from raycer.env import RaycerEnv

STEPS = 300
ACTIONS = [int(a) for a in np.random.default_rng(1234).choice([1, 1, 1, 5, 6, 0, 2], size=STEPS)]


def _game() -> Game:
    """Initialised 64x48 oval game with all buffers enabled."""
    game = Game()
    game.set_scenario("oval-v0")
    game.set_screen_resolution(64, 48)
    game.set_screen_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.init()
    return game


def _digests(game: Game, actions: list[int]) -> list[str]:
    """SHA-256 over all buffers and game variables after each step."""
    out = []
    for a in actions:
        game.make_action(a)
        s = game.get_state()
        assert s is not None
        h = hashlib.sha256()
        for buf in (s.screen_buffer, s.depth_buffer, s.labels_buffer, s.game_variables):
            assert buf is not None
            h.update(buf.tobytes())
        out.append(h.hexdigest())
    return out


def _env_digests(env: RaycerEnv, seed: int, actions: list[int]) -> list[str]:
    """SHA-256 over observation, reward, flags and info after reset(seed) and each step."""
    obs, info = env.reset(seed=seed)
    steps: list[tuple[Any, ...]] = [(obs, 0.0, False, False, info)]
    for a in actions:
        step = env.step(a)
        steps.append(step)
        if step[2] or step[3]:
            break
    out = []
    for obs, reward, terminated, truncated, info in steps:
        h = hashlib.sha256()
        for key in sorted(obs):
            h.update(obs[key].tobytes())
        h.update(repr((reward, terminated, truncated, sorted(info.items()))).encode())
        out.append(h.hexdigest())
    return out


def test_same_instance_QA34() -> None:
    game = _game()
    game.new_episode(5)
    ref = _digests(game, ACTIONS)
    assert len(ref) == STEPS
    game.new_episode(5)
    assert _digests(game, ACTIONS) == ref


def test_two_instances_QA34() -> None:
    a, b = _game(), _game()
    a.new_episode(5)
    b.new_episode(5)
    ref = _digests(a, ACTIONS)
    assert len(ref) == STEPS
    assert _digests(b, ACTIONS) == ref


def test_new_episode_after_preamble_QA34() -> None:
    fresh = _game()
    fresh.new_episode(5)
    ref = _digests(fresh, ACTIONS)
    game = _game()
    game.new_episode(1)
    _digests(game, [4, 3, 1] * 20)
    game.new_episode(2)
    _digests(game, [2, 0] * 37)
    game.new_episode(5)
    assert _digests(game, ACTIONS) == ref


def test_reset_after_preamble_QA34() -> None:
    # frame_skip=1: with the default 4, blind fixed actions get the car stuck before step 300
    kwargs: dict[str, Any] = {"depth": True, "labels": True, "width": 64, "height": 48}
    kwargs["frame_skip"] = 1
    ref = _env_digests(RaycerEnv(scenario="proc-train-v0", **kwargs), 11, ACTIONS)
    assert len(ref) == STEPS + 1
    env = RaycerEnv(scenario="proc-train-v0", **kwargs)
    _env_digests(env, 3, [4, 3, 1] * 20)
    _env_digests(env, 4, [2, 0] * 37)
    assert _env_digests(env, 11, ACTIONS) == ref
