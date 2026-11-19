import math
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from raycer import RaycerEnv
from raycer.wrappers import NormalizeGameVariables, StackChannels

W, H = 32, 24
SCALES = np.array([30.0, 1.0, 3.0, 8.0, math.pi, 1.0, 1.0, 600.0], np.float32)


def _env(**kwargs: Any) -> RaycerEnv:
    return RaycerEnv(**{"width": W, "height": H, **kwargs})


def test_stack_screen_depth_labels() -> None:
    raw = _env(depth=True, labels=True)
    env = StackChannels(raw, keys=("screen", "depth", "labels"))
    assert env.observation_space == spaces.Box(0, 255, (H, W, 5), np.uint8)
    obs, _ = env.reset(seed=0)
    for o in (obs, env.step(1)[0]):
        assert o.dtype == np.uint8
        assert env.observation_space.contains(o)
        r = raw._obs(raw.game._snapshot())
        assert np.array_equal(o[..., :3], r["screen"])
        assert np.array_equal(o[..., 3:4], r["depth"])
        assert np.array_equal(o[..., 4:5], np.round(255 * r["labels"].astype(int) / 5))
    assert set(np.unique(o[..., 4])) <= {0, 51, 102, 153, 204, 255}


def test_stack_one_hot_labels() -> None:
    raw = _env(labels=True, screen=False)
    env = StackChannels(raw, keys=("labels",), one_hot_labels=True)
    assert env.observation_space.shape == (H, W, 6)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert set(np.unique(obs)) <= {0, 255}
    assert np.all((obs == 255).sum(axis=2) == 1)
    labels = raw._obs(raw.game._snapshot())["labels"][..., 0]
    assert np.array_equal(obs.argmax(axis=2), labels)


def test_stack_default_and_errors() -> None:
    env = StackChannels(_env(depth=True))
    assert env.observation_space.shape == (H, W, 4)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for keys in [("labels",), ("nope",), (), ("screen", "screen")]:
        with pytest.raises(ValueError, match=r"key|active"):
            StackChannels(_env(), keys=keys)


def test_stack_is_image_space_and_passes_check_env() -> None:
    env = StackChannels(_env(depth=True))
    space = env.observation_space
    assert space.dtype == np.uint8
    assert len(space.shape) == 3
    assert space.low.min() == 0
    assert space.high.max() == 255
    check_env(env, skip_render_check=True)


def test_normalize_gamevariables() -> None:
    raw = _env(depth=True)
    env = NormalizeGameVariables(raw)
    assert env.observation_space == raw.observation_space
    obs, _ = env.reset(seed=0)
    r = raw._obs(raw.game._snapshot())
    assert obs["gamevariables"].dtype == np.float32
    assert np.array_equal(obs["gamevariables"], r["gamevariables"] / SCALES)
    assert np.array_equal(obs["screen"], r["screen"])
    assert np.array_equal(obs["depth"], r["depth"])
    for _ in range(200):
        obs, _, term, trunc, _ = env.step(1)
        assert env.observation_space.contains(obs)
        assert np.all(np.abs(obs["gamevariables"]) <= 1.5)
        if term or trunc:
            break
