from collections.abc import Callable
from typing import Any

import gymnasium
import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv

import raycer  # noqa: F401  # registers the Raycer/* ids in the test process
from raycer.wrappers import StackChannels

N = 4
STEPS = 50


def make_proc_train() -> gymnasium.Env[Any, Any]:
    return StackChannels(
        gymnasium.make("Raycer/ProcTrain-v0", depth=True, labels=True),
        keys=("screen", "depth", "labels"),
        one_hot_labels=True,
    )


def make_proc_train_dr() -> gymnasium.Env[Any, Any]:
    return gymnasium.make("Raycer/ProcTrainDR-v0")


FACTORIES = pytest.mark.parametrize(
    "factory", [make_proc_train, make_proc_train_dr], ids=["ProcTrain", "ProcTrainDR"]
)


def _check_obs(obs: Any, factory: Callable[[], gymnasium.Env[Any, Any]]) -> None:
    if factory is make_proc_train:
        assert obs.shape == (N, 72, 96, 10)
        assert obs.dtype == np.uint8
    else:
        assert obs["screen"].shape == (N, 72, 96, 3)
        assert obs["screen"].dtype == np.uint8
        assert obs["gamevariables"].shape[0] == N


def _check_flags(rewards: Any, *flags: Any) -> None:
    assert rewards.shape == (N,)
    assert np.issubdtype(rewards.dtype, np.floating)
    for flag in flags:
        assert flag.shape == (N,)
        assert flag.dtype == np.bool_


@FACTORIES
def test_async_vector_spawn_QA41(factory: Callable[[], gymnasium.Env[Any, Any]]) -> None:
    envs = AsyncVectorEnv([factory] * N, context="spawn")
    obs, _ = envs.reset(seed=0)
    envs.action_space.seed(0)
    _check_obs(obs, factory)
    assert obs in envs.observation_space
    for _ in range(STEPS):
        obs, rewards, terminated, truncated, _ = envs.step(envs.action_space.sample())
        _check_obs(obs, factory)
        assert obs in envs.observation_space
        _check_flags(rewards, terminated, truncated)
    envs.close()
    assert envs.closed
    assert not any(p.is_alive() for p in envs.processes)


@FACTORIES
def test_subproc_vec_env_spawn_QA41(factory: Callable[[], gymnasium.Env[Any, Any]]) -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    vec_env = stable_baselines3.common.vec_env
    vec = vec_env.SubprocVecEnv([factory] * N, start_method="spawn")
    vec.seed(0)
    obs = vec.reset()
    vec.action_space.seed(0)
    _check_obs(obs, factory)
    for _ in range(STEPS):
        actions = np.array([vec.action_space.sample() for _ in range(N)])
        obs, rewards, dones, _ = vec.step(actions)
        _check_obs(obs, factory)
        _check_flags(rewards, dones)
    vec.close()
    assert vec.closed
    assert not any(p.is_alive() for p in vec.processes)
