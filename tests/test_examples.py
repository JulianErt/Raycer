import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import raycer  # noqa: F401
from raycer.recording import load_episode
from raycer.wrappers import StackChannels

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FILES = [
    "random_agent",
    "low_level_api",
    "heuristic_driver",
    "record_demo",
    "ppo_rgb_vs_depth",
    "trajectories",
]
SLOW = ("ppo_rgb_vs_depth", "trajectories")


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.timeout(300)) if n in SLOW else n for n in FILES],
)
def test_example_smoke_DOC03(name: str, tmp_path: Path) -> None:
    if name in SLOW:
        pytest.importorskip("stable_baselines3")
    if name == "trajectories":
        pytest.importorskip("matplotlib")
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
    cmd = [sys.executable, str(EXAMPLES / f"{name}.py"), "--steps", "50"]
    if name == "trajectories":
        cmd += ["--train-steps", "64"]
    assert subprocess.run(cmd, cwd=tmp_path, env=env).returncode == 0
    if name == "trajectories":
        assert (tmp_path / "runs" / "trajectories.png").exists()
        assert (tmp_path / "runs" / "trajectories.gif").exists()
    if name == "ppo_rgb_vs_depth":
        assert list((tmp_path / "runs").glob("*.csv"))
    if name == "record_demo":
        files = sorted((tmp_path / "recordings").glob("*.npz"))
        assert len(files) == 3
        for f in files:
            assert load_episode(f)["actions"].shape[0] == 50


def test_heuristic_finishes_lap_QA38() -> None:
    spec = importlib.util.spec_from_file_location(
        "heuristic_driver", EXAMPLES / "heuristic_driver.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    steps, info = module.drive("oval-v0", seed=0, max_steps=3000)
    assert info["is_success"]
    assert info["lap"] >= 1
    assert steps <= 3000


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_ppo_learnability_QA38() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = gym.make("Raycer/Oval-v0", width=64, height=48, max_episode_steps=500)
    monitor = Monitor(StackChannels(env, keys=("screen",)))
    model = PPO("CnnPolicy", monitor, seed=0, verbose=0)
    model.learn(total_timesteps=50_000)
    returns = monitor.get_episode_rewards()
    assert len(returns) >= 20
    first10, last10 = float(np.mean(returns[:10])), float(np.mean(returns[-10:]))
    assert last10 >= first10 + 0.25 * abs(first10)
