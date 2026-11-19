"""Train PPO on ProcTrain and evaluate on ProcTest with RGB, RGB+depth or RGB+labels input.

Needs the train group: uv sync --group train
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import raycer  # noqa: F401  # registers the Raycer/* env IDs
from raycer.wrappers import StackChannels

SEED = 0
OBS_KEYS = {"rgb": ("screen",), "rgbd": ("screen", "depth"), "rgbl": ("screen", "labels")}


def make_env(env_id: str, keys: tuple[str, ...], max_episode_steps: int | None = None) -> gym.Env:
    """Create a Raycer env with the needed buffers, stacked into one image."""
    env = gym.make(
        env_id,
        depth="depth" in keys,
        labels="labels" in keys,
        max_episode_steps=max_episode_steps,
    )
    return StackChannels(env, keys=keys)


def evaluate(model: PPO, keys: tuple[str, ...], max_steps: int | None) -> dict[str, Any]:
    """Run ProcTest seeds 0..49 and return mean return, success rate and mean lap time."""
    env = make_env("Raycer/ProcTest-v0", keys, max_steps)
    returns, successes, lap_times = [], [], []
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
        successes.append(bool(info["is_success"]))
        if info["is_success"]:
            lap_times.extend(info["lap_times"])
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_lap_time": float(np.mean(lap_times)) if lap_times else "",
    }


def main() -> None:
    """Train, evaluate and write the result CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs", choices=tuple(OBS_KEYS), default="rgb")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=None, help="cap training and eval episodes")
    parser.add_argument("--out", default="runs", help="output directory for the CSV")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    keys = OBS_KEYS[args.obs]
    timesteps = args.timesteps if args.steps is None else min(args.timesteps, args.steps)
    n_steps = 2048 if args.steps is None else min(2048, args.steps)

    venv = make_vec_env(
        make_env,
        n_envs=8,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"env_id": "Raycer/ProcTrain-v0", "keys": keys},
    )
    model = PPO("CnnPolicy", venv, n_steps=n_steps, seed=SEED, verbose=0)
    model.learn(total_timesteps=timesteps)
    venv.close()

    result = {"obs": args.obs, "timesteps": timesteps, **evaluate(model, keys, args.steps)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ppo_{args.obs}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(f"wrote {path}: {result}")


if __name__ == "__main__":
    main()
