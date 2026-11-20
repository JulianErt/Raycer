"""Drive PPO checkpoints greedily on one ProcTest track and plot their trajectories (API-27).

Needs the train group and the plot extra: uv sync --group train --extra plot
"""

import argparse
import re
from pathlib import Path

import numpy as np
from ppo_rgb_vs_depth import OBS_KEYS, make_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from raycer.track import SPLITS
from raycer.trajectory import animate_trajectories, plot_trajectories

SEED = 0


def steps_of(path: Path) -> int:
    """Return the step count in an SB3 checkpoint name `<prefix>_<steps>_steps.zip`."""
    match = re.search(r"_(\d+)_steps$", path.stem)
    return int(match.group(1)) if match else 0


def main() -> None:
    """Optionally train, then drive every checkpoint once and write the PNG and GIF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="runs/checkpoints", help="checkpoint directory")
    parser.add_argument("--obs", choices=tuple(OBS_KEYS), default="rgb")
    parser.add_argument("--track", type=int, default=0, help="index into the test split")
    parser.add_argument("--train-steps", type=int, default=None, help="train PPO first")
    parser.add_argument("--steps", type=int, default=None, help="cap each evaluation episode")
    parser.add_argument("--out", default="runs", help="output directory")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.train_steps is not None and args.train_steps < 1:
        parser.error("--train-steps must be positive")
    keys = OBS_KEYS[args.obs]
    ckpt_dir = Path(args.checkpoints)

    if args.train_steps is not None:
        n = args.train_steps
        venv = make_vec_env(
            make_env,
            n_envs=1,
            seed=SEED,
            vec_env_cls=DummyVecEnv,
            env_kwargs={"env_id": "Raycer/ProcTrain-v0", "keys": keys},
        )
        model = PPO("CnnPolicy", venv, n_steps=min(2048, n), seed=SEED, verbose=0)
        callback = CheckpointCallback(save_freq=max(1, n // 4), save_path=str(ckpt_dir))
        model.learn(total_timesteps=n, callback=callback)
        venv.close()

    checkpoints = sorted(ckpt_dir.glob("*.zip"), key=steps_of)
    if not checkpoints:
        parser.error(f"no *.zip checkpoints in {ckpt_dir}; pass --train-steps N to make some")

    seed = SPLITS["test"].start + args.track
    env = make_env("Raycer/ProcTest-v0", keys, args.steps)
    paths = {}
    for ckpt in checkpoints:
        model = PPO.load(ckpt)
        obs, info = env.reset(seed=SEED, options={"track_seed": seed})
        poses, done = [info["pose"]], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            poses.append(info["pose"])
            done = terminated or truncated
        paths[f"{steps_of(ckpt)} steps"] = np.array(poses)
    track = env.unwrapped.game.get_track()
    env.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png, gif = out / "trajectories.png", out / "trajectories.gif"
    plot_trajectories(track, paths, png)
    animate_trajectories(track, paths, gif)
    print(f"wrote {png}")
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
