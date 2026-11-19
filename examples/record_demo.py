"""Record three heuristic-driver episodes on oval-v0 to .npz files."""

import argparse

from heuristic_driver import heuristic_action

from raycer import RaycerEnv
from raycer.recording import RecordEpisode


def main() -> None:
    """Record seeds 0, 1 and 2 into --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None, help="maximum env steps per episode")
    parser.add_argument("--out", default="recordings", help="output directory")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    env = RecordEpisode(RaycerEnv("oval-v0", action_type="continuous"), args.out)
    for seed in (0, 1, 2):
        obs, _ = env.reset(seed=seed)  # also writes the previous capped episode
        steps = 0
        while args.steps is None or steps < args.steps:
            obs, _, terminated, truncated, _ = env.step(heuristic_action(obs["gamevariables"]))
            steps += 1
            if terminated or truncated:
                break
        print(f"seed={seed} steps={steps}")
    env.close()  # flushes a capped last episode


if __name__ == "__main__":
    main()
