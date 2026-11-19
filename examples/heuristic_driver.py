"""Proportional heuristic driver that steers back to the centre line (QA-38)."""

import argparse
from typing import Any

import numpy as np

from raycer import RaycerEnv

K1 = 1.5  # gain on heading_error (rad)
K2 = 0.3  # gain on lateral_offset (m)
THROTTLE = 0.6


def heuristic_action(gamevariables: Any) -> list[float]:
    """Return [steer, throttle]; steer > 0 is left, so steer against the errors."""
    lateral_offset, heading_error = float(gamevariables[3]), float(gamevariables[4])
    steer = float(np.clip(-K1 * heading_error - K2 * lateral_offset, -1.0, 1.0))
    return [steer, THROTTLE]


def drive(scenario: str = "oval-v0", seed: int = 0, max_steps: int = 3000) -> tuple[int, dict]:
    """Drive one episode and return the steps taken and the final info dict."""
    env = RaycerEnv(scenario, action_type="continuous")
    obs, info = env.reset(seed=seed)
    steps = 0
    while steps < max_steps:
        obs, _, terminated, truncated, info = env.step(heuristic_action(obs["gamevariables"]))
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return steps, info


def main() -> None:
    """Parse the arguments, drive one episode and print the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000, help="maximum env steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    steps, info = drive("oval-v0", seed=args.seed, max_steps=args.steps)
    print(f"steps={steps} laps={info['lap']} success={info['is_success']}")


if __name__ == "__main__":
    main()
