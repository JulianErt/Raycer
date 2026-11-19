"""Random agent on Raycer/Oval-v0 with all buffers through the Gymnasium API."""

import argparse

import gymnasium as gym

import raycer  # noqa: F401  # registers the Raycer/* env IDs

SEED = 0


def main() -> None:
    """Run one episode with seeded random actions and print steps and total reward."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None, help="maximum env steps")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    env = gym.make("Raycer/Oval-v0", depth=True, labels=True)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    steps, total = 0, 0.0
    while args.steps is None or steps < args.steps:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        total += float(reward)
        if terminated or truncated:
            break
    env.close()
    print(f"steps={steps} total_reward={total:.3f}")


if __name__ == "__main__":
    main()
