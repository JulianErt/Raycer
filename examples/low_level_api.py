"""Random agent on oval-v0 through the low-level raycer.Game API."""

import argparse
import random

from raycer import DISCRETE_ACTIONS, Game

SEED = 0


def main() -> None:
    """Run one episode with seeded random discrete actions and print the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None, help="maximum game steps")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    game = Game()
    game.set_scenario("oval-v0")
    game.set_screen_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.init()
    game.new_episode(SEED)
    rng = random.Random(SEED)
    steps = 0
    while not game.is_episode_finished() and (args.steps is None or steps < args.steps):
        game.make_action(rng.randrange(len(DISCRETE_ACTIONS)))
        steps += 1
    print(f"steps={steps} total_reward={game.get_total_reward():.3f}")
    game.close()


if __name__ == "__main__":
    main()
