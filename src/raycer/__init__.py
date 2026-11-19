"""Raycer: a raycasting racing environment for machine learning research."""

from importlib.metadata import version

import gymnasium

from raycer.config import gym_name, list_scenarios
from raycer.env import RaycerEnv
from raycer.game import DISCRETE_ACTIONS, GAME_VARIABLES, Game, GameState
from raycer.palette import LABEL_COLORS as LABEL_PALETTE

__version__ = version("raycer")

for _scenario in list_scenarios():
    _name = gym_name(_scenario)
    for _suffix, _kwargs in (("", {}), ("-Continuous", {"action_type": "continuous"})):
        gymnasium.register(
            f"Raycer/{_name}{_suffix}-v0",
            entry_point="raycer.env:RaycerEnv",
            max_episode_steps=None,
            kwargs={"scenario": _scenario, **_kwargs},
        )

__all__ = [
    "DISCRETE_ACTIONS",
    "GAME_VARIABLES",
    "LABEL_PALETTE",
    "Game",
    "GameState",
    "RaycerEnv",
    "__version__",
    "list_scenarios",
]
