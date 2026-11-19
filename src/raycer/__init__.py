"""Raycer: a raycasting racing environment for machine learning research."""

from importlib.metadata import version

from raycer.config import list_scenarios
from raycer.game import DISCRETE_ACTIONS, GAME_VARIABLES, Game, GameState

__version__ = version("raycer")

__all__ = [
    "DISCRETE_ACTIONS",
    "GAME_VARIABLES",
    "Game",
    "GameState",
    "__version__",
    "list_scenarios",
]
