"""ViZDoom-style low-level game API (API-01, 02, 05, 07, 08, 09)."""

import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from raycer import physics
from raycer.physics import DT, Action, CarParams, CarState
from raycer.progress import ProgressState, ProgressTracker
from raycer.render import LABELS, Renderer, RenderOutput
from raycer.track import Track, load_builtin

GAME_VARIABLES: tuple[str, ...] = (
    "speed",
    "lap_progress",
    "lap",
    "lateral_offset",
    "heading_error",
    "wrong_way",
    "collision",
    "episode_time",
)

DISCRETE_ACTIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
)

# scenario -> (builtin track, target laps, reward weight overrides)
_SCENARIOS: dict[str, tuple[str, int, dict[str, float]]] = {
    "oval-v0": ("oval", 1, {}),
    "kidney-v0": ("kidney", 1, {}),
    "technical-v0": ("technical", 1, {}),
    "time-trial-v0": ("oval", 3, {"w_time": 0.05}),
}

_DEFAULT_WEIGHTS = {
    "w_progress": 0.1,
    "w_time": 0.01,
    "w_collision": 1.0,
    "w_wrongway": 0.05,
    "w_lap": 10.0,
}

_COMPONENTS = ("progress", "time", "collision", "wrongway", "lap")


@dataclass(frozen=True)
class GameState:
    """Snapshot of one step: copied buffers and game variables (API-02)."""

    number: int
    screen_buffer: NDArray[np.uint8] | None
    depth_buffer: NDArray[np.float32] | None
    labels_buffer: NDArray[np.uint8] | None
    game_variables: NDArray[np.float32]


def _copy[T: np.generic](a: NDArray[T] | None) -> NDArray[T] | None:
    """Return a copy of a or None."""
    return None if a is None else a.copy()


class Game:
    """Track, physics, progress and renderer behind a ViZDoom-like interface."""

    def __init__(self) -> None:
        """Set the default configuration."""
        self._initialized = False
        self._track_name, self._laps, _ = _SCENARIOS["oval-v0"]
        self._width, self._height = 96, 72
        self._buffers = {"rgb": True, "depth": False, "labels": False}
        self._timeout = 0
        self._frame_skip = 4
        self._seed = 0
        self._pending_weights = dict(_DEFAULT_WEIGHTS)
        self._weights = dict(_DEFAULT_WEIGHTS)
        self._params = CarParams()
        self._steps = 0
        self._total_reward = 0.0
        self._components = dict.fromkeys(_COMPONENTS, 0.0)

    def _check_not_initialized(self, method: str) -> None:
        """Raise RuntimeError if init() has already run."""
        if self._initialized:
            raise RuntimeError(f"{method} must be called before init()")

    def load_config(self, path: str | os.PathLike[str]) -> None:
        """Apply the supported keys of a TOML config through the setters."""
        self._check_not_initialized("load_config")
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        screen = cfg.get("screen", {})
        if "width" in screen or "height" in screen:
            self.set_screen_resolution(
                screen.get("width", self._width), screen.get("height", self._height)
            )
        buffers = cfg.get("buffers", {})
        if "screen" in buffers:
            self.set_screen_buffer_enabled(buffers["screen"])
        if "depth" in buffers:
            self.set_depth_buffer_enabled(buffers["depth"])
        if "labels" in buffers:
            self.set_labels_buffer_enabled(buffers["labels"])
        if "frame_skip" in cfg.get("physics", {}):
            self.set_frame_skip(cfg["physics"]["frame_skip"])
        if "max_episode_steps" in cfg:
            self.set_episode_timeout(cfg["max_episode_steps"])
        if "laps" in cfg:
            self._laps = int(cfg["laps"])
        if "reward" in cfg:
            self.set_reward_config(**{f"w_{k}": v for k, v in cfg["reward"].items()})

    def set_scenario(self, name: str) -> None:
        """Select a builtin-track scenario such as "oval-v0"."""
        self._check_not_initialized("set_scenario")
        if name not in _SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}, expected one of {sorted(_SCENARIOS)}")
        self._track_name, self._laps, weights = _SCENARIOS[name]
        self._pending_weights.update(weights)

    def set_screen_resolution(self, width: int, height: int) -> None:
        """Set the buffer resolution; it is validated by the renderer at init()."""
        self._check_not_initialized("set_screen_resolution")
        self._width, self._height = width, height

    def set_screen_buffer_enabled(self, enabled: bool) -> None:
        """Enable or disable the RGB screen buffer."""
        self._check_not_initialized("set_screen_buffer_enabled")
        self._buffers["rgb"] = enabled

    def set_depth_buffer_enabled(self, enabled: bool) -> None:
        """Enable or disable the metric depth buffer."""
        self._check_not_initialized("set_depth_buffer_enabled")
        self._buffers["depth"] = enabled

    def set_labels_buffer_enabled(self, enabled: bool) -> None:
        """Enable or disable the labels buffer."""
        self._check_not_initialized("set_labels_buffer_enabled")
        self._buffers["labels"] = enabled

    def set_episode_timeout(self, steps: int) -> None:
        """Set the episode timeout in env steps; 0 means unlimited."""
        self._check_not_initialized("set_episode_timeout")
        if steps < 0:
            raise ValueError(f"episode timeout must be >= 0, got {steps}")
        self._timeout = steps

    def set_frame_skip(self, n: int) -> None:
        """Set the number of physics ticks per env step (1..16)."""
        self._check_not_initialized("set_frame_skip")
        if not 1 <= n <= 16:
            raise ValueError(f"frame_skip must be in 1..16, got {n}")
        self._frame_skip = n

    def set_reward_config(self, **weights: float) -> None:
        """Override reward weights (API-09 keys); they apply from the next episode."""
        unknown = set(weights) - set(_DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown reward keys {sorted(unknown)}")
        self._pending_weights.update({k: float(v) for k, v in weights.items()})

    def set_seed(self, seed: int) -> None:
        """Set the seed used from the next episode."""
        self._seed = seed

    def init(self) -> None:
        """Freeze the configuration, build track, renderer and tracker, start an episode."""
        if self._initialized:
            raise RuntimeError("init() was already called")
        self._renderer = Renderer(
            self._width,
            self._height,
            rgb=self._buffers["rgb"],
            depth=self._buffers["depth"],
            labels=self._buffers["labels"],
        )
        self._track: Track = load_builtin(self._track_name)
        self._tracker = ProgressTracker(self._track, dt=DT)
        self._initialized = True
        self.new_episode()

    def new_episode(self, seed: int | None = None) -> None:
        """Reset car, progress, counters and reward and render the start state."""
        if not self._initialized:
            raise RuntimeError("new_episode must be called after init()")
        if seed is not None:
            self._seed = seed
        self._car: CarState = physics.start_state(self._track)
        self._progress: ProgressState = self._tracker.reset(
            self._car.x, self._car.y, self._car.heading
        )
        self._steps = 0
        self._total_reward = 0.0
        self._components = dict.fromkeys(_COMPONENTS, 0.0)
        self._prev_collided = False
        self._step_collided = False
        self._weights = dict(self._pending_weights)
        self._output: RenderOutput = self._renderer.render(self._track, self._car)

    def make_action(self, action: int | Sequence[float]) -> float:
        """Apply an action for frame_skip ticks and return the summed reward."""
        if not self._initialized:
            raise RuntimeError("make_action must be called after init()")
        if self.is_episode_finished():
            raise RuntimeError("make_action called after the episode finished")
        if isinstance(action, int | np.integer):
            if not 0 <= action < len(DISCRETE_ACTIONS):
                raise ValueError(f"discrete action must be in 0..8, got {action}")
            action = DISCRETE_ACTIONS[int(action)]
        if len(action) != 2:
            raise ValueError(f"continuous action must be [steer, throttle], got {action!r}")
        steer, throttle = (min(max(float(a), -1.0), 1.0) for a in action)
        phys = Action(steer, max(throttle, 0.0), max(-throttle, 0.0))
        w = self._weights
        comps = dict.fromkeys(_COMPONENTS, 0.0)
        self._step_collided = False
        for _ in range(self._frame_skip):
            self._car, collided = physics.step(self._car, phys, self._params, self._track)
            old = self._progress
            new = self._tracker.update(
                self._car.x, self._car.y, self._car.heading, self._car.v, collided
            )
            comps["progress"] += w["w_progress"] * (new.total_progress - old.total_progress)
            comps["time"] -= w["w_time"]
            comps["collision"] -= w["w_collision"] * (collided and not self._prev_collided)
            comps["wrongway"] -= w["w_wrongway"] * new.wrong_way
            comps["lap"] += w["w_lap"] * (new.lap > old.lap)
            self._progress = new
            self._prev_collided = collided
            self._step_collided |= collided
        self._output = self._renderer.render(self._track, self._car)
        self._steps += 1
        self._components = comps
        reward = sum(comps.values())
        self._total_reward += reward
        return reward

    def get_state(self) -> GameState | None:
        """Return a copy of the current state, or None if no episode is running."""
        if not self._initialized or self.is_episode_finished():
            return None
        p = self._progress
        variables = np.array(
            [
                self._car.v,
                p.s / self._track.length,
                p.lap,
                p.lateral_offset,
                p.heading_error,
                p.wrong_way,
                self._step_collided,
                self._steps * self._frame_skip * DT,
            ],
            dtype=np.float32,
        )
        out = self._output
        return GameState(
            self._steps, _copy(out.rgb), _copy(out.depth), _copy(out.labels), variables
        )

    def is_episode_finished(self) -> bool:
        """Return whether the target laps or the timeout have been reached."""
        if not self._initialized:
            return True
        return self._progress.lap >= self._laps or self.is_episode_timeout_reached()

    def is_episode_timeout_reached(self) -> bool:
        """Return whether a positive timeout has been reached."""
        return self._timeout > 0 and self._steps >= self._timeout

    def get_episode_time(self) -> int:
        """Return the number of env steps since new_episode."""
        return self._steps

    def get_total_reward(self) -> float:
        """Return the summed reward of the current episode."""
        return self._total_reward

    def get_last_reward_components(self) -> dict[str, float]:
        """Return the weighted reward terms of the last step."""
        return dict(self._components)

    def get_label_names(self) -> dict[int, str]:
        """Return the label id to name table."""
        return dict(LABELS)

    def get_game_variable_names(self) -> tuple[str, ...]:
        """Return the names of the game variables in order."""
        return GAME_VARIABLES

    def close(self) -> None:
        """Release the episode; the game behaves as before init()."""
        self._initialized = False
