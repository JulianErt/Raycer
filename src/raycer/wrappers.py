"""Observation wrappers for RaycerEnv (API-17)."""

import math
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

Obs = dict[str, Any]

_CHANNELS = {"screen": 3, "depth": 1, "labels": 1}
# v_max, lap_progress, max laps, half the widest track, pi, wrong_way, collision, 9000*4/60 s
GAMEVARIABLE_SCALES = np.array([30.0, 1.0, 3.0, 8.0, math.pi, 1.0, 1.0, 600.0], np.float32)


class StackChannels(gym.ObservationWrapper[NDArray[np.uint8], Any, Obs]):
    """Concatenate image buffers into one uint8 (H, W, C) Box; drops gamevariables."""

    def __init__(
        self,
        env: gym.Env[Obs, Any],
        keys: Sequence[str] = ("screen", "depth"),
        one_hot_labels: bool = False,
    ) -> None:
        """Validate keys against the active buffers and build the stacked Box."""
        super().__init__(env)
        keys = tuple(keys)
        if not keys:
            raise ValueError("keys must not be empty")
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate key in {keys}")
        space = env.observation_space
        active = space.spaces if isinstance(space, spaces.Dict) else {}
        for key in keys:
            if key not in _CHANNELS:
                raise ValueError(f"unknown key {key!r}, expected one of {tuple(_CHANNELS)}")
            if key not in active:
                raise ValueError(f"buffer {key!r} is not active in the wrapped env")
        self._keys = keys
        self._one_hot = one_hot_labels
        h, w = active[keys[0]].shape[:2]  # type: ignore[index]
        c = sum(6 if k == "labels" and one_hot_labels else _CHANNELS[k] for k in keys)
        self.observation_space = spaces.Box(0, 255, (h, w, c), np.uint8)

    def observation(self, observation: Obs) -> NDArray[np.uint8]:
        """Return the channels of the selected buffers concatenated in key order."""
        parts = []
        for key in self._keys:
            x = observation[key]
            if key == "labels":
                if self._one_hot:
                    x = np.where(x == np.arange(6, dtype=np.uint8), 255, 0).astype(np.uint8)
                else:
                    x = np.round(255 * x.astype(np.float32) / 5).astype(np.uint8)
            parts.append(x)
        return np.concatenate(parts, axis=2)


class NormalizeGameVariables(gym.ObservationWrapper[Obs, Any, Obs]):
    """Scale gamevariables by fixed factors to roughly [-1, 1]; other keys pass through."""

    def observation(self, observation: Obs) -> Obs:
        """Return the observation with gamevariables divided by GAMEVARIABLE_SCALES."""
        gv = (observation["gamevariables"] / GAMEVARIABLE_SCALES).astype(np.float32)
        return {**observation, "gamevariables": gv}
