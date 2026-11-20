"""Episode recording to compressed .npz files (API-19)."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import raycer

logger = logging.getLogger(__name__)

Obs = dict[str, Any]
_KEYS = ("screen", "depth", "labels", "gamevariables")


class RecordEpisode(gym.Wrapper[Obs, Any, Obs, Any]):
    """Record each episode's observations, actions, rewards and flags to a .npz file."""

    def __init__(self, env: gym.Env[Obs, Any], directory: str | Path) -> None:
        """Wrap a Dict observation env and create the output directory."""
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("RecordEpisode needs a Dict observation env")
        super().__init__(env)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_path: Path | None = None
        self._seed: int | None = None
        self._obs: list[Obs] = []
        self._poses: list[tuple[float, float, float]] = []
        self._actions: list[Any] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._info: dict[str, Any] = {}
        self._written = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        """Write an unfinished episode, then reset and start a new buffer."""
        self._write_pending()
        obs, info = self.env.reset(seed=seed, options=options)
        self._seed = seed
        self._obs = [obs]
        self._poses = [info["pose"]]
        self._actions, self._rewards, self._terminated, self._truncated = [], [], [], []
        self._info = info
        self._written = False
        return obs, info

    def step(self, action: Any) -> tuple[Obs, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the env, append the transition and write the file when the episode ends."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs.append(obs)
        self._poses.append(info["pose"])
        self._actions.append(action)
        self._rewards.append(float(reward))
        self._terminated.append(terminated)
        self._truncated.append(truncated)
        self._info = info
        if terminated or truncated:
            self._write()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Write an unfinished episode and close the wrapped env."""
        self._write_pending()
        super().close()

    def _write_pending(self) -> None:
        """Write the current episode if it has steps and was not written yet."""
        if not self._written and self._actions:
            self._write()

    def _write(self) -> None:
        """Save the buffered episode to a new, never-overwritten file."""
        game = self.env.unwrapped.game  # type: ignore[attr-defined]
        scenario, track_seed = self._info["scenario"], self._info["track_seed"]
        discrete = isinstance(self.env.action_space, spaces.Discrete)
        meta = {
            "raycer_version": raycer.__version__,
            "scenario": scenario,
            "seed": self._seed,
            "track_seed": track_seed,
            "action_type": "discrete" if discrete else "continuous",
            "frame_skip": game._frame_skip,
            "width": game._width,
            "height": game._height,
            "label_names": [name for _, name in sorted(game.get_label_names().items())],
        }
        arrays: dict[str, Any] = {
            k: np.stack([o[k] for o in self._obs]) for k in _KEYS if k in self._obs[0]
        }
        arrays["pose"] = np.array(self._poses, np.float64)
        arrays["actions"] = np.array(self._actions, np.int64 if discrete else np.float32)
        arrays["rewards"] = np.array(self._rewards, np.float32)
        arrays["terminated"] = np.array(self._terminated, bool)
        arrays["truncated"] = np.array(self._truncated, bool)
        arrays["meta"] = np.array(json.dumps(meta))
        stem = f"{scenario}_s{'none' if track_seed is None else track_seed}"
        stem += datetime.now().strftime("_%Y%m%d-%H%M%S")
        path, n = self.directory / f"{stem}.npz", 0
        while path.exists():
            n += 1
            path = self.directory / f"{stem}_{n}.npz"
        np.savez_compressed(path, **arrays)
        self.last_path = path
        self._written = True
        logger.info("recorded %d steps to %s", len(self._actions), path)


def load_episode(path: str | Path) -> dict[str, Any]:
    """Load a recorded episode; meta is returned as the parsed JSON dict."""
    with np.load(path, allow_pickle=False) as data:
        episode: dict[str, Any] = {k: data[k] for k in data.files}
    episode["meta"] = json.loads(str(episode["meta"]))
    return episode
