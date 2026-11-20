"""Gymnasium environment over one Game (API-03, 04, 06, 10, 14, 15, 16)."""

from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from raycer.config import load_config
from raycer.game import Game, GameState
from raycer.palette import LABEL_COLORS
from raycer.track import PRESETS, SPLITS, generate_track, load_builtin

if TYPE_CHECKING:
    from raycer.human import Viewer

Obs = dict[str, Any]


class RaycerEnv(gym.Env[Obs, Any]):
    """Racing environment with Dict observations, discrete or continuous actions."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}  # noqa: RUF012

    def __init__(
        self,
        scenario: str = "oval-v0",
        *,
        config_path: str | None = None,
        render_mode: Literal["rgb_array", "human"] | None = None,
        action_type: Literal["discrete", "continuous"] = "discrete",
        screen: bool = True,
        depth: bool = False,
        labels: bool = False,
        width: int = 96,
        height: int = 72,
        frame_skip: int = 4,
        max_episode_steps: int | None = None,
        reward: dict[str, float] | None = None,
        metric_depth_info: bool = False,
        num_tracks: int | None = None,
    ) -> None:
        """Merge the config, configure and init the Game and build the spaces."""
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError(f"unknown render_mode {render_mode!r}")
        if action_type not in ("discrete", "continuous"):
            raise ValueError(f"action_type must be 'discrete' or 'continuous', got {action_type!r}")
        # Kwargs at their API-03 default count as not given, so a user TOML can set them.
        size = (("width", width, 96), ("height", height, 72))
        toggles = (("screen", screen, True), ("depth", depth, False), ("labels", labels, False))
        overrides: dict[str, Any] = {
            "screen": {k: v for k, v, d in size if v != d},
            "buffers": {k: v for k, v, d in toggles if v != d},
            "physics": {"frame_skip": frame_skip} if frame_skip != 4 else {},
        }
        if max_episode_steps is not None:
            overrides["max_episode_steps"] = max_episode_steps
        if num_tracks is not None:
            overrides["track"] = {"num_tracks": num_tracks}
        cfg = load_config(scenario, config_path, overrides)
        buffers = cfg["buffers"]
        if metric_depth_info and not buffers["depth"]:
            raise ValueError("metric_depth_info=True needs the depth buffer")

        self.render_mode = render_mode
        self._scenario = scenario
        self._track_cfg: dict[str, Any] = cfg["track"]
        self._laps: int = cfg["laps"]
        self._max_steps: int = cfg["max_episode_steps"]
        self._metric_depth_info = metric_depth_info
        w, h = cfg["screen"]["width"], cfg["screen"]["height"]
        self._height: int = h

        self.game = Game()
        self.game.set_screen_resolution(w, h)
        self.game.set_screen_buffer_enabled(buffers["screen"])
        self.game.set_depth_buffer_enabled(buffers["depth"])
        self.game.set_labels_buffer_enabled(buffers["labels"])
        self.game.set_frame_skip(cfg["physics"]["frame_skip"])
        self.game.set_reward_config(**{f"w_{k}": v for k, v in cfg["reward"].items()})
        if reward is not None:
            self.game.set_reward_config(**reward)
        self.game.set_appearance(**cfg["appearance"])
        self.game.init()
        if self._track_cfg["type"] == "builtin":
            self.game._set_track(load_builtin(self._track_cfg["builtin"]), self._laps)
        self.viewer: Viewer | None = None
        if render_mode == "human":
            import raycer.human

            self.viewer = raycer.human.Viewer(w, h, self.game._renderer.max_view_distance)

        obs_spaces: dict[str, spaces.Space[Any]] = {}
        if buffers["screen"]:
            obs_spaces["screen"] = spaces.Box(0, 255, (h, w, 3), np.uint8)
        if buffers["depth"]:
            obs_spaces["depth"] = spaces.Box(0, 255, (h, w, 1), np.uint8)
        if buffers["labels"]:
            obs_spaces["labels"] = spaces.Box(0, 5, (h, w, 1), np.uint8)
        obs_spaces["gamevariables"] = spaces.Box(-np.inf, np.inf, (8,), np.float32)
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space: spaces.Space[Any] = (
            spaces.Discrete(9)
            if action_type == "discrete"
            else spaces.Box(np.array([-1, -1], np.float32), np.array([1, 1], np.float32))
        )

        self._track_seed: int | None = None
        self._steps = 0
        self._wrong_way_steps = 0
        self._lap_times: list[float] = []
        self._done = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        """Start an episode; procedural scenarios pick the track seed from the split."""
        super().reset(seed=seed)
        track_seed = (options or {}).get("track_seed")
        if self._track_cfg["type"] == "procedural":
            split = SPLITS[self._track_cfg["split"]]
            num_tracks = self._track_cfg.get("num_tracks")
            if num_tracks is not None:
                first = range(split.start, split.start + num_tracks)
                if track_seed is None:
                    track_seed = int(self.np_random.integers(first.start, first.stop))
                elif int(track_seed) not in first:
                    raise ValueError(
                        f"track_seed {track_seed} not in the first num_tracks={num_tracks} seeds"
                    )
            elif track_seed is None:
                track_seed = int(self.np_random.integers(split.start, split.stop))
            elif int(track_seed) not in split:
                split_name = self._track_cfg["split"]
                raise ValueError(f"track_seed {track_seed} is not in split {split_name!r}")
            track_seed = int(track_seed)
            track = generate_track(track_seed, PRESETS[self._track_cfg["preset"]])
            self.game._set_track(track, self._laps)
        elif track_seed is not None:
            raise ValueError(f"scenario {self._scenario!r} has a builtin track, no track_seed")
        self._track_seed = track_seed
        # spawn() leaves the parent stream untouched, so track seeds do not depend on DR.
        self.game.new_episode(seed=int(self.np_random.spawn(1)[0].integers(2**63)))
        self._steps = 0
        self._wrong_way_steps = 0
        self._lap_times = []
        self._done = False
        state = self.game._snapshot()
        return self._obs(state), self._info(state)

    def step(self, action: Any) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        """Apply one action and return obs, reward, terminated, truncated and info."""
        if self._done:
            raise RuntimeError("step() called after the episode ended; call reset()")
        reward = self.game.make_action(action)
        self._steps += 1
        p = self.game._progress
        state = self.game._snapshot()
        self._wrong_way_steps += p.wrong_way
        if p.lap > len(self._lap_times):
            self._lap_times.append(float(state.game_variables[7]) - sum(self._lap_times))
        terminated = p.lap >= self._laps
        truncated = not terminated and (
            self._steps == self._max_steps or p.stuck or p.checkpoint_timeout
        )
        self._done = terminated or truncated
        return self._obs(state), float(reward), terminated, truncated, self._info(state)

    def render(self) -> NDArray[np.uint8] | None:
        """Return the active buffers side by side at double size, or None without rgb_array."""
        if self.viewer is not None:
            gv = self.game._snapshot().game_variables
            hud = (
                f"speed {gv[0]:.1f} m/s  lap {int(gv[2])}  progress {gv[1]:.2f}"
                f"  reward {self.game.get_total_reward():.2f}"
            )
            self.viewer.show(self.game._output, hud)
        if self.render_mode != "rgb_array":
            return None
        obs = self._obs(self.game._snapshot())
        parts = []
        if "screen" in obs:
            parts.append(obs["screen"])
        if "depth" in obs:
            parts.append(np.repeat(obs["depth"], 3, axis=2))
        if "labels" in obs:
            parts.append(LABEL_COLORS[obs["labels"][..., 0]])
        image = np.concatenate(parts, axis=1) if parts else np.zeros((self._height, 0, 3), np.uint8)
        return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

    def close(self) -> None:
        """Close the viewer window if there is one."""
        if self.viewer is not None:
            self.viewer.close()

    def _obs(self, state: GameState) -> Obs:
        """Build the Dict observation from a state snapshot."""
        obs: Obs = {}
        if state.screen_buffer is not None:
            obs["screen"] = state.screen_buffer
        if state.depth_buffer is not None:
            mvd = self.game._renderer.max_view_distance
            obs["depth"] = np.round(255 * state.depth_buffer / mvd).astype(np.uint8)[..., None]
        if state.labels_buffer is not None:
            obs["labels"] = state.labels_buffer[..., None]
        obs["gamevariables"] = state.game_variables
        return obs

    def _info(self, state: GameState) -> dict[str, Any]:
        """Build the API-06 info dict."""
        p = self.game._progress
        info: dict[str, Any] = {
            "step": self._steps,
            "lap": p.lap,
            "lap_progress": float(state.game_variables[1]),
            "track_seed": self._track_seed,
            "scenario": self._scenario,
            "reward_components": self.game.get_last_reward_components(),
            "collisions": p.collision_count,
            "wrong_way_steps": self._wrong_way_steps,
            "lap_times": list(self._lap_times),
            "is_success": p.lap >= self._laps,
        }
        if self._metric_depth_info:
            assert state.depth_buffer is not None
            info["depth_m"] = state.depth_buffer.astype(np.float32, copy=True)
        if self.game._appearance is not None:
            info["appearance"] = dict(self.game._appearance)
        return info
