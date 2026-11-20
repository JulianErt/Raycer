# API

Raycer has two layers: the low-level `raycer.Game` (ViZDoom style) and the Gymnasium env
`raycer.RaycerEnv`. `import raycer` registers the Gym IDs listed in [scenarios.md](scenarios.md).

## `raycer.Game`

Configure before `init()`; setters called after `init()` raise `RuntimeError`.

| Method | Description |
| --- | --- |
| `load_config(path)` | Apply the `screen`, `buffers`, `physics.frame_skip`, `max_episode_steps`, `laps` `reward` and `depth_noise` keys of a TOML file |
| `set_scenario(name)` | Select a builtin-track scenario: `oval-v0`, `kidney-v0`, `technical-v0` or `time-trial-v0` |
| `set_screen_resolution(width, height)` | Buffer size, default 96x72 |
| `set_screen_buffer_enabled(bool)` | RGB buffer, on by default |
| `set_depth_buffer_enabled(bool)` | Metric depth buffer, off by default |
| `set_labels_buffer_enabled(bool)` | Labels buffer, off by default |
| `set_episode_timeout(steps)` | Timeout in env steps, 0 (default) means unlimited |
| `set_frame_skip(n)` | Physics ticks (1/60 s) per env step, 1 to 16, default 4 |
| `set_reward_config(**weights)` | Override reward weights (`w_progress`, `w_time`, `w_collision`, `w_wrongway`, `w_lap`); applies from the next episode |
| `set_appearance(randomize, split="train", strength=1.0)` | Per-episode appearance randomization; `split` is `"train"` or `"test"`, `strength` 0 to 1 |
| `set_depth_noise(enabled, sigma_rel=0.02, dropout=0.01, quant=0.05)` | Noisy depth sensor: relative Gaussian noise, quantization in metres and per-pixel dropout to the view distance; only depth changes |
| `set_seed(seed)` | Seed used from the next episode |
| `init()` | Freeze the configuration and start the first episode |
| `new_episode(seed=None)` | Reset car, progress and reward |
| `make_action(action)` | Apply a discrete index (0 to 8) or `[steer, throttle]` for `frame_skip` ticks; returns the summed reward |
| `get_state()` | A copied `GameState`, or `None` when no episode is running |
| `is_episode_finished()` | Target laps or timeout reached |
| `is_episode_timeout_reached()` | Positive timeout reached |
| `get_episode_time()` | Env steps since `new_episode` |
| `get_pose()` | Car pose `(x, y, heading)` as floats in metres and radians, world coordinates of `get_track()` |
| `get_total_reward()` | Summed reward of the episode |
| `get_last_reward_components()` | Weighted reward terms of the last step |
| `get_label_names()` | Label ID to name table; works before `init()` |
| `get_game_variable_names()` | `raycer.GAME_VARIABLES`; works before `init()` |
| `close()` | End the episode; the game behaves as before `init()` |

## `GameState`

A frozen dataclass with copies of the buffers:

| Field | Type |
| --- | --- |
| `number` | `int`, env step counter |
| `screen_buffer` | `uint8 (H, W, 3)` or `None` |
| `depth_buffer` | `float32 (H, W)` in metres, or `None` |
| `labels_buffer` | `uint8 (H, W)` label IDs, or `None` |
| `game_variables` | `float32 (8,)`, order of `raycer.GAME_VARIABLES` |

## `RaycerEnv`

```
RaycerEnv(scenario="oval-v0", *, config_path=None, render_mode=None,
          action_type="discrete", screen=True, depth=False, labels=False,
          width=96, height=72, frame_skip=4, max_episode_steps=None,
          reward=None, metric_depth_info=False, num_tracks=None,
          depth_noise=None)
```

- `scenario` is any name from `raycer.list_scenarios()`, including procedural ones.
- `config_path` is a user TOML layered over the scenario TOML; see [scenarios.md](scenarios.md).
- `render_mode` is `None`, `"rgb_array"` (active buffers side by side at double size) or
  `"human"` (pygame window, needs `raycer[human]`).
- `action_type` is `"discrete"` (`Discrete(9)`) or `"continuous"` (`Box([-1, -1], [1, 1])`).
- `reward` overrides weights by their `w_` names.
- `metric_depth_info=True` adds the metric depth as `info["depth_m"]`; it needs `depth=True`.
- `num_tracks=N` limits a procedural scenario to the first N track seeds of its split (like
  Procgen's `num_levels`); it overrides `[track] num_tracks` and is invalid for builtin tracks.
- `depth_noise=True` turns on the noisy depth sensor with the `[depth_noise]` parameters; it
  overrides `[depth_noise] enabled` and only changes `obs["depth"]` and `info["depth_m"]`.
- `reset(seed=..., options={"track_seed": k})` pins the track of a procedural scenario.
- `env.game` is the underlying `Game`.

### Info keys

`reset` and every `step` return an info dict with these keys:

| Key | Content |
| --- | --- |
| `step` | Env step counter |
| `lap` | Completed laps |
| `lap_progress` | Progress in the current lap, [0, 1) |
| `track_seed` | Seed of the procedural track, `None` for builtin tracks |
| `scenario` | Scenario name |
| `reward_components` | Weighted `progress`, `time`, `collision`, `wrongway`, `lap` terms of the last step, all 0.0 after `reset` |
| `collisions` | Cumulative collisions |
| `wrong_way_steps` | Cumulative steps driving the wrong way |
| `lap_times` | List of lap times in seconds |
| `is_success` | True once all target laps are done |
| `pose` | Car pose `(x, y, heading)` as floats in metres and radians, world coordinates of `get_track()` |
| `appearance` | Only with appearance randomization on: dict of the drawn appearance parameters |
| `depth_m` | Only with `metric_depth_info=True`: `float32 (H, W)` depth in metres |

### Episode end

- `terminated` is True exactly when the target laps are reached.
- `truncated` is True when `max_episode_steps` is reached, the car is stuck (less than 1 m of
  progress in 5 s) or no checkpoint was reached for 30 s, and the episode is not terminated.
- The step limit lives in the env; the Gym IDs are registered with `max_episode_steps=None`.
- `step()` after the episode ended raises `RuntimeError`; call `reset()`.

## Wrappers

`raycer.wrappers` has two observation wrappers:

- `StackChannels(env, keys=("screen", "depth"), one_hot_labels=False)` concatenates the given
  buffers into one `Box(0, 255, (H, W, C), uint8)` and drops `gamevariables`. Labels are scaled
  with `round(255 * id / 5)`, or one-hot encoded as 6 channels of 0/255 with
  `one_hot_labels=True`. Use it for Stable-Baselines3 `CnnPolicy`.
- `NormalizeGameVariables(env)` divides `gamevariables` by fixed scales
  (`GAMEVARIABLE_SCALES`) to roughly [-1, 1].

Recording is in [recording.md](recording.md).
