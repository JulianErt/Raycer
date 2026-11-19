# Observations

All buffers are pixel-aligned and HWC. The default resolution is 96x72 with a 90 degree field
of view.

## Labels

The labels buffer holds one ID per pixel. `raycer.LABEL_PALETTE` (row = ID) gives the colors
used by `render()` and `raycer render-sample`.

| ID | Name | Palette RGB |
| --- | --- | --- |
| 0 | `SKY` | (90, 140, 200) |
| 1 | `ROAD` | (70, 70, 75) |
| 2 | `LEFT_BARRIER` | (200, 40, 40) |
| 3 | `RIGHT_BARRIER` | (40, 90, 200) |
| 4 | `FINISH` | (240, 240, 240) |
| 5 | `OBSTACLE` | (240, 170, 30) |

`Game().get_label_names()` returns this table as `{id: name}`.

## Game variables

`raycer.GAME_VARIABLES`, in the order of the `gamevariables` vector:

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `speed` | m/s, signed |
| 1 | `lap_progress` | [0, 1) |
| 2 | `lap` | completed laps |
| 3 | `lateral_offset` | m, left is positive |
| 4 | `heading_error` | rad, (-pi, pi] |
| 5 | `wrong_way` | 0 or 1 |
| 6 | `collision` | 0 or 1, during the last step |
| 7 | `episode_time` | s |

## Depth

- `Game.get_state().depth_buffer` is `float32 (H, W)`: the perpendicular distance in metres.
  Pixels without a hit hold exactly `max_view_distance` (60 m), not `inf`.
- The Gym observation `depth` is quantized to `uint8 (H, W, 1)` as
  `round(255 * d / max_view_distance)`, so near is dark and 255 is the view limit. This lets
  Stable-Baselines3 treat it as an image.
- With `metric_depth_info=True` the env also returns the metric depth in `info["depth_m"]`.
- `raycer render-sample` writes `depth.png` inverted (near is bright).
