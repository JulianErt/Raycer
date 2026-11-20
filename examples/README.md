# Examples

Run from the repo root. Outputs go to `runs/` and `recordings/` in the current directory
(both gitignored). Every script accepts `--steps N` to cap the run.

| File | Purpose | Run |
| --- | --- | --- |
| `random_agent.py` | Random actions on `Raycer/Oval-v0` with all buffers (Gymnasium API) | `uv run examples/random_agent.py` |
| `low_level_api.py` | Random actions through the low-level `raycer.Game` API | `uv run examples/low_level_api.py` |
| `heuristic_driver.py` | Proportional driver that finishes one lap of oval-v0 | `uv run examples/heuristic_driver.py` |
| `ppo_rgb_vs_depth.py` | PPO on ProcTrain, evaluated on ProcTest, with `--obs rgb/rgbd/rgbl` | `uv run examples/ppo_rgb_vs_depth.py --obs rgbd` |
| `record_demo.py` | Records 3 heuristic episodes to `.npz` | `uv run examples/record_demo.py` |
| `trajectories.py` | Drives PPO checkpoints on one ProcTest track, writes a trajectory PNG and GIF | `uv run examples/trajectories.py --train-steps 20000` |

`ppo_rgb_vs_depth.py` needs Stable-Baselines3 and torch: `uv sync --group train` first.

`trajectories.py` needs the train group and the `plot` extra: `uv sync --group train --extra plot`.
