# Scenarios

Scenarios are TOML files shipped as package data in `raycer/scenarios/` and read with
`tomllib`. `raycer.list_scenarios()` returns their names; `raycer list` prints them with their
Gym IDs.

## TOML format

```toml
name = "oval"                 # required
version = 0                   # required
laps = 1
max_episode_steps = 3000      # env steps

[track]                       # required
type = "builtin"              # "builtin" or "procedural"
builtin = "oval"              # builtin only: oval, kidney or technical
preset = "medium"             # procedural only: easy, medium or hard
split = "train"               # procedural only: train, val or test
num_tracks = 100              # procedural only, optional: use only the first N seeds of the
                              # split (1 to split length); default is the whole split

[screen]
width = 96
height = 72

[buffers]
screen = true
depth = false
labels = false

[physics]
frame_skip = 4

[reward]                      # API reward keys without "w_"
progress = 0.1
time = 0.01
collision = 1.0
wrongway = 0.05
lap = 10.0

[appearance]                  # domain randomization, see the README
randomize = false
split = "train"               # "train" or "test" hue range
strength = 1.0                # 0 to 1
```

Unknown keys raise `ValueError` naming the file and the key.

## Merge precedence

`RaycerEnv` merges, from highest to lowest priority:

1. env kwargs (`width`, `height`, `screen`, `depth`, `labels`, `frame_skip`,
   `max_episode_steps`; a kwarg left at its default does not override), then `reward=`
2. the user TOML passed as `config_path`
3. the scenario TOML
4. the code defaults shown above

## Shipped scenarios

| Name | Gym ID | Track | Laps | Max steps | Purpose |
| --- | --- | --- | --- | --- | --- |
| `oval-v0` | `Raycer/Oval-v0` | builtin `oval` | 1 | 3000 | default, smoke test |
| `kidney-v0` | `Raycer/Kidney-v0` | builtin `kidney` | 1 | 3000 | fixed track, medium |
| `technical-v0` | `Raycer/Technical-v0` | builtin `technical` | 1 | 4000 | fixed track with obstacles |
| `proc-train-v0` | `Raycer/ProcTrain-v0` | procedural `medium`, split train | 1 | 4000 | training distribution |
| `proc-test-v0` | `Raycer/ProcTest-v0` | procedural `medium`, split test | 1 | 4000 | evaluation only, disjoint |
| `proc-train-dr-v0` | `Raycer/ProcTrainDR-v0` | procedural `medium`, split train, appearance `train` | 1 | 4000 | training with randomized appearance |
| `proc-test-dr-v0` | `Raycer/ProcTestDR-v0` | procedural `medium`, split test, appearance `test` | 1 | 4000 | evaluation on unseen appearance |
| `time-trial-v0` | `Raycer/TimeTrial-v0` | builtin `oval` | 3 | 9000 | `w_time = 0.05`, lap times |

`import raycer` registers each ID plus a `-Continuous-v0` variant (for example
`Raycer/ProcTrain-Continuous-v0`), 16 IDs in total, all with
`entry_point="raycer.env:RaycerEnv"`. The version suffix is raised whenever observations,
rewards or dynamics change for the same seed.

`Game.set_scenario` only knows the builtin-track scenarios; procedural scenarios need the env.
