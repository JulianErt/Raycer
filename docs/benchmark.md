# Benchmark

## Usage

```bash
raycer benchmark --scenario technical-v0 --steps 5000 --width 96 --height 72 --buffers screen,depth,labels --seed 0
raycer benchmark --steps 300 --json
raycer benchmark --min-sps 250
```

The benchmark runs 200 warm-up steps, then `--steps` random discrete actions through
`raycer.Game`, starting a new episode whenever one finishes. It then times the renderer and the
physics alone for 200 iterations each. `--scenario` must be a builtin-track scenario.

## Output

One `key: value` line per field, or one JSON object with `--json`:

| Field | Meaning |
| --- | --- |
| `steps_per_second` | Env steps per second |
| `median_ms_per_step` | Median time of one env step in ms |
| `ms_per_render` | Mean time of one render in ms |
| `ms_per_physics_step` | Mean time of one physics tick in ms |
| `platform`, `python_version`, `numpy_version`, `raycer_version` | Environment |
| `scenario`, `steps`, `width`, `height`, `buffers`, `seed` | The arguments used |

With `--min-sps N` the command exits with code 1 when `steps_per_second` is below N.

## `RAYCER_MIN_SPS`

The quality gate `scripts/check.sh` ends with a benchmark of `technical-v0` (2000 steps, all
buffers) and passes `--min-sps "${RAYCER_MIN_SPS:-250}"`. Set the environment variable
`RAYCER_MIN_SPS` to change the threshold on slower or faster machines.
