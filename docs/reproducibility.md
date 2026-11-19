# Reproducibility

## Seeding

- `env.reset(seed=s)` seeds `env.np_random`, the single PCG64 generator of the episode.
  Raycer uses neither the global `np.random` nor `random` nor the wall clock.
- Physics and rendering are deterministic: two envs reset with the same seed and fed the same
  actions produce bit-identical observations on the same platform.
- `Game.new_episode(seed)` and `Game.set_seed` store the seed; builtin tracks always start at
  the same position, so the episode does not depend on it.

## Track seed and episode seed

- Builtin tracks (`oval`, `kidney`, `technical`) are fixed; `info["track_seed"]` is `None`.
- Procedural scenarios (`proc-train-v0`, `proc-test-v0`) generate a new track each episode. The
  episode RNG draws the track seed uniformly from the scenario's split, and
  `info["track_seed"]` reports it.
- `reset(options={"track_seed": k})` pins the track. A `k` outside the scenario's split raises
  `ValueError`, and so does any `track_seed` for a builtin track.

## Splits

| Split | Track seeds |
| --- | --- |
| `train` | 0 to 999 999 |
| `val` | 1 000 000 to 1 009 999 |
| `test` | 1 010 000 to 1 019 999 |

The splits are disjoint. `proc-train-v0` uses `train` and `proc-test-v0` uses `test`; `val` is
available to custom scenarios via `split = "val"`.

## Appearance splits

With appearance randomization on (`[appearance] randomize = true` or `Game.set_appearance`),
the appearance parameters of each episode are drawn from `PCG64([episode_seed, 0xA77])`, a
stream separate from the track. The hue offset lies in [0, 270) degrees for `train` and in
[270, 360) degrees for `test`. The track, depth and labels are identical with and without
randomization.
