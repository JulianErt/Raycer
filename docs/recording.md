# Recording

## `RecordEpisode`

`raycer.recording.RecordEpisode(env, directory)` wraps a Dict-observation env and writes one
file per episode with `np.savez_compressed`:

```
<directory>/<scenario>_s<track_seed>_<YYYYmmdd-HHMMSS>.npz
```

`track_seed` is `none` for builtin tracks. Existing files are never overwritten; a `_1`, `_2`
suffix is added instead. An unfinished episode is written on the next `reset()` or on
`close()`. `last_path` holds the path of the last file.

## `.npz` layout

For an episode with T steps:

| Key | Content |
| --- | --- |
| `screen`, `depth`, `labels` | Observations, length T+1, only the active buffers |
| `gamevariables` | `float32 (T+1, 8)` |
| `actions` | `int64 (T,)` discrete or `float32 (T, 2)` continuous |
| `rewards` | `float32 (T,)` |
| `terminated`, `truncated` | `bool (T,)` |
| `meta` | 0-d string array with JSON: `raycer_version`, `scenario`, `seed`, `track_seed`, `action_type`, `frame_skip`, `width`, `height`, `label_names` |

## `load_episode`

`raycer.recording.load_episode(path)` loads a file with `allow_pickle=False` and returns a dict
of arrays, with `meta` parsed into a dict.

## `raycer play`

`raycer play --record DIR` records the episodes you drive. `raycer play --replay FILE` replays
the recorded actions with the same seed and track seed. Both need `raycer[human]`, and the two
flags cannot be combined. `--replay` skips the 3-2-1 countdown. The keys are listed in the
README section [Playing](../README.md#playing).
