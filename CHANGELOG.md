# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `num_tracks` (`[track] num_tracks` and `RaycerEnv(num_tracks=N)`) limits procedural
  scenarios to the first N track seeds of their split.
- Optional noisy depth sensor: the `[depth_noise]` table, `Game.set_depth_noise` and
  `RaycerEnv(depth_noise=True)`; only the depth outputs change.

## [0.2.0] - 2026-11-20

### Added

- Skyline, sun lighting and road markings in the RGB buffer.
- Human viewer upgrade: scalable window, HUD, minimap, countdown, pause, finish and help
  overlays, ramped continuous controls, `raycer play --discrete` and F11 fullscreen.
- Domain randomization: the `[appearance]` table, `Game.set_appearance`, `info["appearance"]`,
  the scenarios `proc-train-dr-v0` and `proc-test-dr-v0` and their Gym IDs
  `Raycer/ProcTrainDR-v0`, `Raycer/ProcTestDR-v0`, `Raycer/ProcTrainDR-Continuous-v0` and
  `Raycer/ProcTestDR-Continuous-v0`.

### Changed

- `raycer play` uses continuous controls with ramps by default; `--discrete` selects the
  discrete action space.
- The golden RGB images changed with the new look.

## [0.1.0] - 2026-11-19

### Added

- Column raycasting renderer with pixel-aligned RGB, metric depth and semantic-label buffers.
- Builtin tracks `oval`, `kidney` and `technical`, and a procedural track generator with
  disjoint train, val and test seed splits.
- Car physics, progress tracking with laps, checkpoints, wrong-way and stuck detection.
- Low-level ViZDoom-style `raycer.Game` API with `GameState`.
- Gymnasium env `raycer.RaycerEnv` with Dict observations, discrete and continuous actions and
  12 registered IDs for six TOML scenarios.
- Wrappers `StackChannels` and `NormalizeGameVariables`, and episode recording with
  `RecordEpisode` and `load_episode`.
- CLI `raycer` with `benchmark`, `render-sample`, `list` and `play`, and a pygame viewer in the
  `human` extra.
- Runnable examples, documentation, `CITATION.cff` and this changelog.
