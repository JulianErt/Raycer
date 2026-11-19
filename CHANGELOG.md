# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
