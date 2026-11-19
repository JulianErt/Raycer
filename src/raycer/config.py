"""Scenario configs (API-11): packaged TOML scenarios, user TOML and overrides."""

import os
import tomllib
from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from raycer.game import _DEFAULT_WEIGHTS
from raycer.track import PRESETS, SPLITS

# Code defaults and, at the same time, the allowed keys; None means "no default".
_SCHEMA: dict[str, Any] = {
    "name": None,
    "version": None,
    "laps": 1,
    "max_episode_steps": 3000,
    "track": {"type": None, "builtin": None, "preset": None, "split": None},
    "screen": {"width": 96, "height": 72},
    "buffers": {"screen": True, "depth": False, "labels": False},
    "physics": {"frame_skip": 4},
    "reward": {k.removeprefix("w_"): v for k, v in _DEFAULT_WEIGHTS.items()},
}
_BUILTINS = ("oval", "kidney", "technical")


def _scenarios_dir() -> Traversable:
    """Return the packaged scenarios directory."""
    return files("raycer").joinpath("scenarios")


def list_scenarios() -> list[str]:
    """Return the sorted names of all packaged scenarios."""
    return sorted(
        p.name.removesuffix(".toml") for p in _scenarios_dir().iterdir() if p.name.endswith(".toml")
    )


def _check_keys(data: Mapping[str, Any], source: str) -> None:
    """Raise ValueError for any key or section not in the schema."""
    for key, value in data.items():
        if key not in _SCHEMA:
            raise ValueError(f"{source}: unknown key {key!r}")
        section = _SCHEMA[key]
        if not isinstance(section, dict):
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{source}: {key!r} must be a table")
        for sub in value:
            if sub not in section:
                raise ValueError(f"{source}: unknown key '{key}.{sub}'")


def _read_layer(path: Traversable, *, scenario: bool = False) -> dict[str, Any]:
    """Read and key-check one TOML layer; a scenario layer must have name, version and track."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    _check_keys(data, str(path))
    if scenario:
        for key in ("name", "version", "track"):
            if key not in data:
                raise ValueError(f"{path}: missing required key {key!r}")
    return data


def _validate_track(track: Mapping[str, Any]) -> None:
    """Raise ValueError if the merged track settings are invalid."""
    kind = track.get("type")
    if kind == "builtin":
        if track.get("builtin") not in _BUILTINS:
            raise ValueError(f"track.builtin must be one of {list(_BUILTINS)}")
    elif kind == "procedural":
        if track.get("preset") not in PRESETS:
            raise ValueError(f"track.preset must be one of {list(PRESETS)}")
        if track.get("split") not in SPLITS:
            raise ValueError(f"track.split must be one of {list(SPLITS)}")
    else:
        raise ValueError(f"track.type must be 'builtin' or 'procedural', got {kind!r}")


def load_config(
    scenario: str,
    path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the merged config: overrides > user TOML > scenario TOML > code default."""
    names = list_scenarios()
    if scenario not in names:
        raise ValueError(f"unknown scenario {scenario!r}; available: {names}")
    layers = [_read_layer(_scenarios_dir().joinpath(f"{scenario}.toml"), scenario=True)]
    if path is not None:
        layers.append(_read_layer(Path(path)))
    if overrides is not None:
        _check_keys(overrides, "overrides")
        layers.append(dict(overrides))
    config = {k: dict(v) if isinstance(v, dict) else v for k, v in _SCHEMA.items()}
    for layer in layers:
        for key, value in layer.items():
            if isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    config["track"] = {k: v for k, v in config["track"].items() if v is not None}
    _validate_track(config["track"])
    return config
