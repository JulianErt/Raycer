import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

import raycer
from raycer import config
from raycer.config import list_scenarios, load_config

NAMES = ["kidney-v0", "oval-v0", "proc-test-v0", "proc-train-v0", "technical-v0", "time-trial-v0"]
DEFAULT_REWARD = {"progress": 0.1, "time": 0.01, "collision": 1.0, "wrongway": 0.05, "lap": 10.0}
EXPECTED = {
    "oval-v0": ("oval", {"type": "builtin", "builtin": "oval"}, 1, 3000),
    "kidney-v0": ("kidney", {"type": "builtin", "builtin": "kidney"}, 1, 3000),
    "technical-v0": ("technical", {"type": "builtin", "builtin": "technical"}, 1, 4000),
    "proc-train-v0": (
        "proc-train",
        {"type": "procedural", "preset": "medium", "split": "train"},
        1,
        4000,
    ),
    "proc-test-v0": (
        "proc-test",
        {"type": "procedural", "preset": "medium", "split": "test"},
        1,
        4000,
    ),
    "time-trial-v0": ("time-trial", {"type": "builtin", "builtin": "oval"}, 3, 9000),
}


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "user.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_list_scenarios() -> None:
    assert raycer.list_scenarios() == NAMES
    assert "list_scenarios" in raycer.__all__


@pytest.mark.parametrize("scenario", NAMES)
def test_scenario_values(scenario: str) -> None:
    cfg = load_config(scenario)
    name, track, laps, steps = EXPECTED[scenario]
    assert cfg["name"] == name
    assert cfg["version"] == 0
    assert cfg["track"] == track
    assert cfg["laps"] == laps
    assert cfg["max_episode_steps"] == steps
    assert cfg["screen"] == {"width": 96, "height": 72}
    assert cfg["buffers"] == {"screen": True, "depth": False, "labels": False}
    assert cfg["physics"] == {"frame_skip": 4}
    reward = dict(DEFAULT_REWARD, time=0.05) if scenario == "time-trial-v0" else DEFAULT_REWARD
    assert cfg["reward"] == reward


def test_stem_matches_name_and_version() -> None:
    root = Path(raycer.__file__).parent / "scenarios"
    for scenario in list_scenarios():
        data = tomllib.loads((root / f"{scenario}.toml").read_text(encoding="utf-8"))
        assert scenario == f"{data['name']}-v{data['version']}"


@pytest.mark.parametrize(
    ("text", "key"),
    [("[track]\nbiultin = 'oval'\n", "track.biultin"), ("lpas = 2\n", "lpas")],
)
def test_user_toml_typo_key(tmp_path: Path, text: str, key: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ValueError, match=key) as exc:
        load_config("oval-v0", path)
    assert str(path) in str(exc.value)


def test_user_toml_section_not_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="screen"):
        load_config("oval-v0", _write(tmp_path, "screen = 3\n"))


def test_overrides_typo_key() -> None:
    with pytest.raises(ValueError, match=r"overrides.*reward\.tiem"):
        load_config("oval-v0", overrides={"reward": {"tiem": 0.5}})
    with pytest.raises(ValueError, match="widht"):
        load_config("oval-v0", overrides={"widht": 64})


@pytest.mark.parametrize("missing", ["name", "version", "track"])
def test_scenario_missing_required_key(tmp_path: Path, missing: str) -> None:
    lines = {
        "name": 'name = "x"\n',
        "version": "version = 0\n",
        "track": '[track]\ntype = "builtin"\n',
    }
    path = _write(tmp_path, "".join(v for k, v in lines.items() if k != missing))
    with pytest.raises(ValueError, match=missing):
        config._read_layer(path, scenario=True)


def test_merge_order(tmp_path: Path) -> None:
    path = _write(tmp_path, "[screen]\nwidth = 64\n[reward]\ntime = 0.2\n")
    cfg = load_config("oval-v0", path, {"reward": {"time": 0.3}})
    assert cfg["screen"] == {"width": 64, "height": 72}
    assert cfg["reward"]["time"] == 0.3
    assert cfg["reward"]["lap"] == 10.0
    assert cfg["max_episode_steps"] == 3000


@pytest.mark.parametrize("name", ["nope-v0", "../pyproject", "/etc/passwd", "oval-v0.toml"])
def test_unknown_scenario(name: str) -> None:
    with pytest.raises(ValueError, match="available"):
        load_config(name)


@pytest.mark.parametrize(
    ("scenario", "track"),
    [
        ("oval-v0", {"type": "maze"}),
        ("oval-v0", {"builtin": "square"}),
        ("oval-v0", {"type": "procedural"}),
        ("proc-train-v0", {"preset": "insane"}),
        ("proc-train-v0", {"split": "holdout"}),
    ],
)
def test_invalid_track_via_overrides(scenario: str, track: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="track"):
        load_config(scenario, overrides={"track": track})


def test_invalid_track_via_user_toml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"track\.split"):
        load_config("proc-test-v0", _write(tmp_path, '[track]\nsplit = "nope"\n'))


def test_wheel_contains_scenarios(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("*.whl")
    assert wheel.name == f"raycer-{raycer.__version__}-py3-none-any.whl"  # PKG-04
    entries = set(zipfile.ZipFile(wheel).namelist())
    for name in NAMES:
        assert f"raycer/scenarios/{name}.toml" in entries
