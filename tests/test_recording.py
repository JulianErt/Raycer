import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import raycer
from raycer import RaycerEnv
from raycer.recording import RecordEpisode, load_episode
from raycer.wrappers import StackChannels

W, H = 32, 24
META_KEYS = {
    "raycer_version",
    "scenario",
    "seed",
    "track_seed",
    "action_type",
    "frame_skip",
    "width",
    "height",
    "label_names",
}


def _env(scenario: str = "oval-v0", **kwargs: Any) -> RaycerEnv:
    return RaycerEnv(scenario, **{"width": W, "height": H, "max_episode_steps": 20, **kwargs})


def _run(env: RecordEpisode, seed: int | None = 7, agent_seed: int = 0) -> dict[str, Any]:
    env.action_space.seed(agent_seed)
    _, info = env.reset(seed=seed)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
    return info


def test_record_and_replay_API19(tmp_path: Path) -> None:
    env = RecordEpisode(_env(), tmp_path)
    _run(env)
    files = list(tmp_path.glob("*.npz"))
    assert files == [env.last_path]
    assert re.fullmatch(r"oval-v0_snone_\d{8}-\d{6}\.npz", files[0].name)
    ep = load_episode(files[0])
    replay = _env()
    obs, _ = replay.reset(seed=ep["meta"]["seed"])
    screens = [obs["screen"]]
    for a in ep["actions"]:
        screens.append(replay.step(a)[0]["screen"])
    assert np.array_equal(np.stack(screens), ep["screen"])


@pytest.mark.parametrize("action_type", ["discrete", "continuous"])
def test_file_content(tmp_path: Path, action_type: str) -> None:
    env = RecordEpisode(_env(action_type=action_type, frame_skip=2), tmp_path)
    _run(env)
    assert env.last_path is not None
    with np.load(env.last_path, allow_pickle=False) as data:
        assert set(data.files) == {
            "screen",
            "gamevariables",
            "pose",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "meta",
        }
        t = len(data["actions"])
        assert t == 20
        assert data["screen"].dtype == np.uint8
        assert data["screen"].shape == (t + 1, H, W, 3)
        assert data["gamevariables"].dtype == np.float32
        assert data["gamevariables"].shape == (t + 1, 8)
        if action_type == "discrete":
            assert data["actions"].dtype == np.int64
            assert data["actions"].shape == (t,)
        else:
            assert data["actions"].dtype == np.float32
            assert data["actions"].shape == (t, 2)
        assert data["rewards"].dtype == np.float32
        assert data["rewards"].shape == (t,)
        for k in ("terminated", "truncated"):
            assert data[k].dtype == np.bool_
            assert data[k].shape == (t,)
        assert data["truncated"][-1]
        assert not data["truncated"][:-1].any()
        assert data["meta"].shape == ()
        assert data["meta"].dtype.kind == "U"
        meta = json.loads(str(data["meta"]))
    assert set(meta) == META_KEYS
    assert meta["raycer_version"] == raycer.__version__
    assert meta["scenario"] == "oval-v0"
    assert meta["seed"] == 7
    assert meta["track_seed"] is None
    assert meta["action_type"] == action_type
    assert (meta["frame_skip"], meta["width"], meta["height"]) == (2, W, H)
    names = env.unwrapped.game.get_label_names()  # type: ignore[attr-defined]
    assert meta["label_names"] == [names[i] for i in range(6)]


def test_depth_labels_and_procedural(tmp_path: Path) -> None:
    env = RecordEpisode(_env("proc-train-v0", depth=True, labels=True), tmp_path)
    info = _run(env)
    assert env.last_path is not None
    assert f"_s{info['track_seed']}_" in env.last_path.name
    ep = load_episode(env.last_path)
    assert ep["meta"]["track_seed"] == info["track_seed"]
    t = len(ep["actions"])
    for k, c in (("depth", 1), ("labels", 1)):
        assert ep[k].dtype == np.uint8
        assert ep[k].shape == (t + 1, H, W, c)


def test_no_overwrite_partial_and_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = datetime(2026, 1, 2, 3, 4, 5)
    monkeypatch.setattr(
        "raycer.recording.datetime", type("D", (), {"now": staticmethod(lambda: fixed)})
    )
    env = RecordEpisode(_env(max_episode_steps=2), tmp_path)
    _run(env)
    first = env.last_path
    _run(env)
    assert first is not None
    assert env.last_path is not None
    assert first.name == "oval-v0_snone_20260102-030405.npz"
    assert env.last_path.name == "oval-v0_snone_20260102-030405_1.npz"
    assert len(list(tmp_path.glob("*.npz"))) == 2
    assert env.last_path != first
    assert env.last_path.name.endswith(".npz")
    # Zero-step episode is skipped; a partial episode is written on reset and on close.
    env.reset(seed=1)
    env.reset(seed=2)
    assert len(list(tmp_path.glob("*.npz"))) == 2
    env.step(0)
    env.reset(seed=3)
    assert len(list(tmp_path.glob("*.npz"))) == 3
    env.step(0)
    env.close()
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 4
    ep = load_episode(env.last_path)
    assert isinstance(ep["meta"], dict)
    assert ep["meta"]["seed"] == 3
    assert isinstance(ep["actions"], np.ndarray)
    assert len(ep["actions"]) == 1
    env.close()
    assert len(list(tmp_path.glob("*.npz"))) == 4


def test_observations_unchanged_and_needs_dict(tmp_path: Path) -> None:
    env = RecordEpisode(_env(), tmp_path)
    raw = _env()
    a, _ = env.reset(seed=7)
    b, _ = raw.reset(seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in b)
    ra, rb = env.step(1), raw.step(1)
    assert ra[1:4] == rb[1:4]
    assert all(np.array_equal(ra[0][k], rb[0][k]) for k in rb[0])
    with pytest.raises(ValueError, match="Dict"):
        RecordEpisode(StackChannels(_env(), keys=("screen",)), tmp_path)  # type: ignore[arg-type]


def test_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    RecordEpisode(_env(), target)
    assert target.is_dir()


def test_pose_API25(tmp_path: Path) -> None:
    env = RecordEpisode(_env(), tmp_path)
    _, info = env.reset(seed=0)
    poses = [info["pose"]]
    done = False
    while not done:
        _, _, term, trunc, info = env.step(1)
        poses.append(info["pose"])
        done = term or trunc
    assert env.last_path is not None
    ep = load_episode(env.last_path)
    assert ep["pose"].dtype == np.float64
    assert ep["pose"].shape == (len(ep["actions"]) + 1, 3)
    assert ep["pose"].tolist() == [list(p) for p in poses]
