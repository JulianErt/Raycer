import dataclasses
import hashlib
import inspect
import itertools
import math
import pickle
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.utils.passive_env_checker import env_render_passive_checker
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

import raycer
from raycer import LABEL_PALETTE, Game, RaycerEnv
from raycer.track import SPLITS

P = inspect.Parameter
INFO_KEYS = {
    "step",
    "lap",
    "lap_progress",
    "track_seed",
    "scenario",
    "reward_components",
    "collisions",
    "wrong_way_steps",
    "lap_times",
    "is_success",
}
BUFFERS = list(itertools.product([False, True], repeat=3))
IDS = [i for i in gymnasium.registry if i.startswith("Raycer/")]


def _env(**kwargs: Any) -> RaycerEnv:
    return RaycerEnv(**{"width": 64, "height": 48, **kwargs})


def test_signature_API03() -> None:
    params = list(inspect.signature(RaycerEnv).parameters.values())
    kw = P.KEYWORD_ONLY
    expected = [
        ("scenario", P.POSITIONAL_OR_KEYWORD, "oval-v0"),
        ("config_path", kw, None),
        ("render_mode", kw, None),
        ("action_type", kw, "discrete"),
        ("screen", kw, True),
        ("depth", kw, False),
        ("labels", kw, False),
        ("width", kw, 96),
        ("height", kw, 72),
        ("frame_skip", kw, 4),
        ("max_episode_steps", kw, None),
        ("reward", kw, None),
        ("metric_depth_info", kw, False),
        ("num_tracks", kw, None),
        ("depth_noise", kw, None),
    ]
    assert [(p.name, p.kind, p.default) for p in params] == expected
    assert RaycerEnv.metadata == {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    env = gymnasium.make("Raycer/Oval-v0")
    assert isinstance(env.unwrapped.game, Game)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="action_type"):
        RaycerEnv(action_type="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="depth"):
        RaycerEnv(metric_depth_info=True)


@pytest.mark.parametrize("buffers", BUFFERS)
def test_observation_keys_API04(buffers: tuple[bool, bool, bool]) -> None:
    screen, depth, labels = buffers
    env = _env(screen=screen, depth=depth, labels=labels)
    keys = {k for k, on in zip(("screen", "depth", "labels"), buffers, strict=True) if on}
    obs, _ = env.reset(seed=0)
    for o in (obs, env.step(1)[0]):
        assert set(o) == keys | {"gamevariables"}
        assert env.observation_space.contains(o)
    if depth:
        d = env.game.get_state().depth_buffer  # type: ignore[union-attr]
        mvd = env.game._renderer.max_view_distance
        assert np.array_equal(o["depth"], np.round(255 * d / mvd).astype(np.uint8)[..., None])


def test_info_and_reward_API06_API09() -> None:
    env = _env(depth=True)
    _, info = env.reset(seed=0)
    assert set(info) == INFO_KEYS
    assert set(info["reward_components"].values()) == {0.0}
    _, reward, _, _, info = env.step(1)
    assert set(info) == INFO_KEYS
    assert abs(sum(info["reward_components"].values()) - reward) < 1e-6

    env = _env(depth=True, metric_depth_info=True)
    _, info = env.reset(seed=0)
    assert set(info) == INFO_KEYS | {"depth_m"}
    assert info["depth_m"].dtype == np.float32
    assert info["depth_m"].shape == (48, 64)
    assert set(env.step(1)[4]) == INFO_KEYS | {"depth_m"}

    with pytest.raises(ValueError, match="w_bogus"):
        _env(reward={"w_bogus": 1.0})


def test_wrongway_weight_API09() -> None:
    def wrongway(weight: float) -> float:
        env = _env(reward={"w_wrongway": weight})
        env.reset(seed=0)
        car = env.game._car  # the track is too narrow to turn, so start backwards
        env.game._car = dataclasses.replace(car, heading=car.heading + math.pi)
        total = 0.0
        for _ in range(40):
            _, _, _, _, info = env.step(1)
            total += info["reward_components"]["wrongway"]
        assert info["wrong_way_steps"] > 0
        return total

    assert wrongway(0.05) < 0
    assert wrongway(0.5) == pytest.approx(10 * wrongway(0.05))


def test_reward_API09() -> None:
    env = _env()
    env.reset(seed=0)
    assert sum(env.step(1)[1] for _ in range(100)) > 0
    env = _env(reward={"w_progress": 0, "w_collision": 0, "w_wrongway": 0, "w_lap": 0})
    env.reset(seed=0)
    for i in range(30):
        assert env.step(i % 9)[1] == pytest.approx(-0.01 * 4, abs=1e-12)


def test_truncation_API10() -> None:
    env = _env(max_episode_steps=10)
    env.reset(seed=0)
    for i in range(1, 11):
        _, _, terminated, truncated, _ = env.step(1)
        assert not terminated
        assert truncated == (i == 10)
    with pytest.raises(RuntimeError):
        env.step(1)


def test_stuck_API10() -> None:
    env = _env()
    env.reset(seed=0)
    for i in range(1, 76):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert truncated == (i == 75)


def test_lap_terminates_API10() -> None:
    env = _env(action_type="continuous", screen=False)
    obs, _ = env.reset(seed=0)
    for _ in range(3000):
        gv = obs["gamevariables"]
        action = np.clip([-0.3 * gv[3] - 1.5 * gv[4], 0.6], -1, 1)
        obs, _, terminated, truncated, info = env.step(action)
        assert not truncated
        if terminated:
            break
    assert terminated
    assert info["is_success"] is True
    assert len(info["lap_times"]) == 1
    assert info["lap_times"][0] > 0
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2, np.float32))


def test_registration_API12_API13() -> None:
    assert len(IDS) == 16
    for env_id in IDS:
        spec = gymnasium.spec(env_id)
        assert spec.entry_point == "raycer.env:RaycerEnv"
        assert spec.max_episode_steps is None
        env = gymnasium.make(env_id, width=64, height=48)
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample())
        assert isinstance(env.action_space, gymnasium.spaces.Box) == ("Continuous" in env_id)


def _check(env: gymnasium.Env[Any, Any]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # API-04 fixes gamevariables to Box(-inf, inf); gymnasium warns about infinite bounds.
        warnings.filterwarnings("ignore", message=r".*(minimum|maximum) value is (-)?infinity")
        check_env(env, skip_render_check=True)


@pytest.mark.parametrize("env_id", IDS)
def test_check_env_ids_API16(env_id: str) -> None:
    _check(gymnasium.make(env_id, width=64, height=48).unwrapped)


@pytest.mark.parametrize("buffers", BUFFERS)
def test_check_env_buffers_API16(buffers: tuple[bool, bool, bool]) -> None:
    screen, depth, labels = buffers
    _check(_env(screen=screen, depth=depth, labels=labels))


def _digests(env: RaycerEnv, actions: list[int]) -> list[str]:
    """SHA-256 over observation, reward, flags and info after reset and each step."""
    obs, info = env.reset(seed=123)
    steps: list[tuple[Any, ...]] = [(obs, 0.0, False, False, info)]
    for a in actions:
        step = env.step(a)
        steps.append(step)
        if step[2] or step[3]:
            break
    out = []
    for obs, reward, terminated, truncated, info in steps:
        h = hashlib.sha256()
        for key in sorted(obs):
            h.update(obs[key].tobytes())
        h.update(repr((reward, terminated, truncated, sorted(info.items()))).encode())
        out.append(h.hexdigest())
    return out


@pytest.mark.parametrize("scenario", ["oval-v0", "proc-train-v0"])
def test_determinism_API14(scenario: str) -> None:
    rng = np.random.default_rng(1234)
    actions = [int(a) for a in rng.choice([1, 1, 1, 5, 6, 0, 2], size=500)]
    kwargs = {"depth": True, "labels": True}
    assert _digests(_env(scenario=scenario, **kwargs), actions) == _digests(
        _env(scenario=scenario, **kwargs), actions
    )


def test_track_seed_API14() -> None:
    env = _env(scenario="proc-train-v0")
    seed = env.reset(seed=7)[1]["track_seed"]
    assert seed in SPLITS["train"]
    env.reset()
    env.reset(seed=8)
    assert env.reset(seed=7)[1]["track_seed"] == seed
    assert env.reset(options={"track_seed": 42})[1]["track_seed"] == 42
    with pytest.raises(ValueError, match="split"):
        env.reset(options={"track_seed": 1_010_000})
    with pytest.raises(ValueError, match="track_seed"):
        _env().reset(options={"track_seed": 1})
    assert _env().reset()[1]["track_seed"] is None


def _track_seeds(env: RaycerEnv, n: int) -> set[int]:
    env.reset(seed=0)
    return {env.reset()[1]["track_seed"] for _ in range(n)}


def test_num_tracks_API23(tmp_path: Path) -> None:
    start = SPLITS["train"].start
    assert _track_seeds(_env(scenario="proc-train-v0", num_tracks=5), 500) == set(
        range(start, start + 5)
    )
    user = tmp_path / "user.toml"
    user.write_text("[track]\nnum_tracks = 3\n", encoding="utf-8")
    from_toml = _env(scenario="proc-train-v0", config_path=str(user))
    assert _track_seeds(from_toml, 200) == set(range(start, start + 3))
    both = _env(scenario="proc-train-v0", config_path=str(user), num_tracks=2)
    assert _track_seeds(both, 200) == set(range(start, start + 2))
    env = _env(scenario="proc-train-v0", num_tracks=5)
    assert env.reset(options={"track_seed": start + 4})[1]["track_seed"] == start + 4
    with pytest.raises(ValueError, match=r"track_seed.*num_tracks"):
        env.reset(options={"track_seed": start + 5})
    test_len = len(SPLITS["test"])
    _env(scenario="proc-test-v0", num_tracks=test_len)
    with pytest.raises(ValueError, match="num_tracks"):
        _env(scenario="proc-test-v0", num_tracks=test_len + 1)
    with pytest.raises(ValueError, match="num_tracks"):
        _env(scenario="proc-train-v0", num_tracks=0)
    with pytest.raises(ValueError, match="num_tracks"):
        _env(scenario="oval-v0", num_tracks=1)
    with pytest.raises(ValueError, match="num_tracks"):
        _env(scenario="oval-v0", config_path=str(user))


def test_render_API15() -> None:
    env = _env(render_mode="rgb_array", depth=True, labels=True)
    obs, _ = env.reset(seed=0)
    env_render_passive_checker(env)
    obs = env.step(1)[0]
    image = env.render()
    assert image is not None
    assert image.dtype == np.uint8
    assert image.shape == (96, 384, 3)

    def up(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

    assert np.array_equal(image[:, :128], up(obs["screen"]))
    assert np.array_equal(image[:, 128:256], up(np.repeat(obs["depth"], 3, axis=2)))
    assert np.array_equal(image[:, 256:], up(LABEL_PALETTE[obs["labels"][..., 0]]))

    env = _env(render_mode="rgb_array", screen=False)
    env.reset(seed=0)
    assert env.render().shape == (96, 0, 3)  # type: ignore[union-attr]
    env = _env()
    env.reset(seed=0)
    assert env.render() is None
    with pytest.raises(ValueError, match="render_mode"):
        _env(render_mode="bogus")


def test_no_pygame_API15() -> None:
    code = (
        "import sys, gymnasium, raycer\n"
        "env = gymnasium.make('Raycer/Oval-v0', render_mode='rgb_array')\n"
        "env.reset(seed=0); env.step(1); env.render()\n"
        "import raycer.cli\n"
        "assert raycer.cli.main(['list']) == 0\n"
        "assert 'pygame' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_pygame_unimportable_API15() -> None:
    """REND-11: raycer works without pygame for the rgb_array mode."""
    code = (
        "import sys\n"
        "sys.modules['pygame'] = None\n"
        "import raycer\n"
        "env = raycer.RaycerEnv(render_mode='rgb_array')\n"
        "env.reset(seed=0); env.step(1)\n"
        "assert env.render() is not None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pickle_API16() -> None:
    env = _env(depth=True)
    env.reset(seed=3)
    env.step(1)
    clone = pickle.loads(pickle.dumps(env))
    a, b = env.step(5), clone.step(5)
    assert a[1] == b[1]
    for key in a[0]:
        assert np.array_equal(a[0][key], b[0][key])


def test_sync_vector_API16() -> None:
    envs = SyncVectorEnv([lambda: _env() for _ in range(2)])
    envs.reset(seed=0)
    for _ in range(10):
        envs.step(envs.action_space.sample())
    envs.close()


def test_async_vector_API16() -> None:
    envs = AsyncVectorEnv([_make_small for _ in range(4)], context="spawn")
    envs.reset(seed=0)
    envs.action_space.seed(0)
    for _ in range(1000):
        envs.step(envs.action_space.sample())
    envs.close()


def _make_small() -> RaycerEnv:
    return RaycerEnv(width=64, height=48)


def test_make_vec_API16() -> None:
    envs = gymnasium.make_vec("Raycer/Oval-v0", num_envs=2, vectorization_mode="async")
    envs.reset(seed=0)
    envs.step(envs.action_space.sample())
    envs.close()
    assert raycer.RaycerEnv is RaycerEnv


APPEARANCE_KEYS = {
    "hue_offset_deg",
    "brightness",
    "saturation",
    "sky_hue_deg",
    "swap_left",
    "swap_right",
    "swap_obstacle",
    "swap_road",
    "swap_finish",
    "period_scale",
    "skyline_phase",
}


@pytest.mark.parametrize("split", ["train", "test"])
def test_dr_keeps_depth_labels_and_tracks_API21(split: str) -> None:
    kwargs: dict[str, Any] = {"depth": True, "labels": True}
    dr, plain = (
        _env(scenario=f"proc-{split}-dr-v0", **kwargs),
        _env(scenario=f"proc-{split}-v0", **kwargs),
    )
    actions = np.random.default_rng(0).integers(9, size=50)
    for seed in (0, 1, 2):
        (obs_a, info_a), (obs_b, info_b) = dr.reset(seed=seed), plain.reset(seed=seed)
        rgb_differs = False
        for a in [None, *actions]:
            if a is not None:
                obs_a, *_, info_a = dr.step(int(a))
                obs_b, *_, info_b = plain.step(int(a))
            assert np.array_equal(obs_a["depth"], obs_b["depth"])
            assert np.array_equal(obs_a["labels"], obs_b["labels"])
            assert info_a["track_seed"] == info_b["track_seed"]
            rgb_differs |= not np.array_equal(obs_a["screen"], obs_b["screen"])
            # REND-06 point 6 under DR: sky-labelled pixels only show sky colours.
            sky = dr.game._renderer.sky_colors
            px = obs_a["screen"][obs_a["labels"][..., 0] == 0]
            assert np.all((px[:, None, :] == sky[None]).all(axis=2).any(axis=1))
        assert rgb_differs


def test_dr_determinism_API21() -> None:
    a, b = _env(scenario="proc-test-dr-v0"), _env(scenario="proc-test-dr-v0")
    obs_a, obs_b = a.reset(seed=5)[0], b.reset(seed=5)[0]
    assert obs_a["screen"].tobytes() == obs_b["screen"].tobytes()
    for action in np.random.default_rng(1).integers(9, size=20):
        obs_a = a.step(int(action))[0]
        obs_b = b.step(int(action))[0]
        assert obs_a["screen"].tobytes() == obs_b["screen"].tobytes()


@pytest.mark.parametrize("scenario", ["proc-train-dr-v0", "proc-test-dr-v0"])
def test_dr_info_appearance_API21(scenario: str) -> None:
    env = _env(scenario=scenario)
    infos = [env.reset(seed=0)[1], env.step(0)[4]]
    for info in infos:
        assert set(info["appearance"]) == APPEARANCE_KEYS
        assert all(type(v) is float for v in info["appearance"].values())
    oval = _env()
    assert "appearance" not in oval.reset(seed=0)[1]
    assert "appearance" not in oval.step(0)[4]


def test_dr_gym_make_API22() -> None:
    env = gymnasium.make("Raycer/ProcTestDR-Continuous-v0", width=64, height=48)
    assert "appearance" in env.reset(seed=0)[1]


def test_depth_noise_kwarg_API24() -> None:
    kwargs: dict[str, Any] = {"depth": True, "labels": True, "metric_depth_info": True}
    noisy, twin, plain = (
        RaycerEnv("oval-v0", depth_noise=True, **kwargs),
        RaycerEnv("oval-v0", depth_noise=True, **kwargs),
        RaycerEnv("oval-v0", **kwargs),
    )
    assert noisy.game._depth_noise[0]
    assert not plain.game._depth_noise[0]
    (obs_a, info_a), (obs_t, _), (obs_b, info_b) = (e.reset(seed=0) for e in (noisy, twin, plain))
    for a in [None, *np.random.default_rng(0).integers(9, size=20)]:
        if a is not None:
            obs_a, r_a, *_, info_a = noisy.step(int(a))
            obs_t, r_t, *_ = twin.step(int(a))
            obs_b, r_b, *_, info_b = plain.step(int(a))
            assert r_a == r_b == r_t
        for key in ("screen", "labels", "gamevariables"):
            assert np.array_equal(obs_a[key], obs_b[key])
        assert not np.array_equal(obs_a["depth"], obs_b["depth"])
        assert not np.array_equal(info_a["depth_m"], info_b["depth_m"])
        assert np.array_equal(obs_a["depth"], obs_t["depth"])
