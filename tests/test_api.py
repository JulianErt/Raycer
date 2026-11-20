import hashlib
import inspect
import itertools
from pathlib import Path

import numpy as np
import pytest

from raycer import DISCRETE_ACTIONS, GAME_VARIABLES, Game, GameState, palette

P = inspect.Parameter
_POS = P.POSITIONAL_OR_KEYWORD
SIGNATURES: dict[str, list[tuple[str, inspect._ParameterKind, object]]] = {
    "__init__": [],
    "load_config": [("path", _POS, P.empty)],
    "set_scenario": [("name", _POS, P.empty)],
    "set_screen_resolution": [("width", _POS, P.empty), ("height", _POS, P.empty)],
    "set_screen_buffer_enabled": [("enabled", _POS, P.empty)],
    "set_depth_buffer_enabled": [("enabled", _POS, P.empty)],
    "set_labels_buffer_enabled": [("enabled", _POS, P.empty)],
    "set_episode_timeout": [("steps", _POS, P.empty)],
    "set_frame_skip": [("n", _POS, P.empty)],
    "set_appearance": [
        ("randomize", _POS, P.empty),
        ("split", _POS, "train"),
        ("strength", _POS, 1.0),
    ],
    "set_depth_noise": [
        ("enabled", _POS, P.empty),
        ("sigma_rel", _POS, 0.02),
        ("dropout", _POS, 0.01),
        ("quant", _POS, 0.05),
    ],
    "set_reward_config": [("weights", P.VAR_KEYWORD, P.empty)],
    "set_seed": [("seed", _POS, P.empty)],
    "init": [],
    "new_episode": [("seed", _POS, None)],
    "make_action": [("action", _POS, P.empty)],
    "get_state": [],
    "get_track": [],
    "is_episode_finished": [],
    "is_episode_timeout_reached": [],
    "get_episode_time": [],
    "get_total_reward": [],
    "get_last_reward_components": [],
    "get_label_names": [],
    "get_game_variable_names": [],
    "close": [],
}
SETTERS = {
    "load_config": ("x.toml",),
    "set_scenario": ("oval-v0",),
    "set_screen_resolution": (64, 48),
    "set_screen_buffer_enabled": (True,),
    "set_depth_buffer_enabled": (True,),
    "set_labels_buffer_enabled": (True,),
    "set_episode_timeout": (10,),
    "set_frame_skip": (2,),
    "set_appearance": (True,),
    "set_depth_noise": (True,),
}


def _game(*, buffers: tuple[bool, bool, bool] = (True, False, False), **setters: object) -> Game:
    """Initialised 64x48 game with the given buffers and extra single-argument setters."""
    game = Game()
    game.set_screen_resolution(64, 48)
    game.set_screen_buffer_enabled(buffers[0])
    game.set_depth_buffer_enabled(buffers[1])
    game.set_labels_buffer_enabled(buffers[2])
    for name, value in setters.items():
        getattr(game, name)(value)
    game.init()
    return game


def _state(game: Game) -> GameState:
    state = game.get_state()
    assert state is not None
    return state


def test_signatures() -> None:
    public = {n for n, _ in inspect.getmembers(Game, inspect.isfunction) if not n.startswith("_")}
    assert public == set(SIGNATURES) - {"__init__"}
    for name, expected in SIGNATURES.items():
        params = list(inspect.signature(getattr(Game, name)).parameters.values())
        assert params[0].name == "self"
        assert [(p.name, p.kind, p.default) for p in params[1:]] == expected, name


def test_setters_after_init_raise() -> None:
    game = _game()
    for name, args in SETTERS.items():
        with pytest.raises(RuntimeError, match=r"must be called before init\(\)"):
            getattr(game, name)(*args)
    game.set_seed(3)
    game.set_reward_config(w_time=0.0)


def test_make_action_runtime_errors() -> None:
    with pytest.raises(RuntimeError):
        Game().make_action(0)
    game = _game()
    game.close()
    with pytest.raises(RuntimeError):
        game.make_action(0)
    assert game.get_state() is None

    game = _game(set_episode_timeout=1)
    assert not game.is_episode_finished()
    game.make_action(1)
    with pytest.raises(RuntimeError):
        game.make_action(1)
    assert game.get_state() is None
    assert game.is_episode_finished()
    assert game.is_episode_timeout_reached()


@pytest.mark.parametrize("buffers", list(itertools.product([False, True], repeat=3)))
def test_buffer_combinations(buffers: tuple[bool, bool, bool]) -> None:
    game = _game(buffers=buffers)
    game.make_action(1)
    state = _state(game)
    expected = [((48, 64, 3), np.uint8), ((48, 64), np.float32), ((48, 64), np.uint8)]
    bufs = [state.screen_buffer, state.depth_buffer, state.labels_buffer]
    for on, buf, (shape, dtype) in zip(buffers, bufs, expected, strict=True):
        if on:
            assert buf is not None
            assert buf.shape == shape
            assert buf.dtype == dtype
        else:
            assert buf is None
    assert state.game_variables.dtype == np.float32
    assert state.game_variables.shape == (8,)
    assert state.number == game.get_episode_time() == 1


def test_buffers_are_copies() -> None:
    game = _game(buffers=(True, True, True))
    first = _state(game)
    saved = [b.copy() for b in (first.screen_buffer, first.depth_buffer, first.labels_buffer)]
    for buf in (first.screen_buffer, first.depth_buffer, first.labels_buffer):
        assert buf is not None
        buf[...] = 7
    first.game_variables[...] = 7
    second = _state(game)
    for old, new in zip(
        saved, (second.screen_buffer, second.depth_buffer, second.labels_buffer), strict=True
    ):
        assert np.array_equal(old, new)
    assert second.game_variables[2] == 0


def test_game_variables_at_start() -> None:
    names = "speed lap_progress lap lateral_offset heading_error wrong_way collision episode_time"
    assert tuple(names.split()) == GAME_VARIABLES == Game().get_game_variable_names()
    game = _game()
    for _ in range(5):
        game.make_action(5)
    game.new_episode()
    gv = _state(game).game_variables
    assert gv[0] == 0
    assert gv[2] == 0
    assert abs(gv[3]) < 1e-6
    assert abs(gv[4]) < 1e-6
    assert gv[7] == 0
    assert game.get_label_names()[1] == "ROAD"


def test_discrete_matches_continuous() -> None:
    table = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    assert table == list(DISCRETE_ACTIONS)
    for i in range(9):
        a, b = _game(), _game()
        for _ in range(20):
            assert a.make_action(i) == b.make_action(DISCRETE_ACTIONS[i])
            sa, sb = _state(a), _state(b)
            assert np.array_equal(sa.game_variables, sb.game_variables)
            assert np.array_equal(sa.screen_buffer, sb.screen_buffer)
    with pytest.raises(ValueError, match="in 0"):
        _game().make_action(9)
    with pytest.raises(ValueError, match="steer"):
        _game().make_action([1.0, 0.0, 0.0])
    assert _game().make_action(np.int64(1)) == _game().make_action(1)


def test_continuous_clipping() -> None:
    a, b = _game(), _game()
    for _ in range(10):
        assert a.make_action([5.0, 5.0]) == b.make_action([1.0, 1.0])
    assert np.array_equal(_state(a).game_variables, _state(b).game_variables)


def test_validation() -> None:
    game = Game()
    for n in (0, 17):
        with pytest.raises(ValueError, match="frame_skip"):
            game.set_frame_skip(n)
    with pytest.raises(ValueError, match="w_bogus"):
        game.set_reward_config(w_bogus=1.0)
    with pytest.raises(ValueError, match="timeout"):
        game.set_episode_timeout(-1)
    game.set_screen_resolution(63, 48)
    with pytest.raises(ValueError, match="width"):
        game.init()


def test_reward_components_sum() -> None:
    game = _game()
    total = 0.0
    for _ in range(100):
        reward = game.make_action(1)
        comps = game.get_last_reward_components()
        assert set(comps) == {"progress", "time", "collision", "wrongway", "lap"}
        assert abs(sum(comps.values()) - reward) < 1e-6
        assert comps["time"] <= 0
        assert comps["collision"] <= 0
        assert comps["wrongway"] <= 0
        total += reward
    assert total > 0
    assert game.get_total_reward() == pytest.approx(total)
    game.new_episode()
    assert game.get_total_reward() == 0.0
    assert set(game.get_last_reward_components().values()) == {0.0}


@pytest.mark.parametrize("frame_skip", [4, 7])
def test_time_only_reward(frame_skip: int) -> None:
    game = _game(set_frame_skip=frame_skip)
    game.set_reward_config(w_progress=0, w_collision=0, w_wrongway=0, w_lap=0)
    game.new_episode()
    for i in range(30):
        assert abs(game.make_action(i % 9) + 0.01 * frame_skip) < 1e-12


def _digests(game: Game, actions: list[int]) -> list[str]:
    """SHA-256 over all buffers and game variables after each step."""
    out = []
    for a in actions:
        game.make_action(a)
        s = _state(game)
        h = hashlib.sha256()
        for buf in (s.screen_buffer, s.depth_buffer, s.labels_buffer, s.game_variables):
            assert buf is not None
            h.update(buf.tobytes())
        out.append(h.hexdigest())
    return out


def test_determinism() -> None:
    rng = np.random.default_rng(1234)
    actions = [int(a) for a in rng.choice([1, 1, 1, 5, 6, 0, 2], size=300)]
    a = _game(buffers=(True, True, True), set_seed=7)
    b = _game(buffers=(True, True, True), set_seed=7)
    ref = _digests(a, actions)
    assert ref == _digests(b, actions)
    _digests(a, [4] * 50)
    a.new_episode(7)
    assert ref == _digests(a, actions)


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "cfg.toml"
    path.write_text(
        "max_episode_steps = 5\nunknown = 1\n"
        "[screen]\nwidth = 64\nheight = 48\n"
        "[buffers]\ndepth = true\n"
        "[physics]\nframe_skip = 2\n"
        "[reward]\ntime = 0.5\n"
    )
    game = Game()
    game.load_config(path)
    game.init()
    state = _state(game)
    assert state.screen_buffer is not None
    assert state.screen_buffer.shape == (48, 64, 3)
    assert state.depth_buffer is not None
    assert state.depth_buffer.shape == (48, 64)
    for _ in range(5):
        assert abs(game.make_action(0) + 2 * 0.5) < 1e-12
    assert game.is_episode_finished()
    assert game.get_state() is None
    assert game.get_episode_time() == 5


def test_scenarios() -> None:
    game = Game()
    with pytest.raises(ValueError, match="nope-v0"):
        game.set_scenario("nope-v0")
    game.set_scenario("time-trial-v0")
    game.set_screen_resolution(64, 48)
    game.set_reward_config(w_progress=0, w_collision=0, w_wrongway=0, w_lap=0)
    game.init()
    assert abs(game.make_action(0) + 4 * 0.05) < 1e-12
    for name in ("kidney-v0", "technical-v0"):
        other = Game()
        other.set_scenario(name)
        other.set_screen_resolution(64, 48)
        other.init()
        assert _state(other).game_variables[0] == 0


def test_set_appearance_validation_API21() -> None:
    with pytest.raises(ValueError, match="split"):
        Game().set_appearance(True, split="val")
    for strength in (1.5, -0.1):
        with pytest.raises(ValueError, match="strength"):
            Game().set_appearance(True, strength=strength)


def test_set_appearance_load_config_API21(tmp_path: Path) -> None:
    path = tmp_path / "dr.toml"
    path.write_text('[appearance]\nrandomize = true\nsplit = "test"\n', encoding="utf-8")
    game = Game()
    game.load_config(path)
    game.set_seed(5)
    game.init()
    assert game._appearance == palette.draw_appearance(5, "test", 1.0)


def test_set_depth_noise_validation_API24() -> None:
    for kwargs, name in (({"sigma_rel": -0.1}, "sigma_rel"), ({"dropout": 1.5}, "dropout")):
        with pytest.raises(ValueError, match=name):
            Game().set_depth_noise(True, **kwargs)
    for quant in (0.0, -0.05):
        with pytest.raises(ValueError, match="quant"):
            Game().set_depth_noise(True, quant=quant)
    Game().set_depth_noise(True, sigma_rel=0.0, dropout=0.0)
    Game().set_depth_noise(True, dropout=1.0)


def test_set_depth_noise_load_config_API24(tmp_path: Path) -> None:
    path = tmp_path / "noise.toml"
    path.write_text("[depth_noise]\nenabled = true\nquant = 0.1\n", encoding="utf-8")
    game = Game()
    assert game._depth_noise == (False, 0.02, 0.01, 0.05)
    game.load_config(path)
    assert game._depth_noise == (True, 0.02, 0.01, 0.1)
    path.write_text("[depth_noise]\ndropout = 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dropout"):
        Game().load_config(path)


Frame = tuple[GameState, GameState, float, float, dict[str, float], dict[str, float]]


def _noise_pair(seed: int = 0, steps: int = 20, **noise: float) -> list[Frame]:
    """Per-frame (noisy, clean) states, rewards and components on technical-v0 at 96x72."""
    games = []
    for enabled in (True, False):
        game = Game()
        game.set_scenario("technical-v0")
        game.set_depth_buffer_enabled(True)
        game.set_labels_buffer_enabled(True)
        game.set_depth_noise(enabled, **noise)
        game.init()
        game.new_episode(seed)
        games.append(game)
    noisy, clean = games
    frames = [(_state(noisy), _state(clean), 0.0, 0.0, {}, {})]
    for a in np.random.default_rng(seed).integers(len(DISCRETE_ACTIONS), size=steps):
        ra, rb = noisy.make_action(int(a)), clean.make_action(int(a))
        frames.append(
            (
                _state(noisy),
                _state(clean),
                ra,
                rb,
                noisy.get_last_reward_components(),
                clean.get_last_reward_components(),
            )
        )
    return frames


def _near(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray]:
    """Noisy and true depth of all pixels whose true depth is below 0.8 * max_view_distance."""
    noisy = np.stack([f[0].depth_buffer for f in frames])
    clean = np.stack([f[1].depth_buffer for f in frames])
    mask = clean < 0.8 * 60.0
    return noisy[mask].astype(np.float64), clean[mask].astype(np.float64)


def test_depth_noise_relative_error_API24() -> None:
    noisy, clean = _near(_noise_pair(sigma_rel=0.05, quant=0.001, dropout=0.0))
    rel = (noisy - clean) / clean
    assert abs(np.sqrt(np.mean(rel**2)) - 0.05) <= 0.005
    assert abs(rel.mean()) < 0.005


def test_depth_noise_dropout_rate_API24() -> None:
    noisy, _ = _near(_noise_pair(sigma_rel=0.0, quant=0.001, dropout=0.05))
    assert abs(np.mean(noisy == 60.0) - 0.05) <= 0.01
    noisy, _ = _near(_noise_pair(sigma_rel=0.0, quant=0.001, dropout=0.0))
    assert not np.any(noisy == 60.0)


def test_depth_noise_quantized_and_valid_API24() -> None:
    for noisy, *_ in _noise_pair():
        depth = noisy.depth_buffer
        assert depth is not None
        assert depth.dtype == np.float32
        assert np.all(np.isfinite(depth))
        assert depth.min() >= 0.05
        assert depth.max() <= 60.0
        q = depth.astype(np.float64) / 0.05
        assert np.allclose(q, np.round(q), rtol=0, atol=1e-4 / 0.05)


def test_depth_noise_determinism_API24() -> None:
    a, b = _noise_pair(seed=3), _noise_pair(seed=3)
    for fa, fb in zip(a, b, strict=True):
        assert np.array_equal(fa[0].depth_buffer, fb[0].depth_buffer)
    other = _noise_pair(seed=4)
    assert not np.array_equal(a[0][0].depth_buffer, other[0][0].depth_buffer)
    game = _game(buffers=(True, True, False), set_depth_noise=True)
    game.make_action(0)
    assert np.array_equal(_state(game).depth_buffer, _state(game).depth_buffer)


def test_depth_noise_isolation_API24() -> None:
    for noisy, clean, ra, rb, ca, cb in _noise_pair(steps=40):
        assert np.array_equal(noisy.screen_buffer, clean.screen_buffer)
        assert np.array_equal(noisy.labels_buffer, clean.labels_buffer)
        assert np.array_equal(noisy.game_variables, clean.game_variables)
        assert noisy.number == clean.number
        assert ra == rb
        assert ca == cb
        assert not np.array_equal(noisy.depth_buffer, clean.depth_buffer)
