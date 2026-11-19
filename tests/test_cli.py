import json
import struct
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from raycer.cli import main
from test_png import decode_png, read_chunks


def test_benchmark_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "--steps", "300", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["steps_per_second"] > 0
    for key in (
        "median_ms_per_step",
        "ms_per_render",
        "ms_per_physics_step",
        "platform",
        "python_version",
        "numpy_version",
        "raycer_version",
    ):
        assert key in result
    assert result["scenario"] == "technical-v0"
    assert result["steps"] == 300
    assert result["buffers"] == "screen,depth,labels"


def test_benchmark_text_and_min_sps(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "--steps", "300", "--buffers", "screen", "--min-sps", "1e9"]) == 1
    captured = capsys.readouterr()
    assert "steps_per_second: " in captured.out
    assert "buffers: screen\n" in captured.out
    assert "min-sps" in captured.err


def test_render_sample(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "a" / "b"
    assert main(["render-sample", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    for name, width in [
        ("rgb.png", 320),
        ("depth.png", 320),
        ("labels.png", 320),
        ("combined.png", 968),
    ]:
        raw = (out / name).read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        assert raw[12:16] == b"IHDR"
        assert struct.unpack(">II", raw[16:24]) == (width, 240)
        assert str(out / name) in printed
        assert read_chunks(out / name)[-1][0] == b"IEND"
    sample = np.load(out / "sample.npz")
    assert sample["rgb"].shape == (240, 320, 3)
    assert sample["depth"].shape == (240, 320)
    assert sample["labels"].shape == (240, 320)

    depth = decode_png(out / "depth.png")
    sky = sample["labels"] == 0
    assert sky.any()
    assert (depth[sky] == 0).all()
    assert depth[-1].mean() > depth[120].mean()
    combined = decode_png(out / "combined.png")
    assert (combined[:, 320:324] == 0).all()
    assert (combined[:, 644:648] == 0).all()
    np.testing.assert_array_equal(combined[:, 324:644, 0], depth)


def test_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8
    assert any(
        "proc-train-v0" in line
        and "Raycer/ProcTrain-v0" in line
        and "Raycer/ProcTrain-Continuous-v0" in line
        for line in lines
    )
    assert any("Raycer/TimeTrial-v0" in line for line in lines)
    assert "proc-train-dr-v0  Raycer/ProcTrainDR-v0  Raycer/ProcTrainDR-Continuous-v0" in lines
    assert "proc-test-dr-v0  Raycer/ProcTestDR-v0  Raycer/ProcTestDR-Continuous-v0" in lines


@pytest.mark.parametrize(
    "argv",
    [
        ["benchmark", "--buffers", "foo"],
        ["benchmark", "--steps", "0"],
        ["benchmark", "--scenario", "nope-v0"],
        ["benchmark", "--scenario", "proc-train-v0"],
        ["benchmark", "--bogus"],
        ["render-sample", "--width", "33"],
        ["play", "--replay", "missing.npz"],
        ["play", "--replay", __file__, "--record", "out"],
        ["play", "--scenario", "nope-v0"],
        ["play", "--width", "33"],
        ["play", "--discrete", "--continuous"],
        ["bogus"],
    ],
)
def test_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("cmd", ["benchmark", "render-sample", "list", "play"])
def test_help_exits_0(cmd: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main([cmd, "--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    ("flags", "space"),
    [([], spaces.Box), (["--continuous"], spaces.Box), (["--discrete"], spaces.Discrete)],
)
def test_play_action_space_HUM05(
    flags: list[str], space: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import raycer.human

    envs = []
    monkeypatch.setattr(raycer.human, "play", lambda env, seed: envs.append(env))
    assert main(["play", "--width", "32", "--height", "24", *flags]) == 0
    assert isinstance(envs[0].action_space, space)


def test_play_help_lists_action_flags_HUM05(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["play", "--help"])
    out = capsys.readouterr().out
    assert "--discrete" in out
    assert "--continuous" in out
