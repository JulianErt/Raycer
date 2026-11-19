import ast
import json
from pathlib import Path

import pytest

from raycer import render
from raycer.cli import main
from raycer.physics import start_state
from raycer.render import Renderer
from raycer.track import load_builtin

_LOOPS = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


# Meant to run with OMP_NUM_THREADS=1 (set before numpy loads), as in the PERF-01 commands.
@pytest.mark.slow
@pytest.mark.parametrize(
    ("buffers", "width", "height", "min_sps"),
    [
        ("", 96, 72, 5000),
        ("screen", 96, 72, 1200),
        ("screen,depth,labels", 96, 72, 1000),
        ("screen,depth,labels", 160, 120, 500),
    ],
)
def test_perf01_steps_per_second(
    capsys: pytest.CaptureFixture[str], buffers: str, width: int, height: int, min_sps: int
) -> None:
    args = ["benchmark", "--scenario", "technical-v0", "--steps", "2000", "--seed", "0"]
    args += ["--width", str(width), "--height", str(height), "--buffers", buffers, "--json"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["steps_per_second"] >= min_sps


def test_perf03_no_python_loops_in_renderer() -> None:
    tree = ast.parse(Path(render.__file__).read_text(encoding="utf-8"))
    scopes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        or (isinstance(node, ast.ClassDef) and node.name == "Renderer")
    ]
    assert any(isinstance(s, ast.ClassDef) for s in scopes)
    loops = [n for s in scopes for n in ast.walk(s) if isinstance(n, _LOOPS)]
    assert not loops, [f"line {n.lineno}: {type(n).__name__}" for n in loops]


def test_renderer_off_returns_no_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    track = load_builtin("technical")
    renderer = Renderer(rgb=False, depth=False, labels=False)
    monkeypatch.setattr(renderer, "_cast", None)  # casting rays would raise TypeError
    out = renderer.render(track, start_state(track))
    assert (out.rgb, out.depth, out.labels) == (None, None, None)
