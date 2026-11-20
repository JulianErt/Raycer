import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from raycer.track import Track, load_builtin

HINT = r"raycer\[plot\]"


def _track() -> Track:
    return load_builtin("oval")


def _lap(track: Track) -> np.ndarray:
    """A full lap along the centerline, ending just past the start."""
    c = track.center[[*range(len(track.center)), 0, 1]]
    d = np.roll(c, -1, axis=0) - c
    return np.column_stack([c, np.arctan2(d[:, 1], d[:, 0])])


def _paths(track: Track) -> dict[str, np.ndarray]:
    lap = _lap(track)
    return {"lap": lap, "short": lap[:7]}


def test_png_pdf_magic_API26(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from raycer.trajectory import plot_trajectories

    track = _track()
    plot_trajectories(track, _paths(track), tmp_path / "t.png")
    plot_trajectories(track, _paths(track), str(tmp_path / "t.pdf"))
    assert (tmp_path / "t.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "t.pdf").read_bytes().startswith(b"%PDF")


def test_gif_frames_API26(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from PIL import Image

    from raycer.trajectory import animate_trajectories

    track = _track()
    lap = _lap(track)
    animate_trajectories(track, {"a": lap[:12], "b": lap[:5]}, tmp_path / "t.gif", fps=10)
    data = (tmp_path / "t.gif").read_bytes()
    assert data.startswith(b"GIF89a")
    with Image.open(tmp_path / "t.gif") as img:
        assert img.n_frames == 12


def test_finish_rule_and_marker_API26() -> None:
    pytest.importorskip("matplotlib")
    from raycer.trajectory import _finished, _plot_figure

    track = _track()
    paths = _paths(track)
    assert _finished(track, paths["lap"])
    assert not _finished(track, paths["short"])
    ax = _plot_figure(track, paths).axes[0]
    marked = [ln for ln in ax.get_lines() if ln.get_marker() == "x"]
    assert len(marked) == 1
    np.testing.assert_allclose(marked[0].get_xydata()[-1], paths["short"][-1, :2])
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["lap", "short"]


def test_errors_API26(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from raycer.trajectory import animate_trajectories, plot_trajectories

    track = _track()
    good = {"a": _lap(track)[:3]}
    with pytest.raises(ValueError, match="suffix"):
        plot_trajectories(track, good, tmp_path / "t.jpg")
    with pytest.raises(ValueError, match="suffix"):
        animate_trajectories(track, good, tmp_path / "t.png")
    for bad in (np.zeros((4, 2)), np.zeros((0, 3)), np.zeros(3)):
        with pytest.raises(ValueError, match="shape"):
            plot_trajectories(track, {"a": bad}, tmp_path / "t.png")


def test_missing_matplotlib_hint_API26(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "raycer.trajectory", raising=False)
    with pytest.raises(ImportError, match=HINT):
        importlib.import_module("raycer.trajectory")
    code = "import sys; sys.modules['matplotlib'] = None; import raycer"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
