"""Top-down trajectory plot and GIF of car poses with matplotlib (API-26).

A path counts as finished when replaying its poses through `ProgressTracker` gives lap >= 1;
an unfinished path ends with an x marker at its last pose.
"""

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from raycer.progress import ProgressTracker
from raycer.track import Track

INSTALL_HINT = 'matplotlib is not installed; install it with: uv add "raycer[plot]"'

try:
    from matplotlib import colormaps
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
except ImportError as e:
    raise ImportError(INSTALL_HINT) from e

Pose = NDArray[np.float64]


def _poses(paths: Mapping[str, ArrayLike]) -> dict[str, Pose]:
    """Convert every path to a float array of shape (T, 3) with T >= 1."""
    out = {}
    for label, path in paths.items():
        arr = np.asarray(path, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 3:
            raise ValueError(f"path {label!r} must have shape (T, 3), got {arr.shape}")
        out[label] = arr
    return out


def _check_suffix(out: str | Path, allowed: tuple[str, ...]) -> Path:
    """Return out as a Path; raise ValueError if its suffix is not allowed."""
    path = Path(out)
    if path.suffix.lower() not in allowed:
        raise ValueError(f"{path.name}: suffix must be one of {allowed}")
    return path


def _finished(track: Track, poses: Pose) -> bool:
    """Replay poses through ProgressTracker and return whether lap >= 1."""
    tracker = ProgressTracker(track)
    state = tracker.reset(*poses[0])
    for x, y, heading in poses[1:]:
        state = tracker.update(x, y, heading, 0.0, False)
    return state.lap >= 1


def _background(track: Track, n: int) -> tuple[Figure, Axes, list[tuple[float, ...]]]:
    """Draw walls and finish quad; return figure, axes and n viridis colours."""
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.set_axis_off()
    segments = np.stack([track.seg_a, track.seg_b], axis=1)
    ax.add_collection(LineCollection(list(segments), colors="grey", linewidths=1))
    ax.add_patch(Polygon(track.finish_quad, closed=True, color=(1.0, 220 / 255, 0.0)))
    ax.autoscale_view()
    cmap = colormaps["viridis"]
    colors = [tuple(cmap(i / max(n - 1, 1))) for i in range(n)]
    return fig, ax, colors


def _plot_figure(track: Track, paths: Mapping[str, ArrayLike]) -> Figure:
    """Build the static trajectory figure with lines, x markers and a legend."""
    poses = _poses(paths)
    fig, ax, colors = _background(track, len(poses))
    for (label, p), color in zip(poses.items(), colors, strict=True):
        ax.plot(p[:, 0], p[:, 1], color=color, label=label)
        if not _finished(track, p):
            ax.plot(p[-1, 0], p[-1, 1], marker="x", color=color, markersize=10, mew=2)
    ax.legend(loc="upper right", fontsize="small")
    return fig


def plot_trajectories(track: Track, paths: Mapping[str, ArrayLike], out: str | Path) -> None:
    """Write a top-down plot of labelled (T, 3) pose paths to a .png or .pdf file."""
    path = _check_suffix(out, (".png", ".pdf"))
    _plot_figure(track, paths).savefig(path)


def _arrow(pose: Pose, r: float) -> NDArray[np.float64]:
    """Return the three corners of a car arrow at pose (x, y, heading) pointing along heading."""
    x, y, heading = (float(v) for v in pose)
    c, s = math.cos(heading), math.sin(heading)
    return np.array(
        [
            (x + 1.5 * r * c, y + 1.5 * r * s),
            (x - r * c - r * s, y - r * s + r * c),
            (x - r * c + r * s, y - r * s - r * c),
        ]
    )


def animate_trajectories(
    track: Track, paths: Mapping[str, ArrayLike], out: str | Path, fps: int = 30
) -> None:
    """Write a GIF with one frame per pose of the longest path, each tip drawn as an arrow."""
    path = _check_suffix(out, (".gif",))
    poses = _poses(paths)
    fig, ax, colors = _background(track, len(poses))
    points = np.concatenate([track.seg_a, track.seg_b])
    r = 0.012 * float(max(points.max(axis=0) - points.min(axis=0)))
    lines: list[Line2D] = []
    arrows: list[Polygon] = []
    for (label, p), color in zip(poses.items(), colors, strict=True):
        lines.append(ax.plot(p[:1, 0], p[:1, 1], color=color, label=label)[0])
        arrow = Polygon(_arrow(p[0], r), closed=True, color=color)
        ax.add_patch(arrow)
        arrows.append(arrow)
    ax.legend(loc="upper right", fontsize="small")

    def draw(i: int) -> list[Line2D | Polygon]:
        """Show every path up to pose min(i, T - 1)."""
        for p, line, arrow in zip(poses.values(), lines, arrows, strict=True):
            j = min(i, len(p) - 1)
            line.set_data(p[: j + 1, 0], p[: j + 1, 1])
            arrow.set_xy(_arrow(p[j], r))
        return [*lines, *arrows]

    frames = max(len(p) for p in poses.values())
    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps))
