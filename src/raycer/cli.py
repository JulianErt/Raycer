"""Command line interface: ``raycer``."""

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import raycer
from raycer import physics
from raycer._png import write_png
from raycer.config import list_scenarios, load_config
from raycer.game import DISCRETE_ACTIONS, Game
from raycer.palette import LABEL_COLORS
from raycer.render import Renderer
from raycer.track import load_builtin

_BUFFERS = ("screen", "depth", "labels")
_WARMUP_STEPS = 200
_MICRO_STEPS = 200


def _buffers(value: str) -> list[str]:
    """Parse a comma list of buffer names."""
    names = [n for n in value.split(",") if n]
    unknown = sorted(set(names) - set(_BUFFERS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown buffers {unknown}, expected {list(_BUFFERS)}")
    return names


def _positive_int(value: str) -> int:
    """Parse an integer >= 1."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="raycer", description=__doc__)
    parser.add_argument("--version", action="version", version=f"raycer {raycer.__version__}")
    sub = parser.add_subparsers(dest="command")

    bench = sub.add_parser("benchmark", help="measure steps per second")
    bench.add_argument("--scenario", default="technical-v0")
    bench.add_argument("--steps", type=_positive_int, default=5000)
    bench.add_argument("--width", type=int, default=96)
    bench.add_argument("--height", type=int, default=72)
    bench.add_argument("--buffers", type=_buffers, default=list(_BUFFERS))
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--json", action="store_true", help="print one JSON object")
    bench.add_argument("--min-sps", type=float, help="exit 1 below this many steps/s")

    sample = sub.add_parser("render-sample", help="write rgb, depth and labels sample images")
    sample.add_argument("--scenario", default="oval-v0")
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--steps", type=_positive_int, default=60)
    sample.add_argument("--width", type=int, default=320)
    sample.add_argument("--height", type=int, default=240)
    sample.add_argument("--out", default="samples")

    sub.add_parser("list", help="list scenarios and their Gym IDs")
    return parser


def _make_game(
    parser: argparse.ArgumentParser, args: argparse.Namespace, buffers: Sequence[str]
) -> Game:
    """Configure and init a Game; invalid scenario or resolution is a usage error."""
    game = Game()
    try:
        game.set_scenario(args.scenario)
        game.set_screen_resolution(args.width, args.height)
        game.set_screen_buffer_enabled("screen" in buffers)
        game.set_depth_buffer_enabled("depth" in buffers)
        game.set_labels_buffer_enabled("labels" in buffers)
        game.set_seed(args.seed)
        game.init()
    except ValueError as e:
        parser.error(str(e))
    return game


def _benchmark(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run the benchmark and print the results."""
    game = _make_game(parser, args, args.buffers)
    rng = np.random.default_rng(args.seed)

    def run(n: int) -> list[float]:
        """Time n random env steps, starting a new episode when one finishes."""
        times = []
        for a in rng.integers(len(DISCRETE_ACTIONS), size=n):
            t0 = time.perf_counter()
            game.make_action(int(a))
            times.append(time.perf_counter() - t0)
            if game.is_episode_finished():
                game.new_episode()
        return times

    run(_WARMUP_STEPS)
    times = run(args.steps)

    track = load_builtin(load_config(args.scenario)["track"]["builtin"])
    renderer = Renderer(
        args.width,
        args.height,
        rgb="screen" in args.buffers,
        depth="depth" in args.buffers,
        labels="labels" in args.buffers,
    )
    car = physics.start_state(track)
    t0 = time.perf_counter()
    for _ in range(_MICRO_STEPS):
        renderer.render(track, car)
    ms_render = (time.perf_counter() - t0) / _MICRO_STEPS * 1000
    params, action = physics.CarParams(), physics.Action(0.0, 1.0, 0.0)
    t0 = time.perf_counter()
    for _ in range(_MICRO_STEPS):
        car, _collided = physics.step(car, action, params, track)
    ms_physics = (time.perf_counter() - t0) / _MICRO_STEPS * 1000

    sps = len(times) / sum(times)
    result: dict[str, Any] = {
        "steps_per_second": sps,
        "median_ms_per_step": statistics.median(times) * 1000,
        "ms_per_render": ms_render,
        "ms_per_physics_step": ms_physics,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "raycer_version": raycer.__version__,
        "scenario": args.scenario,
        "steps": args.steps,
        "width": args.width,
        "height": args.height,
        "buffers": ",".join(args.buffers),
        "seed": args.seed,
    }
    if args.json:
        print(json.dumps(result))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    if args.min_sps is not None and sps < args.min_sps:
        print(f"steps_per_second {sps:.1f} is below --min-sps {args.min_sps}", file=sys.stderr)
        return 1
    return 0


def _render_sample(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Drive straight at full throttle and write the final buffers as PNG and npz."""
    game = _make_game(parser, args, _BUFFERS)
    max_view = Renderer(args.width, args.height).max_view_distance
    for _ in range(args.steps):
        game.make_action(1)
    state = game.get_state()
    assert state is not None
    assert state.screen_buffer is not None
    assert state.depth_buffer is not None
    assert state.labels_buffer is not None
    rgb, depth, labels = state.screen_buffer, state.depth_buffer, state.labels_buffer

    grey = np.round(255 * (1 - depth / max_view)).astype(np.uint8)
    labels_rgb = LABEL_COLORS[labels]
    gap = np.zeros((args.height, 4, 3), np.uint8)
    grey_rgb = np.repeat(grey[:, :, None], 3, axis=2)
    combined = np.hstack([rgb, gap, grey_rgb, gap, labels_rgb])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = {"rgb.png": rgb, "depth.png": grey, "labels.png": labels_rgb, "combined.png": combined}
    for name, image in images.items():
        write_png(out / name, image)
        print(out / name)
    np.savez(out / "sample.npz", rgb=rgb, depth=depth, labels=labels)
    print(out / "sample.npz")
    return 0


def _list() -> int:
    """Print each scenario with its Gym IDs."""
    for name in list_scenarios():
        camel = "".join(part.capitalize() for part in name.removesuffix("-v0").split("-"))
        print(f"{name}  Raycer/{camel}-v0  Raycer/{camel}-Continuous-v0")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        return _benchmark(parser, args)
    if args.command == "render-sample":
        return _render_sample(parser, args)
    if args.command == "list":
        return _list()
    parser.print_help()
    return 0
