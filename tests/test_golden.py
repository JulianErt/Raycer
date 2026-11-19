"""QA-35: golden-image regression tests at 64x48."""

from pathlib import Path

import numpy as np
import pytest

from raycer.env import RaycerEnv

GOLDEN = Path(__file__).parent / "golden"
CASES = [
    ("oval-v0", 0, 0),
    ("oval-v0", 0, 60),
    ("kidney-v0", 0, 60),
    ("technical-v0", 0, 0),
    ("technical-v0", 0, 60),
    ("proc-test-v0", 0, 30),
]
RGB_MEAN_ABS = 1.0
RGB_PIXEL_DIFF = 8
RGB_PIXEL_FRACTION = 0.005
DEPTH_RTOL = 1e-4
LABELS_FRACTION = 0.002
Buffers = dict[str, np.ndarray]


def _render(scenario: str, seed: int, steps: int) -> Buffers:
    """Buffers after reset(seed) and `steps` full-throttle steps."""
    env = RaycerEnv(scenario, screen=True, depth=True, labels=True, width=64, height=48)
    env.reset(seed=seed)
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(1)
        assert not terminated
        assert not truncated
    s = env.game.get_state()
    assert s is not None
    assert s.screen_buffer is not None
    assert s.depth_buffer is not None
    assert s.labels_buffer is not None
    return {"rgb": s.screen_buffer, "depth": s.depth_buffer, "labels": s.labels_buffer}


def _mismatches(actual: Buffers, golden: Buffers) -> list[str]:
    """Names of buffers that deviate beyond the QA-35 tolerances."""
    bad = []
    rgb = np.abs(actual["rgb"].astype(np.int16) - golden["rgb"].astype(np.int16))
    if rgb.mean() > RGB_MEAN_ABS or (rgb.max(axis=2) > RGB_PIXEL_DIFF).mean() > RGB_PIXEL_FRACTION:
        bad.append("rgb")
    if not np.allclose(actual["depth"], golden["depth"], rtol=DEPTH_RTOL, atol=0):
        bad.append("depth")
    if (actual["labels"] != golden["labels"]).mean() > LABELS_FRACTION:
        bad.append("labels")
    return bad


@pytest.mark.parametrize(("scenario", "seed", "step"), CASES)
def test_golden_QA35(scenario: str, seed: int, step: int, update_golden: bool) -> None:
    path = GOLDEN / f"{scenario}_{seed}_{step}.npz"
    actual = _render(scenario, seed, step)
    if update_golden:
        GOLDEN.mkdir(exist_ok=True)
        np.savez_compressed(path, **actual)
        return
    if not path.exists():
        pytest.fail(f"missing {path}; run `uv run pytest tests/test_golden.py --update-golden`")
    with np.load(path) as d:
        golden = {k: d[k] for k in d.files}
    assert _mismatches(actual, golden) == []


def test_golden_files_QA35() -> None:
    expected = {f"{sc}_{seed}_{step}.npz" for sc, seed, step in CASES}
    assert {p.name for p in GOLDEN.iterdir()} == expected
    for name in expected:
        path = GOLDEN / name
        assert path.stat().st_size <= 50_000
        with np.load(path) as d:
            assert sorted(d.files) == ["depth", "labels", "rgb"]
            assert (d["rgb"].shape, d["rgb"].dtype) == ((48, 64, 3), np.uint8)
            assert (d["depth"].shape, d["depth"].dtype) == ((48, 64), np.float32)
            assert (d["labels"].shape, d["labels"].dtype) == ((48, 64), np.uint8)


def test_tolerances_detect_changes_QA35() -> None:
    ref = _render("oval-v0", 0, 60)

    def perturbed(**changes: np.ndarray) -> Buffers:
        return {k: changes.get(k, v.copy()) for k, v in ref.items()}

    flat = np.arange(48 * 64)
    depth = ref["depth"].copy()
    depth[24, 32] *= 1.01
    assert _mismatches(perturbed(depth=depth), ref) == ["depth"]

    def rgb_shift(n: int) -> np.ndarray:
        rgb = ref["rgb"].astype(np.int16)
        idx = np.unravel_index(flat[:: 3072 // n][:n], (48, 64))
        rgb[idx] = np.where(rgb[idx] < 128, rgb[idx] + 20, rgb[idx] - 20)
        return rgb.astype(np.uint8)

    assert _mismatches(perturbed(rgb=rgb_shift(16)), ref) == ["rgb"]
    assert _mismatches(perturbed(rgb=ref["rgb"] ^ 2), ref) == ["rgb"]  # every value off by 2

    def labels_flip(n: int) -> np.ndarray:
        labels = ref["labels"].copy()
        idx = np.unravel_index(flat[:: 3072 // n][:n], (48, 64))
        labels[idx] = labels[idx] ^ 1
        return labels

    assert _mismatches(perturbed(labels=labels_flip(7)), ref) == ["labels"]
    assert _mismatches(perturbed(), ref) == []
    assert _mismatches(perturbed(rgb=rgb_shift(15), labels=labels_flip(6)), ref) == []
