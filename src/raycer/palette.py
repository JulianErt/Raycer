"""Color palette (STYLE-02), label colors and per-episode appearance randomization (API-21)."""

import colorsys
import math

import numpy as np

SKY_TOP = (90, 140, 200)
SKY_HORIZON = (170, 200, 230)
FOG = SKY_HORIZON
ROAD_A = (70, 70, 75)
ROAD_B = (82, 82, 88)
LEFT_A = (200, 40, 40)
LEFT_B = (235, 235, 235)
RIGHT_A = (40, 90, 200)
RIGHT_B = (235, 235, 235)
FINISH_A = (20, 20, 20)
FINISH_B = (240, 240, 240)
OBSTACLE_A = (240, 170, 30)
OBSTACLE_B = (30, 30, 30)
# Skyline layers (STYLE-03): SKY_HORIZON towards (70, 85, 105) at t = 0.4 and 0.8.
SKYLINE_FAR = (130, 154, 180)
SKYLINE_NEAR = (90, 108, 130)
CENTER_LINE = (235, 235, 235)

# Row i is the color of label ID i: SKY, ROAD, LEFT_BARRIER, RIGHT_BARRIER, FINISH, OBSTACLE.
LABEL_COLORS = np.array(
    [SKY_TOP, ROAD_A, LEFT_A, RIGHT_A, FINISH_B, OBSTACLE_A],
    dtype=np.uint8,
)

# Hue offset range in degrees per split; train and test are disjoint (API-21).
_HUE_RANGES = {"train": (0.0, 270.0), "test": (270.0, 360.0)}
_SWAPS = ("left", "right", "obstacle", "road", "finish")
_BASE = {
    "SKY_TOP": SKY_TOP,
    "SKY_HORIZON": SKY_HORIZON,
    "ROAD_A": ROAD_A,
    "ROAD_B": ROAD_B,
    "LEFT_A": LEFT_A,
    "LEFT_B": LEFT_B,
    "RIGHT_A": RIGHT_A,
    "RIGHT_B": RIGHT_B,
    "FINISH_A": FINISH_A,
    "FINISH_B": FINISH_B,
    "OBSTACLE_A": OBSTACLE_A,
    "OBSTACLE_B": OBSTACLE_B,
    "SKYLINE_FAR": SKYLINE_FAR,
    "SKYLINE_NEAR": SKYLINE_NEAR,
    "CENTER_LINE": CENTER_LINE,
}


def draw_appearance(seed: int, split: str, strength: float) -> dict[str, float]:
    """Draw the appearance parameters of one episode from PCG64([seed, 0xA77])."""
    rng = np.random.Generator(np.random.PCG64([seed, 0xA77]))

    def scaled(low: float, high: float) -> float:
        """Draw U(low, high) and pull it towards 1 by strength."""
        return 1.0 + strength * (rng.uniform(low, high) - 1.0)

    params = {
        "hue_offset_deg": strength * rng.uniform(*_HUE_RANGES[split]),
        "brightness": scaled(0.7, 1.3),
        "saturation": scaled(0.5, 1.2),
        "sky_hue_deg": strength * rng.uniform(-60.0, 60.0),
    }
    for name in _SWAPS:
        params[f"swap_{name}"] = float(rng.uniform() < 0.5 * strength)
    params["period_scale"] = scaled(0.75, 1.5)
    params["skyline_phase"] = strength * rng.uniform(0.0, 2 * math.pi)
    return params


def _transform(
    color: tuple[int, int, int], hue_deg: float, sat: float, value: float
) -> tuple[int, int, int]:
    """Rotate the hue by hue_deg and scale saturation and value, clipped to [0, 1]."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in color))
    rgb = colorsys.hsv_to_rgb((h + hue_deg / 360) % 1.0, min(s * sat, 1.0), min(v * value, 1.0))
    r, g, b = (round(255 * c) for c in rgb)
    return r, g, b


def appearance_colors(params: dict[str, float] | None) -> dict[str, tuple[int, int, int]]:
    """Return the palette colors by name, randomized by params; None gives the plain palette."""
    colors = dict(_BASE)
    if params is not None:
        hue, sat, value = params["hue_offset_deg"], params["saturation"], params["brightness"]
        for name, color in colors.items():
            sky = name in ("SKY_TOP", "SKY_HORIZON")
            colors[name] = _transform(color, hue + sky * params["sky_hue_deg"], sat, value)
        for name in _SWAPS:
            if params[f"swap_{name}"]:
                a, b = f"{name.upper()}_A", f"{name.upper()}_B"
                colors[a], colors[b] = colors[b], colors[a]
    colors["FOG"] = colors["SKY_HORIZON"]
    return colors
