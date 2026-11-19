import numpy as np

from raycer import palette


def test_style_02_values() -> None:
    assert palette.SKY_TOP == (90, 140, 200)
    assert palette.SKY_HORIZON == (170, 200, 230)
    assert palette.FOG == palette.SKY_HORIZON
    assert palette.ROAD_A == (70, 70, 75)
    assert palette.ROAD_B == (82, 82, 88)
    assert palette.LEFT_A == (200, 40, 40)
    assert palette.LEFT_B == (235, 235, 235)
    assert palette.RIGHT_A == (40, 90, 200)
    assert palette.RIGHT_B == (235, 235, 235)
    assert palette.FINISH_A == (20, 20, 20)
    assert palette.FINISH_B == (240, 240, 240)
    assert palette.OBSTACLE_A == (240, 170, 30)
    assert palette.OBSTACLE_B == (30, 30, 30)


def test_style_03_05_values() -> None:
    assert palette.SKYLINE_FAR == (130, 154, 180)
    assert palette.SKYLINE_NEAR == (90, 108, 130)
    assert palette.CENTER_LINE == (235, 235, 235)
    far, near = np.array(palette.SKYLINE_FAR), np.array(palette.SKYLINE_NEAR)
    assert np.all(near < far)
    assert np.all(far < palette.SKY_HORIZON)


def test_label_colors() -> None:
    colors = palette.LABEL_COLORS
    assert colors.shape == (6, 3)
    assert colors.dtype == np.uint8
    expected = [
        palette.SKY_TOP,
        palette.ROAD_A,
        palette.LEFT_A,
        palette.RIGHT_A,
        palette.FINISH_B,
        palette.OBSTACLE_A,
    ]
    assert [tuple(int(v) for v in row) for row in colors] == expected
    assert len(np.unique(colors, axis=0)) == 6


def test_draw_appearance_ranges_API21() -> None:
    for seed in range(1000):
        train = palette.draw_appearance(seed, "train", 1.0)
        test = palette.draw_appearance(seed, "test", 1.0)
        assert 0.0 <= train["hue_offset_deg"] < 270.0
        assert 270.0 <= test["hue_offset_deg"] < 360.0
        for p in (train, test):
            assert 0.7 <= p["brightness"] <= 1.3
            assert 0.5 <= p["saturation"] <= 1.2
            assert 0.75 <= p["period_scale"] <= 1.5


def test_draw_appearance_determinism_and_strength_API21() -> None:
    p = palette.draw_appearance(3, "train", 0.7)
    assert p == palette.draw_appearance(3, "train", 0.7)
    assert p != palette.draw_appearance(4, "train", 0.7)
    assert len(p) == 11
    assert all(type(v) is float for v in p.values())
    zero = palette.draw_appearance(3, "test", 0.0)
    swaps = {f"swap_{m}": 0.0 for m in ("left", "right", "obstacle", "road", "finish")}
    assert zero == {
        "hue_offset_deg": 0.0,
        "brightness": 1.0,
        "saturation": 1.0,
        "sky_hue_deg": 0.0,
        **swaps,
        "period_scale": 1.0,
        "skyline_phase": 0.0,
    }


def test_appearance_colors_API21() -> None:
    assert palette.appearance_colors(None)["LEFT_A"] == palette.LEFT_A
    params = dict(palette.draw_appearance(0, "train", 0.0), swap_left=1.0)
    colors = palette.appearance_colors(params)
    assert (colors["LEFT_A"], colors["LEFT_B"]) == (palette.LEFT_B, palette.LEFT_A)
    assert colors["ROAD_A"] == palette.ROAD_A
    params = dict(params, swap_left=0.0, hue_offset_deg=120.0, sky_hue_deg=30.0)
    colors = palette.appearance_colors(params)
    assert colors["LEFT_A"] == (40, 200, 40)  # red rotated by 120 degrees is green
    assert colors["FOG"] == colors["SKY_HORIZON"] != palette.SKY_HORIZON
    assert palette.LEFT_A == (200, 40, 40)
    assert palette.LABEL_COLORS[2].tolist() == [200, 40, 40]
