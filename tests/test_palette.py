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
