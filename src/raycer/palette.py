"""Color palette (STYLE-02) and label colors."""

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

# Row i is the color of label ID i: SKY, ROAD, LEFT_BARRIER, RIGHT_BARRIER, FINISH, OBSTACLE.
LABEL_COLORS = np.array(
    [SKY_TOP, ROAD_A, LEFT_A, RIGHT_A, FINISH_B, OBSTACLE_A],
    dtype=np.uint8,
)
