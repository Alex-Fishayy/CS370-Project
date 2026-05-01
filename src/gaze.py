"""Estimate gaze direction from MediaPipe iris landmarks.

Uses iris center position relative to eye corners to compute a normalized
gaze offset: 0.0 = centered, positive = looking right/down, negative = left/up.
"""
import numpy as np

from . import config
from .detection import (
    LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOT,
    RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOT,
)


def estimate_gaze(
    iris_left: np.ndarray,
    iris_right: np.ndarray,
    landmarks: np.ndarray,
) -> tuple[float, float, bool]:
    """
    Estimate horizontal and vertical gaze offset from iris position.

    iris_left / iris_right: (5, 2) arrays; index 0 is the iris center.
    landmarks: (478, 2) full landmark array.

    Returns (gaze_x, gaze_y, gaze_ok) where:
        gaze_x: [-1, 1], negative = looking left,  positive = looking right
        gaze_y: [-1, 1], negative = looking up,    positive = looking down
        gaze_ok: True when gaze is approximately forward (|gaze_x| < GAZE_THRESHOLD)
    """
    l_iris = iris_left[0]   # left iris center (px)
    r_iris = iris_right[0]  # right iris center (px)

    # --- Horizontal gaze ---
    l_outer = landmarks[LEFT_EYE_OUTER]
    l_inner = landmarks[LEFT_EYE_INNER]
    r_inner = landmarks[RIGHT_EYE_INNER]
    r_outer = landmarks[RIGHT_EYE_OUTER]

    l_w = l_inner[0] - l_outer[0]
    r_w = r_outer[0] - r_inner[0]

    l_h_ratio = (l_iris[0] - l_outer[0]) / l_w if l_w > 1 else 0.5
    r_h_ratio = (r_iris[0] - r_inner[0]) / r_w if r_w > 1 else 0.5

    # ratio 0 = far left, 0.5 = center, 1 = far right → scale to [-1, 1]
    gaze_x = float(((l_h_ratio + r_h_ratio) / 2.0 - 0.5) * 2.0)

    # --- Vertical gaze ---
    l_top = landmarks[LEFT_EYE_TOP]
    l_bot = landmarks[LEFT_EYE_BOT]
    r_top = landmarks[RIGHT_EYE_TOP]
    r_bot = landmarks[RIGHT_EYE_BOT]

    l_h = l_bot[1] - l_top[1]
    r_h = r_bot[1] - r_top[1]

    l_v_ratio = (l_iris[1] - l_top[1]) / l_h if l_h > 1 else 0.5
    r_v_ratio = (r_iris[1] - r_top[1]) / r_h if r_h > 1 else 0.5

    gaze_y = float(((l_v_ratio + r_v_ratio) / 2.0 - 0.5) * 2.0)

    gaze_ok = abs(gaze_x) < config.GAZE_THRESHOLD

    return gaze_x, gaze_y, gaze_ok
