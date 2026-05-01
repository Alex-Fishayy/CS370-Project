"""Eye Aspect Ratio (EAR) for eye open/closed detection.

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
Low EAR = eye closed. Standard threshold ~0.20-0.25.

Reference: Soukupova & Cech, "Real-Time Eye Blink Detection Using Facial Landmarks"
"""
import numpy as np


def compute_ear(eye_pts: np.ndarray) -> float:
    """eye_pts: (6, 2) array ordered [outer, top1, top2, inner, bot1, bot2]."""
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


def eyes_open(left_eye: np.ndarray, right_eye: np.ndarray, threshold: float) -> tuple[bool, float]:
    """Returns (is_open, mean_ear)."""
    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)
    mean = (left_ear + right_ear) / 2.0
    return mean >= threshold, mean
