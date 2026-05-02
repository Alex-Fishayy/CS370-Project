"""Head pose estimation using cv2.solvePnP on 6 facial landmarks.

Returns yaw, pitch, roll in degrees. Yaw = left/right head turn,
pitch = up/down nod, roll = head tilt to shoulder.
"""
import cv2
import numpy as np

from .detection import POSE_3D_MODEL


def estimate_head_pose(pose_2d: np.ndarray, frame_shape: tuple) -> tuple[float, float, float]:
    """
    pose_2d: (6, 2) array of image points corresponding to POSE_3D_MODEL.
    frame_shape: (H, W, ...) for camera matrix approximation.
    Returns (yaw, pitch, roll) in degrees.
    """
    h, w = frame_shape[:2]
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion for prototype

    success, rvec, tvec = cv2.solvePnP(
        POSE_3D_MODEL,
        pose_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to Euler angles.
    rmat, _ = cv2.Rodrigues(rvec)
    # Use RQ decomposition (cv2.decomposeProjectionMatrix expects 3x4)
    proj = np.hstack([rmat, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)

    pitch, yaw, roll = euler.flatten()[:3]

    # Normalize pitch to [-90, 90] range; cv2 sometimes returns around 180.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return float(yaw), float(pitch), float(roll)
