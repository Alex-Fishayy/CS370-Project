"""Face detection + landmarks via MediaPipe Face Mesh.

Returns face bboxes and the landmark points we need downstream:
- 6 points for solvePnP head pose
- 6 points per eye for EAR
"""
from dataclasses import dataclass
import cv2
import mediapipe as mp
import numpy as np

from . import config

# MediaPipe Face Mesh landmark indices we care about.
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj

# For head pose (solvePnP) - 6 stable anatomical points:
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

# 3D model points (in mm) for the above, canonical face coords.
POSE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # nose tip
    [0.0, -63.6, -12.5],      # chin
    [-43.3, 32.7, -26.0],     # left eye outer corner
    [43.3, 32.7, -26.0],      # right eye outer corner
    [-28.9, -28.9, -24.1],    # mouth left
    [28.9, -28.9, -24.1],     # mouth right
], dtype=np.float64)

# Eye landmarks for EAR calculation.
# Using MediaPipe's 6-point eye convention: [p1, p2, p3, p4, p5, p6]
# where p1=outer corner, p4=inner corner, and p2,p3,p5,p6 are upper/lower lid points.
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [263, 387, 385, 362, 380, 373]

# Iris landmarks (available with refine_landmarks=True, indices 468-477).
# Index 0 of each group is the iris center; indices 1-4 are boundary points.
IRIS_LEFT_INDICES = [468, 469, 470, 471, 472]
IRIS_RIGHT_INDICES = [473, 474, 475, 476, 477]

# Eye corner landmarks used for gaze horizontal ratio.
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_EYE_TOP = 159
LEFT_EYE_BOT = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOT = 374


@dataclass
class FaceResult:
    bbox: tuple              # (x1, y1, x2, y2) in image coords
    landmarks: np.ndarray    # (478, 2) pixel coords of all landmarks (468 face + 10 iris)
    pose_2d: np.ndarray      # (6, 2) pixel coords of the 6 pose points
    left_eye: np.ndarray     # (6, 2) pixel coords for EAR
    right_eye: np.ndarray    # (6, 2) pixel coords for EAR
    iris_left: np.ndarray    # (5, 2) left iris: [0]=center, [1-4]=boundary
    iris_right: np.ndarray   # (5, 2) right iris: [0]=center, [1-4]=boundary
    detection_conf: float    # from bbox area heuristic


class FaceDetector:
    def __init__(self, max_faces: int = config.MAX_FACES):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,   # enables iris landmarks (indices 468-477)
            min_detection_confidence=config.MIN_DETECTION_CONF,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return []

        faces = []
        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized landmarks to pixel coords
            lms = np.array(
                [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark],
                dtype=np.float32,
            )

            # Bbox from landmark extent
            x1, y1 = lms.min(axis=0)
            x2, y2 = lms.max(axis=0)
            bbox = (int(x1), int(y1), int(x2), int(y2))

            pose_2d = np.array([
                lms[POSE_LANDMARKS["nose_tip"]],
                lms[POSE_LANDMARKS["chin"]],
                lms[POSE_LANDMARKS["left_eye_outer"]],
                lms[POSE_LANDMARKS["right_eye_outer"]],
                lms[POSE_LANDMARKS["mouth_left"]],
                lms[POSE_LANDMARKS["mouth_right"]],
            ], dtype=np.float64)

            left_eye = lms[LEFT_EYE_EAR_IDX]
            right_eye = lms[RIGHT_EYE_EAR_IDX]

            # Iris landmarks (requires refine_landmarks=True)
            iris_left = lms[IRIS_LEFT_INDICES] if len(lms) > 468 else np.zeros((5, 2), dtype=np.float32)
            iris_right = lms[IRIS_RIGHT_INDICES] if len(lms) > 473 else np.zeros((5, 2), dtype=np.float32)

            # Cheap confidence proxy: face size relative to frame
            face_w = x2 - x1
            face_h = y2 - y1
            conf = min(1.0, (face_w * face_h) / (w * h * 0.02))  # heuristic

            faces.append(FaceResult(
                bbox=bbox,
                landmarks=lms,
                pose_2d=pose_2d,
                left_eye=left_eye,
                right_eye=right_eye,
                iris_left=iris_left,
                iris_right=iris_right,
                detection_conf=conf,
            ))

        return faces

    def close(self):
        self.mesh.close()


def quality_ok(face: FaceResult, frame: np.ndarray) -> tuple[bool, str]:
    """Quality gate: returns (ok, reason_if_not)."""
    x1, y1, x2, y2 = face.bbox
    w = x2 - x1
    h = y2 - y1
    short = min(w, h)

    if short < config.MIN_FACE_SIZE_PX:
        return False, f"too_small({short}px)"

    # Clamp to frame bounds
    H, W = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return False, "empty_crop"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < config.MIN_BLUR_LAPLACIAN:
        return False, f"blurry({lap_var:.1f})"

    return True, ""
