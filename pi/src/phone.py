"""Phone detection using Ultralytics YOLOv8n.

Returns list of phone bboxes per frame. The pipeline associates them
with nearby faces in scoring.py.
"""
import numpy as np
from ultralytics import YOLO

from . import config


class PhoneDetector:
    def __init__(self, model_name: str = config.YOLO_MODEL):
        # Ultralytics will auto-download yolov8n.pt on first use.
        self.model = YOLO(model_name)

    def detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        """Returns list of (x1, y1, x2, y2, conf) for phones."""
        results = self.model.predict(
            frame_bgr,
            classes=[config.PHONE_CLASS_ID],
            conf=config.PHONE_CONF_THRESHOLD,
            verbose=False,
        )
        phones = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                phones.append((int(x1), int(y1), int(x2), int(y2), conf))
        return phones


def phone_near_face(face_bbox: tuple, phones: list[tuple], mult: float) -> bool:
    """True if any phone bbox is within `mult` * face_height of the face."""
    fx1, fy1, fx2, fy2 = face_bbox
    face_h = fy2 - fy1
    face_cx = (fx1 + fx2) / 2
    face_cy = (fy1 + fy2) / 2

    max_dist = mult * face_h

    for px1, py1, px2, py2, _ in phones:
        phone_cx = (px1 + px2) / 2
        phone_cy = (py1 + py2) / 2
        dist = ((phone_cx - face_cx) ** 2 + (phone_cy - face_cy) ** 2) ** 0.5
        if dist <= max_dist:
            return True
    return False
