"""Simple IoU + Kalman tracker for assigning stable IDs to faces across frames.

No appearance features, no re-ID across cameras. Good enough for seated students.
"""
from dataclasses import dataclass, field
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from . import config


def iou(a: tuple, b: tuple) -> float:
    """IoU of two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _make_kf() -> KalmanFilter:
    """Kalman filter on bbox center + size; constant velocity model."""
    kf = KalmanFilter(dim_x=7, dim_z=4)
    # state: [cx, cy, s (area), r (aspect), vx, vy, vs]
    kf.F = np.array([
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=float)
    kf.P *= 10.0
    kf.R[2:, 2:] *= 10.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    return kf


def _bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    s = w * h
    r = w / h if h > 0 else 1.0
    return np.array([cx, cy, s, r])


def _z_to_bbox(z):
    cx, cy, s, r = z[:4]
    w = np.sqrt(max(s * r, 1e-6))
    h = s / w if w > 0 else 1
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


@dataclass
class Track:
    id: int
    kf: KalmanFilter
    hits: int = 0
    missed: int = 0
    age: int = 0
    last_bbox: tuple = (0, 0, 0, 0)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.missed += 1
        self.last_bbox = _z_to_bbox(self.kf.x.flatten())
        return self.last_bbox

    def update(self, bbox):
        self.missed = 0
        self.hits += 1
        self.kf.update(_bbox_to_z(bbox))
        self.last_bbox = bbox


class IoUTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: list[tuple]) -> list[tuple[int, tuple]]:
        """
        detections: list of bboxes (x1,y1,x2,y2)
        returns: list of (track_id, bbox) for current frame's assigned tracks
        """
        # Predict all existing tracks forward
        for t in self.tracks:
            t.predict()

        predicted_boxes = [t.last_bbox for t in self.tracks]

        # Build IoU cost matrix
        if predicted_boxes and detections:
            cost = np.zeros((len(predicted_boxes), len(detections)))
            for i, p in enumerate(predicted_boxes):
                for j, d in enumerate(detections):
                    cost[i, j] = 1.0 - iou(p, d)

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_tracks = set()
            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if 1.0 - cost[r, c] >= config.IOU_MATCH_THRESHOLD:
                    self.tracks[r].update(detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)
            unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
        else:
            unmatched_dets = list(range(len(detections)))

        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            kf = _make_kf()
            kf.x[:4] = _bbox_to_z(detections[j]).reshape(4, 1)
            t = Track(id=self.next_id, kf=kf, last_bbox=detections[j], hits=1)
            self.next_id += 1
            self.tracks.append(t)

        # Drop stale tracks
        self.tracks = [t for t in self.tracks if t.missed <= config.MAX_MISSED_FRAMES]

        # Return currently visible tracks (missed=0 means updated this frame)
        return [(t.id, t.last_bbox) for t in self.tracks if t.missed == 0]
