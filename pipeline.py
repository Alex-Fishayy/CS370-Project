"""Orchestrates per-frame processing: detection -> quality -> tracking -> analysis -> log."""
from dataclasses import dataclass

from . import config
from .detection import FaceDetector, quality_ok
from .head_pose import estimate_head_pose
from .eye_state import eyes_open
from .tracking import IoUTracker
from .phone import PhoneDetector, phone_near_face
from .scoring import score


@dataclass
class FrameOutput:
    frame_idx: int
    faces: list         # list of (track_id, bbox, result)
    rejected: list      # list of (bbox, reason) for quality-rejected faces
    phones: list        # list of (x1,y1,x2,y2,conf)


class Pipeline:
    def __init__(self, camera_id: int, logger=None):
        self.camera_id = camera_id
        self.logger = logger
        self.detector = FaceDetector()
        self.tracker = IoUTracker()
        self.phone_det = PhoneDetector()
        self.frame_idx = 0
        self._last_phones: list = []

    def process(self, frame) -> FrameOutput:
        self.frame_idx += 1

        # Phone detection (subsampled)
        if self.frame_idx % config.PHONE_DETECT_EVERY_N_FRAMES == 0:
            self._last_phones = self.phone_det.detect(frame)
        phones = self._last_phones

        # Face detection + quality gate
        faces = self.detector.detect(frame)
        good_faces = []
        rejected = []
        for f in faces:
            ok, reason = quality_ok(f, frame)
            if ok:
                good_faces.append(f)
            else:
                rejected.append((f.bbox, reason))

        # Tracking on good faces only
        detections = [f.bbox for f in good_faces]
        assigned = self.tracker.update(detections)

        # Map tracker bboxes back to face objects by IoU (tracker may have adjusted bbox slightly)
        # Simpler: since assigned returns tracks that were updated this frame,
        # and we passed in our detections in order, we rematch by IoU.
        results = []
        for track_id, tbox in assigned:
            # find best matching face from good_faces
            best_f = None
            best_iou = 0
            for f in good_faces:
                from .tracking import iou as iou_fn
                v = iou_fn(f.bbox, tbox)
                if v > best_iou:
                    best_iou = v
                    best_f = f
            if best_f is None:
                continue

            yaw, pitch, roll = estimate_head_pose(best_f.pose_2d, frame.shape)
            eo_flag, ear = eyes_open(best_f.left_eye, best_f.right_eye,
                                     config.EAR_THRESHOLD)
            phone_flag = phone_near_face(best_f.bbox, phones, config.PHONE_PROXIMITY_MULT)

            result = score(yaw, pitch, eo_flag, ear, phone_flag)
            results.append((track_id, best_f.bbox, result))

            if self.logger is not None:
                self.logger.log(self.camera_id, track_id, self.frame_idx,
                                best_f.bbox, result)

        return FrameOutput(
            frame_idx=self.frame_idx,
            faces=results,
            rejected=rejected,
            phones=phones,
        )

    def close(self):
        self.detector.close()
