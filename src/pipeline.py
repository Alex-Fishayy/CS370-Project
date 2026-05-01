"""Orchestrates per-frame processing: detection -> tracking every frame,
attention analysis (head pose, gaze, eye state, phone) every N frames."""
from dataclasses import dataclass

from . import config
from .detection import FaceDetector, quality_ok
from .head_pose import estimate_head_pose
from .eye_state import eyes_open
from .gaze import estimate_gaze
from .tracking import IoUTracker, iou as iou_fn
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
        self._last_results: dict = {}   # track_id -> AttentionResult

    def process(self, frame) -> FrameOutput:
        self.frame_idx += 1
        do_analysis = (self.frame_idx % config.ATTENTION_ANALYSIS_EVERY_N_FRAMES == 0)

        # --- Always: face detection + quality gate + tracking ---
        faces = self.detector.detect(frame)
        good_faces = []
        rejected = []
        for f in faces:
            ok, reason = quality_ok(f, frame)
            if ok:
                good_faces.append(f)
            else:
                rejected.append((f.bbox, reason))

        detections = [f.bbox for f in good_faces]
        assigned = self.tracker.update(detections)

        # Map each track to its best-matching face object
        track_to_face: dict = {}
        for track_id, tbox in assigned:
            best_f, best_iou = None, 0.0
            for f in good_faces:
                v = iou_fn(f.bbox, tbox)
                if v > best_iou:
                    best_iou = v
                    best_f = f
            if best_f is not None:
                track_to_face[track_id] = (tbox, best_f)

        # --- Every N frames: full attention analysis ---
        if do_analysis:
            # Phone detection (subsampled to analysis frames)
            if self.frame_idx % config.PHONE_DETECT_EVERY_N_FRAMES == 0:
                self._last_phones = self.phone_det.detect(frame)

            for track_id, (tbox, best_f) in track_to_face.items():
                yaw, pitch, roll = estimate_head_pose(best_f.pose_2d, frame.shape)
                eo_flag, ear = eyes_open(best_f.left_eye, best_f.right_eye,
                                         config.EAR_THRESHOLD)
                gaze_x, gaze_y, gaze_ok = estimate_gaze(
                    best_f.iris_left, best_f.iris_right, best_f.landmarks
                )
                phone_flag = phone_near_face(best_f.bbox, self._last_phones,
                                              config.PHONE_PROXIMITY_MULT)

                result = score(yaw, pitch, eo_flag, ear, phone_flag,
                               gaze_ok, gaze_x, gaze_y)
                self._last_results[track_id] = result

                if self.logger is not None:
                    self.logger.log(self.camera_id, track_id, self.frame_idx,
                                    best_f.bbox, result)

        # Build output: current bbox + last known result (skip unanalyzed tracks)
        results = []
        for track_id, (tbox, _) in track_to_face.items():
            if track_id in self._last_results:
                results.append((track_id, tbox, self._last_results[track_id]))

        # Clean up stale track results
        active_ids = set(track_to_face.keys())
        all_tracked_ids = {t.id for t in self.tracker.tracks}
        stale = [tid for tid in list(self._last_results)
                 if tid not in active_ids and tid not in all_tracked_ids]
        for tid in stale:
            del self._last_results[tid]

        return FrameOutput(
            frame_idx=self.frame_idx,
            faces=results,
            rejected=rejected,
            phones=self._last_phones,
        )

    def close(self):
        self.detector.close()
