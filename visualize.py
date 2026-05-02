"""Debug overlay for annotated output video."""
import cv2
import numpy as np


GREEN = (0, 200, 0)
RED = (0, 0, 220)
YELLOW = (0, 200, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_face(frame, track_id, bbox, result):
    x1, y1, x2, y2 = bbox
    color = GREEN if result.attentive else RED
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"#{track_id} {'OK' if result.attentive else 'INATTN'}"
    sub = f"y{result.yaw:+.0f} p{result.pitch:+.0f} ear{result.ear:.2f}"

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, sub, (x1, y2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if not result.attentive:
        cv2.putText(frame, result.reason, (x1, y2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1, cv2.LINE_AA)


def draw_phones(frame, phones):
    for x1, y1, x2, y2, conf in phones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)
        cv2.putText(frame, f"phone {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1, cv2.LINE_AA)


def draw_hud(frame, frame_idx, n_faces, n_attentive, fps):
    h, w = frame.shape[:2]
    text = f"Frame {frame_idx}  Faces {n_faces}  Attentive {n_attentive}/{n_faces}  {fps:.1f} FPS"
    cv2.rectangle(frame, (0, 0), (w, 24), BLACK, -1)
    cv2.putText(frame, text, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
