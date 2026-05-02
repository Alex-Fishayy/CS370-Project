"""Combine head pose, eye state, gaze, and phone proximity into an attention score."""
from dataclasses import dataclass

from . import config


@dataclass
class AttentionResult:
    attentive: bool
    head_pose_ok: bool
    eyes_open: bool
    no_phone: bool
    gaze_ok: bool
    yaw: float
    pitch: float
    roll: float
    ear: float
    gaze_x: float
    gaze_y: float
    reason: str  # human-readable why-not-attentive


def score(yaw: float, pitch: float, eyes_open_flag: bool, ear: float,
          phone_nearby: bool, gaze_ok: bool = True,
          gaze_x: float = 0.0, gaze_y: float = 0.0) -> AttentionResult:
    head_pose_ok = (
        abs(yaw) < config.HEAD_YAW_LIMIT and
        config.HEAD_PITCH_MIN < pitch < config.HEAD_PITCH_MAX
    )
    no_phone = not phone_nearby
    attentive = head_pose_ok and eyes_open_flag and no_phone and gaze_ok

    reasons = []
    if not head_pose_ok:
        reasons.append(f"looking_away(yaw={yaw:.0f},pitch={pitch:.0f})")
    if not eyes_open_flag:
        reasons.append(f"eyes_closed(ear={ear:.2f})")
    if not no_phone:
        reasons.append("phone_nearby")
    if not gaze_ok:
        reasons.append(f"gaze_away({gaze_x:+.2f})")
    reason = "attentive" if attentive else ",".join(reasons)

    return AttentionResult(
        attentive=attentive,
        head_pose_ok=head_pose_ok,
        eyes_open=eyes_open_flag,
        no_phone=no_phone,
        gaze_ok=gaze_ok,
        yaw=yaw,
        pitch=pitch,
        roll=0.0,
        ear=ear,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        reason=reason,
    )
