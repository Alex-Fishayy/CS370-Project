# Attention Tracker — Laptop Prototype

Classroom attention tracking: counts clearly-visible faces, scores each for attentiveness (head pose + eye state + phone presence), logs per-measurement data to SQLite, and generates a time-bucketed report.

## Setup

```bash
cd attention
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run will auto-download `yolov8n.pt` (~6 MB).

## Run on a video file

```bash
python -m scripts.run_video --source path/to/class.mp4
```

Press **q** in the display window to stop early.

## Other options

```bash
# Two "cameras" from two video files
python -m scripts.run_video --source cam1.mp4 --source2 cam2.mp4

# Live webcam
python -m scripts.run_video --source 0

# Save annotated output + skip display (headless / faster)
python -m scripts.run_video --source class.mp4 --out out.mp4 --no-display

# Quick test on first 300 frames
python -m scripts.run_video --source class.mp4 --max-frames 300
```

## Generate the end-of-class report

```bash
python -m scripts.generate_report --session session_XXXXXX --db attention_log.db
```

The `session_XXXXXX` ID is printed at the start and end of `run_video`.

## Project layout

```
src/
  config.py       # all thresholds/tunables
  detection.py    # MediaPipe Face Mesh wrapper + quality gate
  tracking.py     # IoU + Kalman tracker
  head_pose.py    # solvePnP head pose
  eye_state.py    # Eye Aspect Ratio
  phone.py        # YOLOv8n phone detection
  scoring.py      # attention score combiner
  logger.py       # SQLite writer
  visualize.py    # debug overlay
  pipeline.py     # per-frame orchestration
scripts/
  run_video.py       # main entry point
  generate_report.py # post-class report
```

## Tuning

Edit `src/config.py` — all thresholds are there.

Common things to tune against your test footage:
- `MIN_FACE_SIZE_PX` — lower if faces are far from camera
- `MIN_BLUR_LAPLACIAN` — lower in dim rooms
- `EAR_THRESHOLD` — varies 0.15-0.25 depending on subject
- `HEAD_YAW_LIMIT` / `HEAD_PITCH_*` — tune by watching the overlay

## What this is (and isn't)

This is the laptop prototype for development and threshold tuning. Models used:
- Face detection + landmarks: **MediaPipe Face Mesh**
- Head pose: **solvePnP on 6 landmarks**
- Eye state: **EAR (Eye Aspect Ratio)**
- Phone detection: **YOLOv8n**

When porting to Raspberry Pi, we plan to swap face detection to **RetinaFace-MobileNet (ONNX)**, head pose to **6DRepNet (ONNX)**, and eye state to a small CNN classifier — all for better accuracy/perf on ARM CPU. The pipeline architecture and scoring logic stay identical.
