# Attention Tracker — Raspberry Pi Edition

> **Platform:** Raspberry Pi 4 (4 GB RAM) or Pi 5 · 64-bit Raspberry Pi OS Bookworm (aarch64)  
> **Python:** 3.11 via Miniforge (Conda) · No PyTorch required

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Dependencies](#software-dependencies)
5. [Installation](#installation)
6. [Running the Tracker](#running-the-tracker)
7. [Generating Reports](#generating-reports)
8. [Module Reference](#module-reference)
   - [config.py](#configpy)
   - [detection.py](#detectionpy)
   - [head_pose.py](#head_posepy)
   - [eye_state.py](#eye_statepy)
   - [gaze.py](#gazepy)
   - [phone.py](#phonepy)
   - [tracking.py](#trackingpy)
   - [scoring.py](#scoringpy)
   - [pipeline.py](#pipelinepy)
   - [logger.py](#loggerpy)
   - [visualize.py](#visualizepy)
9. [Configuration Reference](#configuration-reference)
10. [Database Schema](#database-schema)
11. [MJPEG Live Stream](#mjpeg-live-stream)
12. [Performance Notes](#performance-notes)
13. [Troubleshooting](#troubleshooting)

---

## Overview

This is the **Raspberry Pi build** of the CS 370 Attention Tracker. It monitors student attention in a classroom by analysing a live camera feed and tracking four signals per student per frame:

| Signal | What it measures |
|---|---|
| **Head pose** | Is the student's head oriented toward the front of the room? |
| **Eye state** | Are the student's eyes open (Eye Aspect Ratio)? |
| **Gaze direction** | Are the student's irises pointed roughly forward? |
| **Phone proximity** | Is a phone visible and close to the student's face? |

A student is considered **attentive** when all four signals pass simultaneously. Results are logged to a local SQLite database and can be exported as a per-session text report.

The Pi build is intentionally lightweight:

- **No PyTorch / ultralytics** — phone detection uses a 4 MB quantized TFLite model (SSD MobileNet V1 COCO).
- Attention analysis runs **every 15 frames** (~2×/sec at 30 fps) to stay real-time on the Pi's ARM CPU.
- Phone detection runs **every 10 frames** (further subsampled within the analysis window).
- Up to **4 faces** are tracked simultaneously.

---

## System Architecture

```
Camera feed (V4L2 / file)
        │
        ▼
   ┌─────────────┐
   │  FaceDetector│  ← MediaPipe Face Mesh (478 landmarks + iris)
   └──────┬──────┘
          │ FaceResult (bbox, landmarks, eye pts, iris pts)
          ▼
   ┌──────────────┐
   │ Quality Gate │  ← min face size, blur (Laplacian), max yaw
   └──────┬───────┘
          │ good faces only
          ▼
   ┌──────────────┐
   │  IoU Tracker │  ← Kalman filter + Hungarian algorithm
   └──────┬───────┘
          │ track_id → bbox
          ▼
   ┌────────────────────────────────────────────┐
   │        Attention Analysis (every N frames)  │
   │  ┌────────────┐  ┌──────────┐  ┌────────┐ │
   │  │ head_pose  │  │eye_state │  │  gaze  │ │
   │  └────────────┘  └──────────┘  └────────┘ │
   │  ┌────────────────────────────────────────┐│
   │  │       PhoneDetector (TFLite)           ││
   │  └────────────────────────────────────────┘│
   └──────────────────┬─────────────────────────┘
                      │ AttentionResult
                      ▼
               ┌────────────┐
               │  Scoring   │  ← boolean AND of all signals
               └──────┬─────┘
                      │
          ┌───────────┴──────────┐
          ▼                      ▼
   ┌────────────┐        ┌──────────────┐
   │  Logger    │        │  Visualizer  │
   │ (SQLite)   │        │  (OpenCV HUD)│
   └────────────┘        └──────────────┘
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Board | Raspberry Pi 4 (4 GB) | Raspberry Pi 5 |
| OS | 64-bit Raspberry Pi OS Bookworm (aarch64) | Same |
| Camera | USB webcam (UVC) | Pi Camera Module v2/3 (CSI) |
| Storage | 4 GB free | 8 GB free |
| RAM | 4 GB | 4 GB+ |
| Network | Required for first-time setup | — |

---

## Software Dependencies

Managed by `setup.sh`. No manual installation is needed.

| Package | Version | Purpose |
|---|---|---|
| `mediapipe` | ≥ 0.10.9, < 0.11 | Face mesh + iris landmark detection |
| `opencv-python` | ≥ 4.8.0 | Frame capture, image processing, rendering |
| `numpy` | < 2 | Numerical operations |
| `tflite-runtime` | latest | Lightweight TFLite inference (phone detection) |
| `filterpy` | ≥ 1.4.5 | Kalman filter for multi-object tracking |
| `scipy` | ≥ 1.10.0 | Hungarian algorithm for track assignment |

> **Why no PyTorch?** PyTorch wheels for aarch64 are large (~800 MB) and slow to install. The TFLite model (`detect.tflite`, ~4 MB quantized) provides adequate phone detection at a fraction of the footprint.

---

## Installation

### Step 1 — Get the files onto the Pi

**Option A — USB drive**
```bash
cp -r /media/pi/USBDRIVE/pi ~/attention
cd ~/attention
```

**Option B — Git clone**
```bash
git clone https://github.com/Alex-Fishayy/CS370-Project -b jackson-dev
cp -r CS370-Project/pi ~/attention
cd ~/attention
```

### Step 2 — Run the one-time setup script

```bash
bash setup.sh
```

The script performs the following steps automatically:

1. Installs system libraries (`libgl1`, `libopenblas`, `libglib2.0`, etc.) via `apt-get`.
2. Downloads and installs **Miniforge** (pre-compiled Python 3.11 for aarch64, ~85 MB) to `~/miniforge3`.
3. Creates a Conda environment named `attention` with Python 3.11.
4. Installs all Python packages via pip (PyPI only, bypasses piwheels).
5. Downloads the **SSD MobileNet V1 COCO TFLite model** (~4 MB) to `models/detect.tflite`.

**Duration:** approximately 5–10 minutes on a fresh Pi 4. Internet access is required throughout.

### Step 3 — Camera setup (Pi Camera Module only)

USB webcams require no additional setup.

For a **CSI ribbon-cable Pi Camera Module**:

```bash
# Enable via raspi-config (one time)
sudo raspi-config
# Interface Options → Camera → Enable → Reboot

# Load V4L2 driver (once per boot, or add to /etc/rc.local)
sudo modprobe bcm2835-v4l2
```

---

## Running the Tracker

Always activate the Conda environment first:

```bash
source ~/miniforge3/bin/activate attention
cd ~/attention
```

### Basic usage

```bash
# Webcam (index 0)
python run_video.py --source 0

# Video file
python run_video.py --source path/to/recording.mp4

# Second camera
python run_video.py --source 0 --source2 1

# Save annotated output video
python run_video.py --source 0 --out annotated.mp4
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--source` | *required* | Camera index or video file path |
| `--source2` | — | Optional second camera/file |
| `--out` | — | Save annotated video (first source only) |
| `--no-display` | off | Headless mode — no OpenCV window (saves CPU) |
| `--stream` | off | Serve MJPEG stream on port 8080 |
| `--session` | auto UUID | Override the session identifier |
| `--db` | `attention_log.db` | Path to the SQLite database |
| `--max-frames` | unlimited | Stop after N frames (for quick testing) |

### Live controls

| Key | Action |
|---|---|
| `Q` | Quit and flush the database |

---

## Generating Reports

After a session, generate a text report from the logged data:

```bash
python generate_report.py --session session_<id> --db attention_log.db
```

The session ID is printed at startup: `[info] session_id = session_1715000000`.

### Report output

```
============================================================
Session: session_1715000000
Source:  0
Duration: 3612.4s
Total measurements: 7224

Overall attention rate:   83.2%
  Eyes open rate:         91.5%
  Head pose OK rate:      89.7%
  No phone rate:          97.8%

Per-student breakdown (by tracker ID):
  cam  trk     n  attn%  eyes%  pose%  nop%
    0    1   540  88.5%  94.1%  92.2%  99.1%
    0    2   540  76.3%  88.5%  85.0%  96.5%
    ...

Struggling periods (1-minute windows below 60% attention):
  ...
```

### Report flags

| Flag | Default | Description |
|---|---|---|
| `--session` | *required* | Session ID to query |
| `--db` | `attention_log.db` | Path to SQLite database |
| `--bucket-seconds` | `60` | Window size for "struggling periods" |

---

## Module Reference

### `config.py`

Central configuration file. All tunable thresholds and paths are defined here. Edit this file to adjust sensitivity without touching algorithmic code. See the [Configuration Reference](#configuration-reference) section for all values.

---

### `detection.py`

**Class:** `FaceDetector`

Wraps MediaPipe Face Mesh to detect up to `MAX_FACES` faces per frame. Uses `refine_landmarks=True` to enable iris landmarks (indices 468–477).

Returns a list of `FaceResult` dataclasses:

| Field | Type | Description |
|---|---|---|
| `bbox` | `(x1, y1, x2, y2)` | Face bounding box in image pixels |
| `landmarks` | `ndarray (478, 2)` | All 478 landmark pixel coordinates |
| `pose_2d` | `ndarray (6, 2)` | 6-point subset for head-pose solvePnP |
| `left_eye` | `ndarray (6, 2)` | Left eye landmarks for EAR |
| `right_eye` | `ndarray (6, 2)` | Right eye landmarks for EAR |
| `iris_left` | `ndarray (5, 2)` | Left iris center + 4 boundary points |
| `iris_right` | `ndarray (5, 2)` | Right iris center + 4 boundary points |
| `detection_conf` | `float` | Confidence proxy from bounding box area |

**Function:** `quality_ok(face, frame) → (bool, reason)`

Rejects faces that are too small, too blurry (Laplacian variance), or rotated beyond `MAX_HEAD_YAW_FOR_QUALITY` degrees.

---

### `head_pose.py`

**Function:** `estimate_head_pose(pose_2d, frame_shape) → (yaw, pitch, roll)`

Estimates 3-D head orientation in degrees using OpenCV `solvePnP` with a canonical 6-point 3-D face model (nose tip, chin, eye corners, mouth corners).

- **Yaw** — left/right head rotation (± degrees from frontal)
- **Pitch** — up/down nod
- **Roll** — side tilt (not used in attention decision)

Camera intrinsics are approximated from frame width (focal length = frame width in pixels, principal point = frame center). No lens distortion is modeled.

---

### `eye_state.py`

**Function:** `eyes_open(left_eye, right_eye, threshold) → (bool, mean_ear)`

Computes the **Eye Aspect Ratio (EAR)** for each eye using the 6-point landmark formula:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

Both eyes must exceed `EAR_THRESHOLD` (default 0.20) for the eyes-open flag to be `True`. Returns the mean EAR for logging.

*Reference: Soukupová & Čech, "Real-Time Eye Blink Detection Using Facial Landmarks", 2016.*

---

### `gaze.py`

**Function:** `estimate_gaze(iris_left, iris_right, landmarks) → (gaze_x, gaze_y, gaze_ok)`

Computes normalized iris offset relative to the eye corners:

- **`gaze_x`** ∈ [−1, 1] — negative = looking left, positive = looking right
- **`gaze_y`** ∈ [−1, 1] — negative = looking up, positive = looking down
- **`gaze_ok`** — `True` when `|gaze_x| < GAZE_THRESHOLD` (default 0.45)

Horizontal ratio: iris center mapped between outer and inner eye corners, averaged across both eyes, scaled to [−1, 1].

---

### `phone.py`

**Class:** `PhoneDetector`

Runs **SSD MobileNet V1 COCO** (quantized TFLite) to detect cell phones (COCO class 77, 0-indexed: 76). The model is a 4 MB quantized `uint8` model — no GPU or PyTorch required.

Input is resized to the model's expected resolution, converted RGB, and passed through `tflite_runtime.Interpreter`. Output tensors (boxes, classes, scores, count) are read back and filtered by `PHONE_CONF_THRESHOLD` (default 0.35).

**Function:** `phone_near_face(face_bbox, phones, mult) → bool`

Returns `True` if any detected phone bounding box is within `PHONE_PROXIMITY_MULT × face_height` of the face center.

---

### `tracking.py`

**Class:** `IoUTracker`

Maintains stable numeric IDs for faces across frames using:

1. **Kalman filter** (7-state constant-velocity model on bbox center + area + aspect ratio) per track.
2. **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) on an IoU cost matrix for detection-to-track assignment.

Tracks are dropped after `MAX_MISSED_FRAMES` (default 15) consecutive frames without a matching detection.

This is a single-camera tracker — no appearance features, no cross-camera re-identification.

---

### `scoring.py`

**Function:** `score(yaw, pitch, eyes_open, ear, phone_nearby, gaze_ok, gaze_x, gaze_y) → AttentionResult`

Combines all signals into a single boolean:

```
attentive = head_pose_ok AND eyes_open AND no_phone AND gaze_ok
```

where:

- `head_pose_ok`: `|yaw| < HEAD_YAW_LIMIT` AND `HEAD_PITCH_MIN < pitch < HEAD_PITCH_MAX`
- `no_phone`: phone was not detected near the face

Returns an `AttentionResult` dataclass with all intermediate values and a human-readable `reason` string (e.g. `"looking_away(yaw=42,pitch=3),eyes_closed(ear=0.15)"`).

---

### `pipeline.py`

**Class:** `Pipeline`

Orchestrates the full per-frame processing loop for one camera:

1. **Every frame:** face detection → quality gate → IoU tracker update.
2. **Every `ATTENTION_ANALYSIS_EVERY_N_FRAMES` frames (default 15):**
   - Run phone detection (further subsampled to every `PHONE_DETECT_EVERY_N_FRAMES` frames).
   - Run head pose, eye state, gaze, and scoring for each tracked face.
   - Write results to the `AttentionLogger`.
3. Returns a `FrameOutput` with current bboxes, track IDs, results, and quality-rejected faces for the visualizer.

Stale track results are garbage-collected when a track is dropped.

---

### `logger.py`

**Class:** `AttentionLogger`

Writes to an SQLite database. Inserts are batched in groups of 50 for write efficiency. The database schema creates two tables (`sessions`, `measurements`) and two indices on session+time and track ID. See [Database Schema](#database-schema) for details.

---

### `visualize.py`

Provides OpenCV drawing functions used in `run_video.py`:

- `draw_face(frame, track_id, bbox, result)` — colored bbox (green = attentive, red = not attentive) with track ID and reason label.
- `draw_phones(frame, phones)` — yellow bounding boxes for detected phones.
- `draw_hud(frame, faces)` — top-left HUD showing overall session attention rate.

---

## Configuration Reference

All values are in `src/config.py`.

### Detection

| Key | Default | Description |
|---|---|---|
| `MIN_FACE_SIZE_PX` | `48` | Minimum bounding-box short edge (px) |
| `MIN_DETECTION_CONF` | `0.5` | MediaPipe detection confidence threshold |
| `MAX_FACES` | `4` | Maximum simultaneous faces (keep low for Pi) |

### Quality Gate

| Key | Default | Description |
|---|---|---|
| `MIN_BLUR_LAPLACIAN` | `40.0` | Laplacian variance; faces below this are skipped |
| `MAX_HEAD_YAW_FOR_QUALITY` | `60.0°` | Extreme yaw makes landmarks unreliable |

### Attention Thresholds

| Key | Default | Description |
|---|---|---|
| `HEAD_YAW_LIMIT` | `35.0°` | Maximum allowed head yaw |
| `HEAD_PITCH_MIN` | `−20.0°` | Looking down at notes is acceptable |
| `HEAD_PITCH_MAX` | `15.0°` | Head tilted back counts as inattentive |
| `EAR_THRESHOLD` | `0.20` | Mean EAR below this → eyes closed |
| `EAR_CLOSED_FRAMES` | `3` | Consecutive closed-eye frames before flagging |
| `GAZE_THRESHOLD` | `0.45` | Normalized iris offset; beyond this → gaze away |

### Phone Detection

| Key | Default | Description |
|---|---|---|
| `PHONE_CONF_THRESHOLD` | `0.35` | TFLite score to accept a phone detection |
| `PHONE_DETECT_EVERY_N_FRAMES` | `10` | Run phone model every N frames |
| `PHONE_PROXIMITY_MULT` | `2.0` | Phone within N × face height counts as nearby |
| `PHONE_CLASS_ID` | `76` | COCO 0-indexed class for "cell phone" |
| `TFLITE_MODEL` | `models/detect.tflite` | Path to quantized TFLite model |

### Tracking

| Key | Default | Description |
|---|---|---|
| `IOU_MATCH_THRESHOLD` | `0.3` | Minimum IoU to associate a detection to a track |
| `MAX_MISSED_FRAMES` | `15` | Frames before a track is dropped |

### Processing & Logging

| Key | Default | Description |
|---|---|---|
| `PROCESS_EVERY_N_FRAMES` | `1` | Frame skip (increase to trade accuracy for speed) |
| `ATTENTION_ANALYSIS_EVERY_N_FRAMES` | `15` | Attention analysis cadence (~2×/sec at 30 fps) |
| `OUTPUT_FPS` | `30` | FPS for annotated output video |
| `DB_PATH` | `attention_log.db` | Default SQLite database path |

---

## Database Schema

```sql
CREATE TABLE sessions (
    session_id   TEXT PRIMARY KEY,
    started_at   REAL NOT NULL,     -- Unix timestamp
    ended_at     REAL,
    video_source TEXT,
    notes        TEXT
);

CREATE TABLE measurements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    camera_id    INTEGER NOT NULL,
    track_id     INTEGER NOT NULL,
    timestamp    REAL    NOT NULL,  -- Unix timestamp
    frame_idx    INTEGER NOT NULL,
    bbox_x1      INTEGER,
    bbox_y1      INTEGER,
    bbox_x2      INTEGER,
    bbox_y2      INTEGER,
    yaw          REAL,
    pitch        REAL,
    ear          REAL,              -- mean Eye Aspect Ratio
    eyes_open    INTEGER,           -- 1 / 0
    head_pose_ok INTEGER,           -- 1 / 0
    no_phone     INTEGER,           -- 1 / 0
    attentive    INTEGER,           -- 1 / 0
    reason       TEXT               -- human-readable why-not-attentive
);

CREATE INDEX idx_session_time ON measurements(session_id, timestamp);
CREATE INDEX idx_track        ON measurements(session_id, camera_id, track_id);
```

---

## MJPEG Live Stream

Start the tracker with `--stream` to serve a live MJPEG feed on port 8080:

```bash
python run_video.py --source 0 --stream
```

Open in any browser or VLC on a device on the same network:

```
http://<pi-ip-address>:8080
```

The server runs on a background daemon thread and encodes frames as JPEG at quality 70 (~30 ms/frame). The stream continues as long as the main process is running.

---

## Performance Notes

| Scenario | Approximate FPS (Pi 4) |
|---|---|
| Detection + tracking only, no display | ~25–30 fps |
| Full pipeline (4 faces), no display | ~15–20 fps |
| Full pipeline with display window | ~10–15 fps |
| Full pipeline + MJPEG stream | ~10–15 fps |

**Tips for better performance:**
- Use `--no-display` for unattended deployments — removes the OpenCV `imshow` overhead.
- Increase `PROCESS_EVERY_N_FRAMES` to 2 or 3 to halve/third the workload (trade-off: coarser attention timestamps).
- Reduce `MAX_FACES` to 2 if only one or two students are in frame.
- `ATTENTION_ANALYSIS_EVERY_N_FRAMES = 15` is already tuned for the Pi 4; lowering it will increase CPU usage.

---

## Troubleshooting

### Camera not found

```
[error] could not open source: 0
```

- Run `ls /dev/video*` to list available devices.
- For Pi Camera (CSI): run `sudo modprobe bcm2835-v4l2` first.
- Permission denied: `sudo usermod -aG video $USER` then reboot.

### `tflite-runtime` import error

```
ModuleNotFoundError: No module named 'tflite_runtime'
```

The Conda environment may not be active. Run:

```bash
source ~/miniforge3/bin/activate attention
```

### Phone model not found

```
FileNotFoundError: models/detect.tflite
```

Re-run `setup.sh` from the `~/attention` directory, or manually download:

```bash
mkdir -p models
wget "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip" -O models/detect.zip
unzip models/detect.zip detect.tflite -d models/
rm models/detect.zip
```

### MediaPipe version conflict

`setup.sh` pins `mediapipe>=0.10.9,<0.11` to avoid breaking API changes in 0.11+. Do not upgrade mediapipe without testing.

### apt lock during setup

If `setup.sh` stalls waiting for `apt`, it will poll automatically every 2 seconds. If it still fails, disable unattended upgrades:

```bash
sudo systemctl disable unattended-upgrades
sudo systemctl stop unattended-upgrades
```

---

*CS 370 Project — Pi Build*
