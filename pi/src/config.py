"""All thresholds, paths, and tunables in one place."""

# Detection
MIN_FACE_SIZE_PX = 48        # reject faces smaller than this on short edge
MIN_DETECTION_CONF = 0.5     # MediaPipe face detection confidence
MAX_FACES = 4                # Pi 4: keep low to stay real-time

# Quality gate
MIN_BLUR_LAPLACIAN = 40.0    # reject blurry faces (higher = sharper required)
MAX_HEAD_YAW_FOR_QUALITY = 60.0  # degrees; beyond this, landmarks unreliable

# Attention thresholds
HEAD_YAW_LIMIT = 35.0        # degrees; |yaw| must be under this
HEAD_PITCH_MIN = -20.0       # degrees; looking down at notes ok
HEAD_PITCH_MAX = 15.0        # degrees; head back = zoned out

EAR_THRESHOLD = 0.20         # Eye Aspect Ratio below this = closed
EAR_CLOSED_FRAMES = 3        # consecutive frames of closed eyes before flagging

# Phone detection
PHONE_CONF_THRESHOLD = 0.35  # YOLO confidence for phone class
PHONE_DETECT_EVERY_N_FRAMES = 10  # Pi: run phone detection less often to save CPU
PHONE_PROXIMITY_MULT = 2.0   # phone within N * face_height counts as "nearby"

# Tracking
IOU_MATCH_THRESHOLD = 0.3    # min IoU for track association
MAX_MISSED_FRAMES = 15       # frames a track can go undetected before dropping

# Processing
PROCESS_EVERY_N_FRAMES = 1   # frame skip (1 = every frame). Increase to speed up.
OUTPUT_FPS = 30              # for annotated output video

# Logging
DB_PATH = "attention_log.db"

# TFLite phone detection (no PyTorch needed)
TFLITE_MODEL = "models/detect.tflite"   # downloaded by setup.sh
PHONE_CLASS_ID = 76          # 0-indexed COCO class for "cell phone" (77 in 1-indexed labelmap)

# Gaze tracking
GAZE_THRESHOLD = 0.45        # normalized iris offset [0-1]; beyond this = looking away

# Analysis subsampling
ATTENTION_ANALYSIS_EVERY_N_FRAMES = 15  # Pi: analyze every 15 frames (~2x/sec at 30fps)
