"""Run the attention pipeline on one or two video files (or webcams).

Usage:
    # Single video file
    python -m scripts.run_video --source path/to/class.mp4

    # Two video files (treated as two cameras)
    python -m scripts.run_video --source cam1.mp4 --source2 cam2.mp4

    # Webcam
    python -m scripts.run_video --source 0

    # With output video
    python -m scripts.run_video --source class.mp4 --out annotated.mp4

    # Skip display (faster, headless)
    python -m scripts.run_video --source class.mp4 --no-display
"""
import argparse
import sys
import time
import uuid
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.logger import AttentionLogger
from src.pipeline import Pipeline
from src.visualize import draw_face, draw_phones, draw_hud


def open_source(src):


# ── MJPEG stream ──────────────────────────────────────────────────────────────
_stream_frame = None
_stream_lock = threading.Lock()

def _set_stream_frame(frame):
    global _stream_frame
    jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    with _stream_lock:
        _stream_frame = jpg

class _MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence request logs
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _stream_lock:
                    frame = _stream_frame
                if frame:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.033)
        except Exception:
            pass

def _start_stream_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), _MjpegHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return port
# ─────────────────────────────────────────────────────────────────────────────


def open_source(src):
    """Accepts a file path or an integer camera index."""
    import platform
    try:
        idx = int(src)
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DirectShow avoids 0-frame issue on Windows
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)  # V4L2 default on Linux/Pi
        return cap
    except ValueError:
        return cv2.VideoCapture(str(src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file path or camera index")
    ap.add_argument("--source2", default=None, help="optional second source")
    ap.add_argument("--out", default=None, help="path to write annotated output video (first camera only)")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--stream", action="store_true", help="serve MJPEG stream on port 8080")
    ap.add_argument("--session", default=None, help="session id (default: uuid)")
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (for quick testing)")
    args = ap.parse_args()

    session_id = args.session or f"session_{int(time.time())}"
    print(f"[info] session_id = {session_id}")

    if args.stream:
        port = _start_stream_server()
        print(f"[info] MJPEG stream at http://<pi-ip>:{port}")

    # Open sources
    cap1 = open_source(args.source)
    if not cap1.isOpened():
        print(f"[error] could not open source: {args.source}")
        print("[hint]  run: ls /dev/video*  to see available cameras")
        print("[hint]  if permission denied: sudo usermod -aG video $USER  then reboot")
        return 1
    cap2 = None
    if args.source2:
        cap2 = open_source(args.source2)
        if not cap2.isOpened():
            print(f"[error] could not open source2: {args.source2}")
            return 1

    # Writer (from cam1 properties)
    writer = None
    if args.out:
        w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap1.get(cv2.CAP_PROP_FPS) or config.OUTPUT_FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        print(f"[info] writing output to {args.out} @ {w}x{h} {fps:.1f}fps")

    # Logger (one DB, both cameras write)
    logger = AttentionLogger(args.db, session_id, video_source=str(args.source))

    # Pipelines (one per camera)
    src_name_1 = str(args.source)
    pipe1 = Pipeline(camera_id=1, logger=logger)
    pipe2 = Pipeline(camera_id=2, logger=logger) if cap2 else None

    frame_idx = 0
    t_start = time.time()
    t_last = t_start

    try:
        while True:
            ok1, frame1 = cap1.read()
            if not ok1:
                break

            out1 = pipe1.process(frame1)

            frame2 = None
            out2 = None
            if cap2 is not None:
                ok2, frame2 = cap2.read()
                if ok2:
                    out2 = pipe2.process(frame2)

            frame_idx += 1

            # Draw
            if not args.no_display or writer is not None or args.stream:
                vis = frame1.copy()
                draw_phones(vis, out1.phones)
                for track_id, bbox, result in out1.faces:
                    draw_face(vis, track_id, bbox, result)

                now = time.time()
                fps = 1.0 / max(now - t_last, 1e-3)
                t_last = now
                n_attn = sum(1 for _, _, r in out1.faces if r.attentive)
                draw_hud(vis, frame_idx, len(out1.faces), n_attn, fps)

                if writer is not None:
                    writer.write(vis)

                if args.stream:
                    _set_stream_frame(vis)

                if not args.no_display:
                    cv2.imshow("attention cam1", vis)
                    if frame2 is not None and out2 is not None:
                        vis2 = frame2.copy()
                        draw_phones(vis2, out2.phones)
                        for track_id, bbox, result in out2.faces:
                            draw_face(vis2, track_id, bbox, result)
                        draw_hud(vis2, frame_idx, len(out2.faces),
                                 sum(1 for _, _, r in out2.faces if r.attentive), fps)
                        cv2.imshow("attention cam2", vis2)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[info] quit requested")
                        break

            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start
                print(f"[info] frame {frame_idx}  avg {frame_idx/elapsed:.2f} fps  "
                      f"cam1 faces={len(out1.faces)} rejected={len(out1.rejected)}")

            if args.max_frames and frame_idx >= args.max_frames:
                break

    finally:
        cap1.release()
        if cap2 is not None:
            cap2.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        pipe1.close()
        if pipe2 is not None:
            pipe2.close()
        logger.close()

    total = time.time() - t_start
    print(f"[done] {frame_idx} frames in {total:.1f}s = {frame_idx/total:.2f} fps")
    print(f"[done] session_id = {session_id}")
    print(f"[done] DB: {args.db}")
    print(f"[next] run: python -m scripts.generate_report --session {session_id} --db {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
