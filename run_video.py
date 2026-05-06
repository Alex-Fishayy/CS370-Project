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
import threading
import time
import uuid
from pathlib import Path

import cv2

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.logger import AttentionLogger
from src.pipeline import Pipeline
from src.visualize import draw_face, draw_phones, draw_hud


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
    ap.add_argument("--session", default=None, help="session id (default: uuid)")
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (for quick testing)")
    args = ap.parse_args()

    session_id = args.session or f"session_{int(time.time())}"
    print(f"[info] session_id = {session_id}")

    # Open sources
    cap1 = open_source(args.source)
    if not cap1.isOpened():
        print(f"[error] could not open source: {args.source}")
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
    pipe1 = Pipeline(camera_id=1, logger=logger)
    pipe2 = Pipeline(camera_id=2, logger=logger) if cap2 else None

    # ── Shared buffers: capture loop writes frames, pipeline threads read them ─
    _buf_lock = threading.Lock()
    _buf1 = {"frame": None, "fid": 0, "out": None}
    _buf2 = {"frame": None, "fid": 0, "out": None}
    _stop_ev = threading.Event()

    def _pipeline_thread(pipe, buf):
        last_fid = 0
        while not _stop_ev.is_set():
            with _buf_lock:
                fid   = buf["fid"]
                frame = buf["frame"]
            if frame is None or fid == last_fid:
                time.sleep(0.002)
                continue
            last_fid = fid
            out = pipe.process(frame)
            with _buf_lock:
                buf["out"] = out

    t1 = threading.Thread(target=_pipeline_thread, args=(pipe1, _buf1), daemon=True)
    t1.start()
    t2 = None
    if pipe2:
        t2 = threading.Thread(target=_pipeline_thread, args=(pipe2, _buf2), daemon=True)
        t2.start()

    # ── Capture + display loop — always runs at ~30 fps ───────────────────────
    _TARGET_FRAME_S = 1.0 / 30.0
    frame_idx = 0
    t_start   = time.time()
    t_last    = t_start

    try:
        while True:
            t_frame_start = time.time()

            ok1, frame1 = cap1.read()
            if not ok1:
                break

            frame_idx += 1

            frame2 = None
            if cap2 is not None:
                ok2, frame2 = cap2.read()
                if not ok2:
                    frame2 = None

            # Hand frames to pipeline threads (non-blocking)
            with _buf_lock:
                _buf1["frame"] = frame1
                _buf1["fid"]   = frame_idx
                out1 = _buf1["out"]

                if frame2 is not None:
                    _buf2["frame"] = frame2
                    _buf2["fid"]   = frame_idx
                out2 = _buf2["out"]

            # Draw with last-known pipeline results (may be None on first frames)
            if not args.no_display or writer is not None:
                vis = frame1.copy()
                if out1 is not None:
                    draw_phones(vis, out1.phones)
                    for track_id, bbox, result in out1.faces:
                        draw_face(vis, track_id, bbox, result)

                now = time.time()
                fps = 1.0 / max(now - t_last, 1e-3)
                t_last = now
                n_faces = len(out1.faces) if out1 else 0
                n_attn  = sum(1 for _, _, r in out1.faces if r.attentive) if out1 else 0
                draw_hud(vis, frame_idx, n_faces, n_attn, fps)

                if writer is not None:
                    writer.write(vis)

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
                n_rej = len(out1.rejected) if out1 else 0
                print(f"[info] frame {frame_idx}  avg {frame_idx/elapsed:.2f} fps  "
                      f"cam1 faces={n_faces} rejected={n_rej}")

            if args.max_frames and frame_idx >= args.max_frames:
                break

            # Sleep out remaining budget to hold ~30 fps display
            spare = _TARGET_FRAME_S - (time.time() - t_frame_start)
            if spare > 0:
                time.sleep(spare)

    finally:
        _stop_ev.set()
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
