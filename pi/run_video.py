"""Multi-camera attention tracker with auto-detection.

Auto-detects all connected cameras, runs a tracking pipeline on each in a
background thread, and serves a live web dashboard on port 8080:

    http://<pi-ip>:8080          → HTML dashboard (all feeds + DB metrics)
    http://<pi-ip>:8080/cam/N    → raw MJPEG for camera N
    http://<pi-ip>:8080/metrics  → JSON metrics

Usage:
    # Auto-detect all cameras, headless + stream
    python run_video.py --no-display --stream

    # Specify cameras explicitly
    python run_video.py --source 0,1 --no-display --stream

    # Single camera (legacy)
    python run_video.py --source 0
"""
import argparse
import json
import os
import platform
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.logger import AttentionLogger
from src.pipeline import Pipeline
from src.visualize import draw_face, draw_phones, draw_hud


# ── Shared state (written by camera threads, read by HTTP handler) ─────────────
_lock = threading.Lock()
_frames: dict = {}   # cam_idx -> latest JPEG bytes
_stats: dict = {}    # cam_idx -> {faces, attentive, fps, session}
_db_path: str = ""   # always absolute
_session_id: str = ""
_stop_event = threading.Event()


def _cpu_temp_c() -> float | None:
    """Read CPU temperature from thermal zone or vcgencmd (Pi-specific)."""
    # Standard Linux thermal zone (works on Pi and most ARM boards)
    for i in range(8):
        try:
            with open(f'/sys/class/thermal/thermal_zone{i}/temp') as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            continue
    return None


def _cpu_freq_ghz() -> float | None:
    """Read current CPU clock speed from /proc/cpuinfo or sysfs."""
    # Try sysfs scaling_cur_freq (most accurate live value)
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq') as f:
            return round(int(f.read().strip()) / 1_000_000, 2)
    except Exception:
        pass
    # Fallback: parse /proc/cpuinfo 'cpu MHz'
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('cpu MHz'):
                    return round(float(line.split(':')[1].strip()) / 1000, 2)
    except Exception:
        pass
    return None


def _sys_stats() -> dict:
    """CPU %, RAM used/total, temperature, clock. Falls back gracefully."""
    temp = _cpu_temp_c()
    freq = _cpu_freq_ghz()

    if not _HAS_PSUTIL:
        # Fallback: read /proc/meminfo directly (Linux only)
        try:
            with open('/proc/meminfo') as f:
                lines = {l.split(':')[0]: int(l.split()[1]) for l in f if ':' in l}
            total_kb = lines.get('MemTotal', 0)
            avail_kb = lines.get('MemAvailable', 0)
            used_mb  = (total_kb - avail_kb) // 1024
            total_mb = total_kb // 1024
            pct_ram  = round(100 * (total_kb - avail_kb) / max(total_kb, 1), 1)
        except Exception:
            used_mb = total_mb = 0
            pct_ram = 0
        return {"cpu": None, "ram_used_mb": used_mb,
                "ram_total_mb": total_mb, "ram_pct": pct_ram,
                "temp_c": temp, "freq_ghz": freq}
    cpu = psutil.cpu_percent(interval=None)
    vm  = psutil.virtual_memory()
    # psutil can also read temp/freq if available, but our direct reads are more reliable on Pi
    if temp is None:
        try:
            temps = psutil.sensors_temperatures()
            for key in ('cpu_thermal', 'cpu-thermal', 'coretemp'):
                if key in temps and temps[key]:
                    temp = round(temps[key][0].current, 1)
                    break
        except Exception:
            pass
    if freq is None:
        try:
            freq = round(psutil.cpu_freq().current / 1000, 2)
        except Exception:
            pass
    return {"cpu": cpu, "ram_used_mb": vm.used // (1024*1024),
            "ram_total_mb": vm.total // (1024*1024),
            "ram_pct": round(vm.percent, 1),
            "temp_c": temp, "freq_ghz": freq}


# ── Camera helpers ────────────────────────────────────────────────────────────
def _open_cam(idx: int):
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
    else:
        # On Linux/Pi, open by device path to bypass V4L2 enumeration gaps.
        # /dev/video0, /dev/video2, etc. are skipped when using integer indices
        # if the kernel assigns non-contiguous nodes (e.g. metadata devices).
        cap = cv2.VideoCapture(f"/dev/video{idx}", cv2.CAP_V4L2)
    if cap.isOpened():
        # Force MJPEG before setting resolution — YUYV at 1280x720 saturates USB
        # bandwidth on the Pi and causes V4L2 select() timeouts.  MJPEG compresses
        # on-camera so the USB payload is ~10x smaller.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency / stale frames
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def detect_cameras(max_index: int = 32) -> list:
    """Scan /dev/video0../dev/video(max_index-1), return indices that deliver frames.

    On Raspberry Pi, UVC webcams may appear at non-contiguous indices (e.g. 0 and 2)
    because metadata/ISP devices occupy the gaps.  We probe every slot directly by
    device path rather than relying on OpenCV's V4L2 enumeration.
    """
    found = []
    for i in range(max_index):
        dev = f"/dev/video{i}"
        if not os.path.exists(dev):
            continue
        cap = _open_cam(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
                print(f"[info] found camera {i} ({dev})", flush=True)
        cap.release()
    return found


# ── Dashboard HTML ────────────────────────────────────────────────────────────
_DASHBOARD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Attention Tracker</title>
<style>
  * { box-sizing: border-box; }
  body { background: #0d0d0d; color: #e0e0e0; font-family: 'Courier New', monospace;
         margin: 0; padding: 16px; }
  h1 { margin: 0 0 14px; font-size: 1.3em; color: #7dd3fc; letter-spacing: 0.05em; }
  h2 { font-size: 0.9em; color: #7dd3fc; margin: 14px 0 6px; border-bottom: 1px solid #1e3a5f; padding-bottom: 3px; }
  .feeds { display: flex; flex-wrap: wrap; gap: 10px; }
  .cam-card { background: #161616; border: 1px solid #2a2a2a; border-radius: 6px;
              padding: 8px; flex: 1 1 300px; min-width: 280px; max-width: 640px; }
  .cam-card h3 { margin: 0 0 6px; font-size: 0.85em; color: #fbbf24; }
  .cam-card img { width: 100%; border-radius: 4px; display: block; background: #111; }
  table { border-collapse: collapse; width: 100%; font-size: 0.82em; margin-bottom: 6px; }
  th { background: #1a1a1a; color: #7dd3fc; padding: 5px 8px; text-align: center;
       border: 1px solid #2a2a2a; }
  td { padding: 4px 8px; border: 1px solid #222; text-align: right; }
  td:first-child, td:nth-child(2) { text-align: left; }
  .good { color: #4ade80; font-weight: bold; }
  .warn { color: #fbbf24; font-weight: bold; }
  .bad  { color: #f87171; font-weight: bold; }
  .sysbar { display: flex; gap: 24px; font-size: 0.82em; background: #111;
            border: 1px solid #2a2a2a; border-radius: 4px; padding: 6px 12px;
            margin-bottom: 12px; }
  .sysbar span { color: #a0a0a0; }
  .sysbar b { color: #e0e0e0; }
  #db-error { font-size: 0.75em; color: #f87171; margin-top: 4px; min-height: 1em; }
  #status { font-size: 0.75em; color: #555; margin-top: 10px; }
</style></head><body>
<h1>Attention Tracker -- Live Dashboard</h1>

<div class="sysbar" id="sysbar">loading system info...</div>

<div class="feeds" id="feeds"></div>

<h2>Live per-camera</h2>
<table><thead><tr>
  <th>Camera</th><th>Faces</th><th>Avg Attn%</th><th>FPS</th>
</tr></thead><tbody id="live-body"></tbody></table>

<h2>DB metrics -- avg per person</h2>
<div id="db-error"></div>
<table><thead><tr>
  <th>Cam</th><th>People</th><th>Samples</th>
  <th>Avg Attn%</th><th>Avg Eyes%</th><th>Avg Pose%</th><th>Avg NoPhone%</th>
</tr></thead><tbody id="db-body"></tbody></table>

<div id="status">connecting...</div>

<script>
let knownCams = null;

function cls(pct) {
  return pct >= 70 ? 'good' : pct >= 50 ? 'warn' : 'bad';
}

async function refresh() {
  try {
    const r = await fetch('/metrics');
    const d = await r.json();

    // Camera feed tiles
    if (JSON.stringify(d.cameras) !== JSON.stringify(knownCams)) {
      knownCams = d.cameras;
      const feeds = document.getElementById('feeds');
      feeds.innerHTML = '';
      d.cameras.forEach(idx => {
        feeds.innerHTML +=
          `<div class="cam-card"><h3>Camera ${idx}</h3>` +
          `<img src="/cam/${idx}" alt="Camera ${idx}"></div>`;
      });
    }

    // Live table — use DB per-person avg when available, else fall back to live ratio
    const dbAvg = {};
    (d.db_rows || []).forEach(r => { dbAvg[r.cam] = r.attn_pct; });
    let live = '';
    (d.live || []).forEach(s => {
      const rate = dbAvg[s.cam] != null ? dbAvg[s.cam]
                   : (s.faces > 0 ? Math.round(100 * s.attentive / s.faces) : 0);
      live += `<tr><td>Cam ${s.cam}</td><td>${s.faces}</td>` +
              `<td class="${cls(rate)}">${rate}%</td><td>${s.fps}</td></tr>`;
    });
    document.getElementById('live-body').innerHTML = live;

    // DB table
    let db = '';
    if (d.db_rows && d.db_rows.length) {
      d.db_rows.forEach(row => {
        db += `<tr><td>${row.cam}</td><td>${row.people}</td><td>${row.n}</td>` +
              `<td class="${cls(row.attn_pct)}">${row.attn_pct}%</td>` +
              `<td class="${cls(row.eyes_pct)}">${row.eyes_pct}%</td>` +
              `<td class="${cls(row.pose_pct)}">${row.pose_pct}%</td>` +
              `<td class="${cls(row.nophone_pct)}">${row.nophone_pct}%</td></tr>`;
      });
    } else {
      db = '<tr><td colspan="7" style="text-align:center;color:#555">no data yet</td></tr>';
    }
    document.getElementById('db-body').innerHTML = db;

    // System bar
    const sys = d.sys || {};
    const cpu = sys.cpu != null ? sys.cpu + '%' : 'n/a';
    const ram = sys.ram_used_mb != null
      ? sys.ram_used_mb + ' / ' + sys.ram_total_mb + ' MB (' + sys.ram_pct + '%%)'
      : 'n/a';
    const temp = sys.temp_c != null ? sys.temp_c + ' C' : 'n/a';
    const freq = sys.freq_ghz != null ? sys.freq_ghz + ' GHz' : 'n/a';
    document.getElementById('sysbar').innerHTML =
      '<span>CPU:</span> <b>' + cpu + '</b>' +
      '&nbsp;&nbsp;<span>Temp:</span> <b>' + temp + '</b>' +
      '&nbsp;&nbsp;<span>Freq:</span> <b>' + freq + '</b>' +
      '&nbsp;&nbsp;<span>RAM:</span> <b>' + ram + '</b>' +
      '&nbsp;&nbsp;<span>Session:</span> <b>' + (d.session || '-') + '</b>';

    // DB error
    document.getElementById('db-error').textContent = d.db_error || '';

    document.getElementById('status').textContent =
      'last update: ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('status').textContent = 'connection error: ' + e;
  }
}

refresh();
setInterval(refresh, 2000);
</script></body></html>"""


# ── HTTP server ───────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence per-request logs

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            body = _DASHBOARD.encode()
            self._respond(200, "text/html", body)

        elif path.startswith("/cam/"):
            try:
                idx = int(path.rsplit("/", 1)[-1])
            except ValueError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _stop_event.is_set():
                    with _lock:
                        jpg = _frames.get(idx)
                    if jpg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                        )
                    time.sleep(0.033)
            except Exception:
                pass

        elif path == "/metrics":
            with _lock:
                cams = sorted(_frames.keys())
                live = [{"cam": k, **v} for k, v in _stats.items()]

            db_rows = []
            db_error = ""
            if _db_path and _session_id:
                try:
                    # Use absolute path + WAL mode to read while writer thread writes
                    conn = sqlite3.connect(_db_path, timeout=10,
                                           check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")
                    rows = conn.execute(
                        """WITH pp AS (
                             SELECT camera_id, track_id,
                                    1.0*SUM(attentive)/COUNT(*)    AS attn,
                                    1.0*SUM(eyes_open)/COUNT(*)    AS eyes,
                                    1.0*SUM(head_pose_ok)/COUNT(*) AS pose,
                                    1.0*SUM(no_phone)/COUNT(*)     AS nophone,
                                    COUNT(*) AS n
                             FROM measurements WHERE session_id=?
                             GROUP BY camera_id, track_id
                           )
                           SELECT camera_id,
                                  COUNT(*) AS people,
                                  SUM(n)   AS total_n,
                                  ROUND(100.0*AVG(attn),    1),
                                  ROUND(100.0*AVG(eyes),    1),
                                  ROUND(100.0*AVG(pose),    1),
                                  ROUND(100.0*AVG(nophone), 1)
                           FROM pp
                           GROUP BY camera_id
                           ORDER BY camera_id""",
                        (_session_id,),
                    ).fetchall()
                    conn.close()
                    db_rows = [
                        {"cam": r[0], "people": r[1], "n": r[2],
                         "attn_pct": r[3] or 0, "eyes_pct": r[4] or 0,
                         "pose_pct": r[5] or 0, "nophone_pct": r[6] or 0}
                        for r in rows
                    ]
                except Exception as exc:
                    db_error = str(exc)

            payload = {
                "cameras": cams,
                "live": live,
                "db_rows": db_rows,
                "db_error": db_error,
                "session": _session_id,
                "sys": _sys_stats(),
            }
            body = json.dumps(payload).encode()
            self._respond(200, "application/json", body)

        else:
            self.send_error(404)

    def _respond(self, code, content_type, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


def _start_server(port: int = 8080):
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[info] dashboard → http://0.0.0.0:{port}")


# ── Per-camera worker thread ──────────────────────────────────────────────────
def _camera_worker(cam_idx: int, logger: AttentionLogger, session_id: str,
                   no_display: bool, max_frames, analysis_every: int):
    cap = _open_cam(cam_idx)
    if not cap.isOpened():
        print(f"[cam {cam_idx}] failed to open — skipping")
        return

    pipe = Pipeline(camera_id=cam_idx, logger=logger, analysis_every=analysis_every)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam {cam_idx}] resolution {w}x{h}  analysis_every={analysis_every}", flush=True)

    # ── Shared state between capture loop and pipeline thread ─────────────────
    _buf_lock   = threading.Lock()
    _buf_frame  = [None]   # latest raw frame from cap.read()
    _buf_fid    = [0]      # increments each capture so pipeline knows if new frame arrived
    _buf_out    = [None]   # latest FrameOutput from pipeline
    _buf_nattn  = [0]
    _fps_val    = [0.0]
    _t_fps      = [time.time()]

    # ── Pipeline thread — runs at MediaPipe speed (~10 fps) ───────────────────
    def _pipeline_loop():
        last_fid = 0
        pipeline_idx = 0
        t_start = time.time()
        while not _stop_event.is_set():
            with _buf_lock:
                fid   = _buf_fid[0]
                frame = _buf_frame[0]
            if frame is None or fid == last_fid:
                time.sleep(0.002)
                continue
            last_fid = fid
            pipeline_idx += 1

            out    = pipe.process(frame)
            n_attn = sum(1 for _, _, r in out.faces if r.attentive)

            with _buf_lock:
                _buf_out[0]   = out
                _buf_nattn[0] = n_attn

            with _lock:
                _stats[cam_idx] = {
                    "faces":    len(out.faces),
                    "attentive": n_attn,
                    "fps":       round(_fps_val[0], 1),
                    "session":   session_id,
                }

            if pipeline_idx % 30 == 0:
                elapsed = time.time() - t_start
                print(f"[cam {cam_idx}] pipeline {pipeline_idx} frames  "
                      f"avg {pipeline_idx/elapsed:.1f} fps  faces={len(out.faces)}", flush=True)

    pt = threading.Thread(target=_pipeline_loop, daemon=True, name=f"pipeline-{cam_idx}")
    pt.start()

    # ── Capture loop — always runs at ~30 fps; pipeline analysis is decoupled ──
    _TARGET_FRAME_S = 1.0 / 30.0
    capture_idx = 0
    _consecutive_fail = 0
    try:
        while not _stop_event.is_set():
            t_frame_start = time.time()   # stamp before cap.read() for accurate budget

            ok, frame = cap.read()
            if not ok:
                _consecutive_fail += 1
                time.sleep(0.05)
                if _consecutive_fail >= 20:  # ~1 second of failures → reconnect
                    print(f"[cam {cam_idx}] read timeout — reconnecting…", flush=True)
                    cap.release()
                    time.sleep(1.0)
                    cap = _open_cam(cam_idx)
                    if not cap.isOpened():
                        print(f"[cam {cam_idx}] reconnect failed — retrying in 5s", flush=True)
                        time.sleep(5.0)
                    else:
                        print(f"[cam {cam_idx}] reconnected", flush=True)
                    _consecutive_fail = 0
                continue
            _consecutive_fail = 0

            capture_idx += 1
            now = time.time()
            _fps_val[0] = 1.0 / max(now - _t_fps[0], 1e-3)
            _t_fps[0]   = now

            # Hand latest frame to pipeline thread (non-blocking)
            with _buf_lock:
                _buf_frame[0] = frame
                _buf_fid[0]   = capture_idx
                out    = _buf_out[0]
                n_attn = _buf_nattn[0]

            # Annotate with last-known results and push to stream at full fps
            vis = frame.copy()
            if out:
                draw_phones(vis, out.phones)
                for track_id, bbox, result in out.faces:
                    draw_face(vis, track_id, bbox, result)
                draw_hud(vis, capture_idx, len(out.faces), n_attn, _fps_val[0])

            jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            with _lock:
                _frames[cam_idx] = jpg

            if not no_display:
                cv2.imshow(f"attention cam{cam_idx}", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    _stop_event.set()
                    break

            if max_frames and capture_idx >= max_frames:
                break

            # Sleep out any remaining budget so display stays at 30 fps.
            # t_frame_start was captured before cap.read(), so this correctly
            # accounts for both the read and the annotation time.
            spare = _TARGET_FRAME_S - (time.time() - t_frame_start)
            if spare > 0:
                time.sleep(spare)
    finally:
        cap.release()
        try:
            pipe.close()
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global _db_path, _session_id

    ap = argparse.ArgumentParser(
        description="Multi-camera attention tracker with live web dashboard"
    )
    ap.add_argument(
        "--source", default=None,
        help="comma-separated camera indices (e.g. '0,1') or omit to auto-detect"
    )
    ap.add_argument("--no-display", action="store_true",
                    help="headless mode — no OpenCV window")
    ap.add_argument("--stream", action="store_true",
                    help="serve web dashboard on port 8080")
    ap.add_argument("--session", default=None)
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop each camera after N frames (testing)")
    args = ap.parse_args()

    session_id = args.session or f"session_{int(time.time())}"
    _session_id = session_id
    # Always store absolute path so HTTP thread can find the DB regardless of cwd
    _db_path = str(Path(args.db).resolve())
    print(f"[info] session_id = {session_id}", flush=True)

    # Start HTTP server first so dashboard is reachable during camera init
    if args.stream or args.no_display:
        _start_server(args.port)

    # Determine camera list
    if args.source:
        cam_indices = [int(x.strip()) for x in args.source.split(",")]
    else:
        print("[info] auto-detecting cameras…", flush=True)
        cam_indices = detect_cameras()

    if not cam_indices:
        print("[error] no cameras found", flush=True)
        print("[hint]  plug in a USB webcam, or pass --source 0 explicitly")
        return 1

    # Scale analysis frequency so CPU stays at ~100% regardless of camera count.
    # Fewer cameras → more analysis passes per second (more work per thread).
    # More cameras → less frequent analysis so all threads can keep up.
    _analysis_scale = {1: 10, 2: 25, 3: 30}
    analysis_every = _analysis_scale.get(len(cam_indices), 35)
    print(f"[info] running on cameras: {cam_indices}  "
          f"analysis_every={analysis_every} frames", flush=True)

    logger = AttentionLogger(args.db, session_id, video_source=str(cam_indices))

    threads = []
    for idx in cam_indices:
        t = threading.Thread(
            target=_camera_worker,
            args=(idx, logger, session_id, args.no_display, args.max_frames, analysis_every),
            daemon=True,
            name=f"cam-{idx}",
        )
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[info] stopping…")
        _stop_event.set()
    finally:
        # Give threads a moment to exit cleanly
        for t in threads:
            t.join(timeout=3)
        logger.close()
        cv2.destroyAllWindows()

    print(f"[done] session_id = {session_id}")
    print(f"[done] DB: {args.db}")
    print(f"[next] python generate_report.py --session {session_id} --db {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
