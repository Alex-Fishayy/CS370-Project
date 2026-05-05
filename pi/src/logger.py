"""SQLite writer for per-measurement attention data."""
import sqlite3
import threading
import time
from contextlib import contextmanager


SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    camera_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    frame_idx INTEGER NOT NULL,
    bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
    yaw REAL, pitch REAL, ear REAL,
    eyes_open INTEGER, head_pose_ok INTEGER, no_phone INTEGER, attentive INTEGER,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_session_time ON measurements(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_track ON measurements(session_id, camera_id, track_id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    video_source TEXT,
    notes TEXT
);
"""


class AttentionLogger:
    def __init__(self, db_path: str, session_id: str, video_source: str = ""):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.session_id = session_id
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, started_at, video_source) VALUES (?, ?, ?)",
            (session_id, time.time(), video_source),
        )
        self.conn.commit()
        self._pending = []
        self._batch_size = 5

    def log(self, camera_id: int, track_id: int, frame_idx: int, bbox, result):
        with self._lock:
            self._pending.append((
                self.session_id, camera_id, track_id, time.time(), frame_idx,
                bbox[0], bbox[1], bbox[2], bbox[3],
                result.yaw, result.pitch, result.ear,
                int(result.eyes_open), int(result.head_pose_ok),
                int(result.no_phone), int(result.attentive),
                result.reason,
            ))
            if len(self._pending) >= self._batch_size:
                self._flush_locked()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        """Must be called with self._lock held."""
        if not self._pending:
            return
        self.conn.executemany(
            """INSERT INTO measurements
               (session_id, camera_id, track_id, timestamp, frame_idx,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                yaw, pitch, ear,
                eyes_open, head_pose_ok, no_phone, attentive, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            self._pending,
        )
        self.conn.commit()
        self._pending.clear()

    def close(self):
        self.flush()
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (time.time(), self.session_id),
            )
            self.conn.commit()
        self.conn.close()
