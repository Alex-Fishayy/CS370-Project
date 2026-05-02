"""Generate an end-of-class attention report from the SQLite log.

Usage:
    python -m scripts.generate_report --session session_xxx --db attention_log.db
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--db", default="attention_log.db")
    ap.add_argument("--bucket-seconds", type=int, default=60,
                    help="time-window size for 'struggling periods' (default 60s)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    c = conn.cursor()

    # Basic counts
    c.execute(
        "SELECT COUNT(*) FROM measurements WHERE session_id = ?",
        (args.session,),
    )
    (n,) = c.fetchone()
    if n == 0:
        print(f"[error] no measurements for session {args.session}")
        return 1

    c.execute(
        """SELECT started_at, ended_at, video_source FROM sessions
           WHERE session_id = ?""",
        (args.session,),
    )
    row = c.fetchone()
    started_at, ended_at, video_source = row if row else (None, None, None)

    print("=" * 60)
    print(f"Session: {args.session}")
    print(f"Source:  {video_source}")
    if started_at and ended_at:
        duration_s = ended_at - started_at
        print(f"Duration: {duration_s:.1f}s")
    print(f"Total measurements: {n}")

    # Overall attention rate
    c.execute(
        """SELECT AVG(attentive), AVG(eyes_open), AVG(head_pose_ok), AVG(no_phone)
           FROM measurements WHERE session_id = ?""",
        (args.session,),
    )
    overall, eye_rate, pose_rate, nophone_rate = c.fetchone()
    print()
    print(f"Overall attention rate:  {overall*100:5.1f}%")
    print(f"  Eyes open rate:        {eye_rate*100:5.1f}%")
    print(f"  Head pose OK rate:     {pose_rate*100:5.1f}%")
    print(f"  No phone rate:         {nophone_rate*100:5.1f}%")

    # Per-track breakdown
    print()
    print("Per-student breakdown (by tracker ID):")
    print(f"  {'cam':>3} {'trk':>4} {'n':>5} {'attn%':>6} {'eyes%':>6} {'pose%':>6} {'nop%':>6}")
    c.execute(
        """SELECT camera_id, track_id, COUNT(*),
                  AVG(attentive), AVG(eyes_open), AVG(head_pose_ok), AVG(no_phone)
           FROM measurements WHERE session_id = ?
           GROUP BY camera_id, track_id
           ORDER BY camera_id, track_id""",
        (args.session,),
    )
    for cam, tid, count, a, e, p, np_ in c.fetchall():
        print(f"  {cam:>3} {tid:>4} {count:>5} {a*100:>5.1f}% {e*100:>5.1f}% "
              f"{p*100:>5.1f}% {np_*100:>5.1f}%")

    # Time-bucketed attention rate (struggling periods)
    print()
    print(f"Attention by {args.bucket_seconds}s bucket (from session start):")
    c.execute(
        """SELECT timestamp, attentive FROM measurements
           WHERE session_id = ? ORDER BY timestamp""",
        (args.session,),
    )
    rows = c.fetchall()
    if not rows:
        return 0

    t0 = rows[0][0]
    buckets = {}
    for ts, attn in rows:
        b = int((ts - t0) // args.bucket_seconds)
        d = buckets.setdefault(b, [0, 0])
        d[0] += attn
        d[1] += 1

    worst = []
    for b in sorted(buckets.keys()):
        a, n_ = buckets[b]
        rate = a / n_
        bar = "#" * int(rate * 40)
        t_start = b * args.bucket_seconds
        t_end = t_start + args.bucket_seconds
        print(f"  {t_start:4d}-{t_end:<4d}s  {rate*100:5.1f}%  {bar}")
        worst.append((rate, t_start, t_end, n_))

    # Struggling periods = bottom-3 buckets
    worst.sort()
    print()
    print("Struggling periods (lowest attention):")
    for rate, t_start, t_end, n_ in worst[:3]:
        print(f"  {t_start}-{t_end}s -> {rate*100:.1f}% attentive ({n_} measurements)")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
