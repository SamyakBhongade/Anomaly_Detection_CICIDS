"""
db.py
=====
SQLite database manager for the Anomaly Detection Dashboard.
Stores every detection event with timestamp, label, confidence and model scores.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detections.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            ip_address   TEXT    DEFAULT 'Unknown',
            label        INTEGER NOT NULL,       -- 0=BENIGN, 1=ATTACK
            attack_type  TEXT    NOT NULL DEFAULT 'Unknown',
            confidence   REAL    NOT NULL,
            bilstm       REAL    NOT NULL,
            cnn          REAL    NOT NULL,
            transformer  REAL    NOT NULL,
            vae          REAL    NOT NULL
        )
    """)
    
    # Simple migration: add ip_address if it doesn't exist (fails silently if it does)
    try:
        conn.execute("ALTER TABLE detections ADD COLUMN ip_address TEXT DEFAULT 'Unknown'")
    except sqlite3.OperationalError:
        pass
        
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blocked_ips (
            ip_address TEXT PRIMARY KEY,
            blocked_at TEXT NOT NULL
        )
    """)
        
    conn.commit()
    conn.close()

def block_ip(ip_address: str):
    """Add an IP address to the blocklist."""
    conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO blocked_ips (ip_address, blocked_at)
        VALUES (?, ?)
    """, (ip_address, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def is_ip_blocked(ip_address: str) -> bool:
    """Check if an IP address is blocked."""
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM blocked_ips WHERE ip_address = ?", (ip_address,)).fetchone()
    conn.close()
    return row is not None


def insert_detection(label: int, confidence: float,
                     bilstm: float, cnn: float,
                     transformer: float, vae: float,
                     attack_type: str = 'Unknown'):
    """Insert a single detection event."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO detections
            (timestamp, label, attack_type, confidence, bilstm, cnn, transformer, vae)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
          int(label), attack_type, round(float(confidence), 5),
          round(float(bilstm), 5), round(float(cnn), 5),
          round(float(transformer), 5), round(float(vae), 5)))
    conn.commit()
    conn.close()


def insert_batch(records: list):
    """
    Insert multiple detection events at once.
    Each record: (timestamp, ip_address, label, attack_type, confidence, bilstm, cnn, transformer, vae)
    """
    conn = get_connection()
    conn.executemany("""
        INSERT INTO detections
            (timestamp, ip_address, label, attack_type, confidence, bilstm, cnn, transformer, vae)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()


def get_stats() -> dict:
    """Return aggregate statistics for the dashboard."""
    conn = get_connection()
    row = conn.execute("""
        SELECT
            COUNT(*)                                       AS total,
            SUM(CASE WHEN label=1 THEN 1 ELSE 0 END)      AS attacks,
            SUM(CASE WHEN label=0 THEN 1 ELSE 0 END)      AS benign,
            ROUND(AVG(CASE WHEN label=1 THEN confidence END), 4) AS avg_attack_conf
        FROM detections
    """).fetchone()

    # Attack type breakdown
    type_rows = conn.execute("""
        SELECT attack_type, COUNT(*) as cnt
        FROM detections
        WHERE label = 1
        GROUP BY attack_type
        ORDER BY cnt DESC
    """).fetchall()
    conn.close()

    total   = row['total']   or 0
    attacks = row['attacks'] or 0
    benign  = row['benign']  or 0
    return {
        'total':            total,
        'attacks':          attacks,
        'benign':           benign,
        'attack_rate':      round(100 * attacks / max(total, 1), 2),
        'avg_attack_conf':  row['avg_attack_conf'] or 0.0,
        'attack_types':     {r['attack_type']: r['cnt'] for r in type_rows},
    }


def get_recent_feed(limit: int = 50) -> list:
    """Return the most recent N detection events."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, timestamp, ip_address, label, attack_type, confidence, bilstm, cnn, transformer, vae
        FROM detections
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_id() -> int:
    conn = get_connection()
    row  = conn.execute("SELECT MAX(id) AS mid FROM detections").fetchone()
    conn.close()
    return row['mid'] or 0

def get_ip_history(ip_address: str, limit: int = 50) -> list:
    """Return the recent detection history for a specific IP."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, timestamp, label, attack_type, confidence, bilstm, cnn, transformer, vae
        FROM detections
        WHERE ip_address = ?
        ORDER BY id DESC
        LIMIT ?
    """, (ip_address, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_new_events_since(last_id: int) -> list:
    """Return all events inserted after last_id (for SSE alerts)."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, timestamp, ip_address, label, attack_type, confidence
        FROM detections
        WHERE id > ?
        ORDER BY id ASC
    """, (last_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]



def reset_db():
    """Wipe all data — useful for demo restarts."""
    conn = get_connection()
    conn.execute("DELETE FROM detections")
    conn.execute("DELETE FROM sqlite_sequence WHERE name='detections'")
    conn.commit()
    conn.close()
