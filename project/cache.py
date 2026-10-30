import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("CACHE_DB_PATH", Path(__file__).parent / "cache.db"))


def _connect() -> sqlite3.Connection:
    # One short-lived connection per call: FastAPI runs sync endpoints in a
    # threadpool and sqlite connections shouldn't be shared across threads.
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_cache() -> None:
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS response_cache (
                   key         TEXT PRIMARY KEY,
                   model       TEXT NOT NULL,
                   reply       TEXT NOT NULL,
                   created_at  REAL NOT NULL,
                   hits        INTEGER NOT NULL DEFAULT 0,
                   last_hit_at REAL
               )"""
        )


def make_key(model: str, system: str | None, rendered_prompt: str) -> str:
    """SHA-256 over everything that can change the answer. JSON-encoding the
    tuple keeps fields unambiguous (and system=None distinct from system="")."""
    payload = json.dumps([model, system, rendered_prompt], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(key: str) -> str | None:
    """Cached reply or None. A cache problem is logged and treated as a miss."""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT reply FROM response_cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE response_cache SET hits = hits + 1, last_hit_at = ? WHERE key = ?",
                (time.time(), key),
            )
            return row[0]
    except Exception:
        logger.exception("Cache lookup failed")
        return None


def put(key: str, model: str, reply: str) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO response_cache (key, model, reply, created_at) VALUES (?, ?, ?, ?)",
                (key, model, reply, time.time()),
            )
    except Exception:
        logger.exception("Cache write failed")