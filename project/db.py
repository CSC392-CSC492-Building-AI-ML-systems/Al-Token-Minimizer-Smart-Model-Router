"""SQLite telemetry store: schema, init script and a small client.

One row per request in the `requests` table. Other modules should go through
`insert_request` / `fetch_requests` and not write raw SQL.

Run `python db.py` to create the database file.
"""

import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "telemetry.db"

# Allowed values for the `status` column. Details go in http_status,
# error_type and error_message.
STATUS_OK = "ok"  # Gemini returned status "completed" with no errors
STATUS_UPSTREAM_ERROR = "upstream_error"  # SDK raised an HTTP error (4xx/5xx, incl. 429)
STATUS_TIMEOUT = "timeout"  # the call to Gemini timed out
STATUS_CONNECTION_ERROR = "connection_error"  # could not reach Gemini
STATUS_NOT_COMPLETED = "not_completed"  # returned, but Gemini status wasn't "completed"
STATUS_INTERNAL_ERROR = "internal_error"  # our own code failed
STATUSES = (
    STATUS_OK,
    STATUS_UPSTREAM_ERROR,
    STATUS_TIMEOUT,
    STATUS_CONNECTION_ERROR,
    STATUS_NOT_COMPLETED,
    STATUS_INTERNAL_ERROR,
)

# Column name -> SQL type. Order here is the order of the table.
COLUMNS = {
    # identity
    "request_id": "TEXT PRIMARY KEY", # something we probably will generate
    "timestamp": "TEXT NOT NULL",  # UTC, ISO 8601
    "endpoint": "TEXT", 
    "model": "TEXT",
    "interaction_id": "TEXT",  # Gemini's id for the interaction
    "prompt_hash": "TEXT",  # never the prompt itself
    # tokens
    "input_tokens_raw": "INTEGER",  # our count before compression
    "input_tokens_compressed": "INTEGER",  # same as raw until compression exists
    "input_tokens_reported": "INTEGER",  # from gemini
    "output_tokens": "INTEGER", # from gemini 
    "thought_tokens": "INTEGER", # from gemini
    "cached_tokens": "INTEGER", # from gemini
    "total_tokens": "INTEGER", # from gemini
    # latency, in milliseconds
    "latency_total_ms": "REAL",
    "latency_upstream_ms": "REAL",
    "latency_count_ms": "REAL",
    # outcome
    "status": "TEXT NOT NULL CHECK (status IN (%s))"
    % ", ".join(f"'{s}'" for s in STATUSES),
    "http_status": "INTEGER",  # HTTP code for upstream_error, else NULL
    "error_type": "TEXT",  # exception class, or Gemini's raw status for not_completed
    "error_message": "TEXT",
    "cost_usd": "REAL",  # NULL if the model has no price
}

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS requests (\n    "
    + ",\n    ".join(f"{name} {sql_type}" for name, sql_type in COLUMNS.items())
    + "\n);\n"
    "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests (timestamp);\n"
)


@dataclass
class RequestRecord:
    """One request's information
    Anything never set is saved as NULL. `status` must be set before saving (use the STATUS_* constants).
    Fields must match COLUMNS exactly, in the same order (checked below).
    """
    # identity
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    endpoint: str | None = None
    model: str | None = None
    interaction_id: str | None = None
    prompt_hash: str | None = None
    # tokens
    input_tokens_raw: int | None = None
    input_tokens_compressed: int | None = None
    input_tokens_reported: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    # latency, in milliseconds
    latency_total_ms: float | None = None
    latency_upstream_ms: float | None = None
    latency_count_ms: float | None = None
    # outcome
    status: str | None = None
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    cost_usd: float | None = None


if [f.name for f in fields(RequestRecord)] != list(COLUMNS):
    raise RuntimeError(
        "RequestRecord fields and db.COLUMNS are out of sync: "
        "add or remove the column in both places, in the same order"
    )


def _resolve_path(db_path=None):
    return Path(db_path or os.environ.get("TELEMETRY_DB_PATH") or DEFAULT_DB_PATH)


def get_connection(db_path=None):
    """Open a connection. Rows come back as sqlite3.Row (index by column name)."""
    conn = sqlite3.connect(_resolve_path(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Create the database file and table if they don't exist, and turn on WAL."""
    path = _resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(get_connection(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
    return path


def insert_request(record, db_path=None):
    """Insert one request row and return its request_id.

    `record` is a RequestRecord, or a dict keyed by column name. For a dict,
    `request_id` and `timestamp` are filled in if missing. `status` is required.
    Unknown keys raise ValueError.
    """
    if isinstance(record, RequestRecord):
        record = asdict(record)
    unknown = set(record) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown columns: {sorted(unknown)}")

    row = dict(record)
    row.setdefault("request_id", str(uuid.uuid4()))
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    if not row.get("status"):
        raise ValueError("status is required")

    names = list(row)
    sql = (
        f"INSERT INTO requests ({', '.join(names)}) "
        f"VALUES ({', '.join('?' for _ in names)})"
    )
    with closing(get_connection(db_path)) as conn:
        conn.execute(sql, [row[name] for name in names])
        conn.commit()
    return row["request_id"]


def fetch_requests(
    db_path=None, model=None, status=None, limit=None, newest_first=False, request_id=None
):
    """Return rows as a list of dicts, oldest first unless newest_first is set."""
    where, params = [], []
    if request_id is not None:
        where.append("request_id = ?")
        params.append(request_id)
    if model is not None:
        where.append("model = ?")
        params.append(model)
    if status is not None:
        where.append("status = ?")
        params.append(status)

    sql = "SELECT * FROM requests"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp " + ("DESC" if newest_first else "ASC")
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    with closing(get_connection(db_path)) as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def delete_request(request_id, db_path=None):
    """Delete the row with this request_id. Returns how many rows were deleted (0 or 1)."""
    with closing(get_connection(db_path)) as conn:
        cursor = conn.execute("DELETE FROM requests WHERE request_id = ?", (request_id,))
        conn.commit()
        return cursor.rowcount


if __name__ == "__main__":
    print(f"Initialized {init_db()}")
