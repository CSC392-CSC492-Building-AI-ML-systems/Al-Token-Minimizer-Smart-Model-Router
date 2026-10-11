"""SQLite telemetry store: schema, init script and a small client.

One row per request in the `requests` table. Other modules should go through
`insert_request` / `fetch_requests` / `summarize_requests` and not write raw SQL.

Run `python db.py` to create the database file, and `python db.py summary` to
print totals, success rate and latency for what's stored (`--help` for filters).
"""

import argparse
import json
import math
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
STATUS_UPSTREAM_ERROR = "upstream_error"  # SDK raised an HTTP error (4xx/5xx, incl. 429) or couldn't read the response
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
    "http_status": "INTEGER",  # HTTP code for upstream_error (2xx if the body couldn't be read), else NULL
    "error_type": "TEXT",  # exception class, or Gemini's raw status for not_completed
    "error_message": "TEXT",  # Gemini's errors or the exception's message, never the response body
    "cost_usd": "REAL",  # NULL if the model has no price
}

# Token columns that summarize_requests adds up.
TOKEN_COLUMNS = (
    "input_tokens_raw",
    "input_tokens_compressed",
    "input_tokens_reported",
    "output_tokens",
    "thought_tokens",
    "cached_tokens",
    "total_tokens",
)
# Name in the summary -> latency column that summarize_requests reports stats for.
LATENCY_COLUMNS = {
    "total": "latency_total_ms",
    "upstream": "latency_upstream_ms",
    "count": "latency_count_ms",
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
    `request_id` and `timestamp` are filled in if missing. `timestamp` can be a
    datetime or an ISO 8601 string and is stored in UTC. `status` is required.
    Unknown keys, or a timestamp that isn't ISO 8601, raise ValueError.
    """
    if isinstance(record, RequestRecord):
        record = asdict(record)
    unknown = set(record) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown columns: {sorted(unknown)}")

    row = dict(record)
    row.setdefault("request_id", str(uuid.uuid4()))
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    if row["timestamp"] is not None:
        # since/until compare timestamps as text, so every row needs the same UTC format
        row["timestamp"] = _to_utc_iso(row["timestamp"])
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


def _to_utc_iso(value):
    """Turn a datetime or ISO 8601 string into the same UTC format as the timestamp column.

    Timestamps are compared as text, so a bound like "2026-10-08T09:00-04:00" has
    to be converted first. A value with no UTC offset is taken to be UTC already.
    Raises ValueError if a string isn't ISO 8601, or the time can't be shown in UTC.
    """
    if isinstance(value, str):
        # fromisoformat only accepts a trailing "Z" from Python 3.11 on
        value = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    try:
        return value.astimezone(timezone.utc).isoformat()
    except OverflowError:  # e.g. 0001-01-01T00:00+01:00 would be before year 1 in UTC
        raise ValueError(f"{value.isoformat()} is out of range") from None


def _where_clause(request_id=None, model=None, status=None, since=None, until=None):
    """The " WHERE ..." part (or "") and its params, for the filters that are set."""
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
    if since is not None:
        where.append("timestamp >= ?")
        params.append(_to_utc_iso(since))
    if until is not None:
        where.append("timestamp < ?")
        params.append(_to_utc_iso(until))
    return (" WHERE " + " AND ".join(where) if where else ""), params


def fetch_requests(
    db_path=None, model=None, status=None, limit=None, newest_first=False, request_id=None
):
    """Return rows as a list of dicts, oldest first unless newest_first is set."""
    where, params = _where_clause(request_id=request_id, model=model, status=status)
    sql = "SELECT * FROM requests" + where
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


def summarize_requests(db_path=None, model=None, status=None, since=None, until=None):
    """Add up the stored requests: counts per status, tokens, cost and latency.

    model/status filter like fetch_requests. since/until take a datetime or an
    ISO 8601 string (UTC if it has no offset); since is inclusive, until is not.

    Returns a dict that can go straight to JSON. Token, cost and latency numbers
    only use the rows where that column was recorded, and are None when no row
    has it (e.g. cost_usd while no model has a price). "by_model" has the same
    numbers for each model, sorted by name, with rows that have no model last.
    """
    where, params = _where_clause(model=model, status=status, since=since, until=until)
    columns = ["model", "timestamp", "status", "cost_usd", *TOKEN_COLUMNS, *LATENCY_COLUMNS.values()]
    sql = f"SELECT {', '.join(columns)} FROM requests{where}"

    overall, per_model = _Totals(), {}
    with closing(get_connection(db_path)) as conn:
        for row in conn.execute(sql, params):
            overall.add(row)
            per_model.setdefault(row["model"], _Totals()).add(row)

    summary = overall.summary()
    summary["by_model"] = [
        {"model": name, **totals.summary()}
        for name, totals in sorted(per_model.items(), key=lambda item: (item[0] is None, item[0] or ""))
    ]
    return summary


class _Totals:
    """Running totals for one group of rows (all of them, or one model's)."""

    def __init__(self):
        self.requests = 0
        self.first_timestamp = None
        self.last_timestamp = None
        self.status_counts = dict.fromkeys(STATUSES, 0)
        self.tokens = dict.fromkeys(TOKEN_COLUMNS)  # stays None until a row has the column
        self.raw_tokens_compressed = None  # raw input tokens of rows that also have a compressed count
        self.tokens_saved = None  # raw - compressed, over those same rows
        self.cost = None
        self.priced_requests = 0
        self.latencies = {name: [] for name in LATENCY_COLUMNS}

    def add(self, row):
        self.requests += 1
        timestamp = row["timestamp"]
        if self.first_timestamp is None or timestamp < self.first_timestamp:
            self.first_timestamp = timestamp
        if self.last_timestamp is None or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp
        self.status_counts[row["status"]] = self.status_counts.get(row["status"], 0) + 1

        for column in TOKEN_COLUMNS:
            self.tokens[column] = _add(self.tokens[column], row[column])
        raw, compressed = row["input_tokens_raw"], row["input_tokens_compressed"]
        if raw is not None and compressed is not None:
            self.raw_tokens_compressed = _add(self.raw_tokens_compressed, raw)
            self.tokens_saved = _add(self.tokens_saved, raw - compressed)

        if row["cost_usd"] is not None:
            self.cost = _add(self.cost, row["cost_usd"])
            self.priced_requests += 1

        for name, column in LATENCY_COLUMNS.items():
            if row[column] is not None:
                self.latencies[name].append(row[column])

    def summary(self):
        return {
            "requests": self.requests,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "status_counts": dict(self.status_counts),
            "success_rate": _ratio(self.status_counts[STATUS_OK], self.requests),
            "tokens": dict(self.tokens),
            "compression": {
                "tokens_saved": self.tokens_saved,
                "saved_rate": _ratio(self.tokens_saved, self.raw_tokens_compressed),
            },
            "cost_usd": {
                "total": self.cost,
                "avg_per_priced_request": _ratio(self.cost, self.priced_requests),
                "unpriced_requests": self.requests - self.priced_requests,
            },
            "latency_ms": {name: _latency_stats(values) for name, values in self.latencies.items()},
        }


def _add(total, value):
    """total + value, where None means nothing recorded (so None + None stays None)."""
    if value is None:
        return total
    return value if total is None else total + value


def _ratio(part, whole):
    return part / whole if part is not None and whole else None


def _latency_stats(values):
    """avg, p50, p95 and max of a list of latencies, all None if it's empty."""
    if not values:
        return {"avg": None, "p50": None, "p95": None, "max": None}
    values = sorted(values)
    return {
        "avg": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": values[-1],
    }


def _percentile(sorted_values, pct):
    """pct-th percentile of a sorted, non-empty list, interpolating between the two
    nearest values (same as numpy's default and statistics.quantiles(method="inclusive"))."""
    rank = (len(sorted_values) - 1) * pct / 100
    low = math.floor(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def format_summary(summary):
    """Plain-text version of a summarize_requests() result, for the CLI."""
    if not summary["requests"]:
        return "No requests stored (for these filters)."

    cost, compression = summary["cost_usd"], summary["compression"]
    statuses = ", ".join(f"{name} {count:,}" for name, count in summary["status_counts"].items() if count)
    lines = [
        f"requests      {summary['requests']:,}  "
        f"({summary['first_timestamp']} to {summary['last_timestamp']})",
        f"success rate  {_fmt_pct(summary['success_rate'])}  ({statuses})",
        f"cost          {_fmt_usd(cost['total'])} total, "
        f"{_fmt_usd(cost['avg_per_priced_request'])} avg per priced request, "
        f"{cost['unpriced_requests']:,} unpriced",
        f"compression   {_fmt_num(compression['tokens_saved'])} input tokens saved "
        f"({_fmt_pct(compression['saved_rate'])})",
        "",
        "tokens",
    ]
    lines += [f"  {column:<25}{_fmt_num(total):>12}" for column, total in summary["tokens"].items()]
    lines += ["", f"{'latency (ms)':<12}{'avg':>12}{'p50':>12}{'p95':>12}{'max':>12}"]
    for name, stats in summary["latency_ms"].items():
        lines.append(f"  {name:<10}" + "".join(f"{_fmt_num(stats[key], 1):>12}" for key in ("avg", "p50", "p95", "max")))
    lines += ["", "by model"]
    for group in summary["by_model"]:
        lines.append(
            f"  {group['model'] or '(no model)'}: {group['requests']:,} "
            f"request{'' if group['requests'] == 1 else 's'}, "
            f"{_fmt_pct(group['success_rate'])} ok, "
            f"{_fmt_num(group['tokens']['total_tokens'])} total tokens, "
            f"{_fmt_usd(group['cost_usd']['total'])}, "
            f"p95 {_fmt_num(group['latency_ms']['total']['p95'], 1)} ms"
        )
    return "\n".join(lines)


def _fmt_num(value, decimals=0):
    return "-" if value is None else f"{value:,.{decimals}f}"


def _fmt_pct(value):
    return "-" if value is None else f"{value:.1%}"


def _fmt_usd(value):
    return "-" if value is None else f"${value:,.6f}"


def _parse_time(text):
    """argparse type for --since/--until."""
    try:
        return _to_utc_iso(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} isn't an ISO 8601 time, try e.g. 2026-10-08 or 2026-10-08T14:30"
        ) from None


def cli(argv=None):
    """`python db.py` creates the db; `python db.py summary [filters]` prints a summary."""
    parser = argparse.ArgumentParser(description="Create the telemetry db, or summarize what's in it.")
    commands = parser.add_subparsers(dest="command")
    summary_parser = commands.add_parser(
        "summary", help="print totals, success rate and latency for the stored requests"
    )
    summary_parser.add_argument("--model", help="only requests to this model")
    summary_parser.add_argument("--status", choices=STATUSES, help="only requests with this status")
    summary_parser.add_argument(
        "--since", type=_parse_time, help="from this time, inclusive (ISO 8601, UTC if no offset)"
    )
    summary_parser.add_argument("--until", type=_parse_time, help="up to this time, not inclusive")
    summary_parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    path = _resolve_path()
    if args.command != "summary":
        print(f"Initialized {init_db(path)}")
        return
    if not path.exists():
        init_db(path)  # so `summary` works on a fresh clone; an existing db is only read
    summary = summarize_requests(path, model=args.model, status=args.status, since=args.since, until=args.until)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"db            {path}\n{format_summary(summary)}")


if __name__ == "__main__":
    cli()
