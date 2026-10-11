"""Check db.summarize_requests, `python db.py summary` and GET /telemetry/summary.

Every test gets its own temporary db, so the real telemetry.db is never touched.

Run from the project folder:
    uv run --with pytest pytest tests/test_summary.py -v
"""

import json
import os
import statistics
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

from fastapi.testclient import TestClient  # noqa: E402

import db  # noqa: E402
import main  # noqa: E402

START = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
NO_LATENCY = {"avg": None, "p50": None, "p95": None, "max": None}


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A fresh db, also used by default (TELEMETRY_DB_PATH) by db.py and main.py."""
    path = tmp_path / "telemetry.db"
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(path))
    db.init_db(path)
    return path


def add(db_path, minute=0, **fields):
    """Insert a row `minute` minutes after START. Defaults to an ok gemini-3.5-flash request."""
    fields.setdefault("status", db.STATUS_OK)
    fields.setdefault("model", "gemini-3.5-flash")
    timestamp = (START + timedelta(minutes=minute)).isoformat()
    return db.insert_request(db.RequestRecord(timestamp=timestamp, **fields), db_path)


def test_summary_columns_exist():
    assert set(db.TOKEN_COLUMNS) <= set(db.COLUMNS)
    assert set(db.LATENCY_COLUMNS.values()) <= set(db.COLUMNS)


def test_empty_db(db_path):
    summary = db.summarize_requests(db_path)

    assert summary["requests"] == 0
    assert summary["first_timestamp"] is None
    assert summary["last_timestamp"] is None
    assert summary["status_counts"] == dict.fromkeys(db.STATUSES, 0)
    assert summary["success_rate"] is None
    assert summary["tokens"] == dict.fromkeys(db.TOKEN_COLUMNS)
    assert summary["compression"] == {"tokens_saved": None, "saved_rate": None}
    assert summary["cost_usd"] == {
        "total": None,
        "avg_per_priced_request": None,
        "unpriced_requests": 0,
    }
    assert summary["latency_ms"] == dict.fromkeys(db.LATENCY_COLUMNS, NO_LATENCY)
    assert summary["by_model"] == []


def test_totals(db_path):
    # inserted out of time order, like requests that finish in a different order than they start
    add(db_path, 3, status=db.STATUS_UPSTREAM_ERROR, http_status=429)
    add(db_path, 0, input_tokens_raw=100, input_tokens_compressed=80, input_tokens_reported=90,
        output_tokens=10, total_tokens=100, cost_usd=0.001,
        latency_total_ms=100.0, latency_upstream_ms=90.0)
    # no compressed count, so it's left out of the compression numbers
    add(db_path, 2, status=db.STATUS_TIMEOUT, input_tokens_raw=50,
        latency_total_ms=1000.0, latency_upstream_ms=900.0)
    add(db_path, 1, input_tokens_raw=200, input_tokens_compressed=150, input_tokens_reported=180,
        output_tokens=20, total_tokens=200, cost_usd=0.003,
        latency_total_ms=300.0, latency_upstream_ms=250.0)

    summary = db.summarize_requests(db_path)

    assert summary["requests"] == 4
    assert summary["first_timestamp"] == START.isoformat()
    assert summary["last_timestamp"] == (START + timedelta(minutes=3)).isoformat()
    assert summary["status_counts"] == {
        **dict.fromkeys(db.STATUSES, 0),
        db.STATUS_OK: 2,
        db.STATUS_TIMEOUT: 1,
        db.STATUS_UPSTREAM_ERROR: 1,
    }
    assert summary["success_rate"] == 0.5
    assert summary["tokens"] == {
        "input_tokens_raw": 350,
        "input_tokens_compressed": 230,
        "input_tokens_reported": 270,
        "output_tokens": 30,
        "thought_tokens": None,  # never recorded
        "cached_tokens": None,
        "total_tokens": 300,
    }
    assert summary["compression"]["tokens_saved"] == 70  # (100 - 80) + (200 - 150)
    assert summary["compression"]["saved_rate"] == pytest.approx(70 / 300)
    assert summary["cost_usd"]["total"] == pytest.approx(0.004)
    assert summary["cost_usd"]["avg_per_priced_request"] == pytest.approx(0.002)
    assert summary["cost_usd"]["unpriced_requests"] == 2

    total = summary["latency_ms"]["total"]
    assert total["avg"] == pytest.approx(1400 / 3)
    assert total["p50"] == 300.0
    assert total["p95"] == pytest.approx(930.0)  # 90% of the way from 300 to 1000
    assert total["max"] == 1000.0
    assert summary["latency_ms"]["upstream"] == {
        "avg": pytest.approx(1240 / 3),
        "p50": 250.0,
        "p95": pytest.approx(835.0),  # 90% of the way from 250 to 900
        "max": 900.0,
    }
    assert summary["latency_ms"]["count"] == NO_LATENCY  # not recorded yet (CSC-6)


def test_success_rate_only_counts_ok(db_path):
    # not_completed (Gemini answered but didn't finish) is a failure too, like the errors
    for minute, status in enumerate(db.STATUSES):
        add(db_path, minute, status=status)

    summary = db.summarize_requests(db_path)

    assert summary["status_counts"] == dict.fromkeys(db.STATUSES, 1)
    assert summary["success_rate"] == pytest.approx(1 / len(db.STATUSES))


@pytest.mark.parametrize(
    "values",
    [[42.0], [10.0, 20.0], [5.0, 1.0, 3.0, 2.0, 4.0], [float(v * v % 97) for v in range(1, 60)]],
)
def test_percentiles_match_statistics_module(values):
    stats = db._latency_stats(values)
    if len(values) == 1:
        assert stats == {"avg": 42.0, "p50": 42.0, "p95": 42.0, "max": 42.0}
        return
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    assert stats["p50"] == pytest.approx(cuts[49])
    assert stats["p95"] == pytest.approx(cuts[94])
    assert stats["avg"] == pytest.approx(statistics.mean(values))
    assert stats["max"] == max(values)


def test_by_model(db_path):
    add(db_path, 0, model="b-model", cost_usd=0.5)
    add(db_path, 1, model="a-model", status=db.STATUS_TIMEOUT)
    add(db_path, 2, model="a-model", latency_total_ms=20.0)
    add(db_path, 3, model=None)

    summary = db.summarize_requests(db_path)
    groups = summary["by_model"]

    assert [g["model"] for g in groups] == ["a-model", "b-model", None]  # no model goes last
    assert [g["requests"] for g in groups] == [2, 1, 1]
    assert sum(g["requests"] for g in groups) == summary["requests"]
    a_model, b_model, _ = groups
    assert a_model["success_rate"] == 0.5
    assert a_model["latency_ms"]["total"]["max"] == 20.0
    assert a_model["cost_usd"]["total"] is None
    assert b_model["cost_usd"]["total"] == 0.5
    assert "by_model" not in a_model


def test_model_and_status_filters(db_path):
    add(db_path, 0, model="a-model")
    add(db_path, 1, model="a-model", status=db.STATUS_TIMEOUT)
    add(db_path, 2, model="b-model")

    assert db.summarize_requests(db_path, model="a-model")["requests"] == 2
    assert db.summarize_requests(db_path, status=db.STATUS_OK)["requests"] == 2
    only = db.summarize_requests(db_path, model="a-model", status=db.STATUS_TIMEOUT)
    assert only["requests"] == 1
    assert only["success_rate"] == 0.0
    assert db.summarize_requests(db_path, model="nope")["requests"] == 0


@pytest.mark.parametrize(
    "since, until, expected_minutes",
    [
        (START + timedelta(minutes=1), None, [1, 2]),  # since is inclusive
        (None, START + timedelta(minutes=1), [0]),  # until is not
        (START + timedelta(minutes=1), START + timedelta(minutes=2), [1]),
        ("2026-10-08T12:01:00+00:00", None, [1, 2]),
        ("2026-10-08T12:01:00Z", None, [1, 2]),
        ("2026-10-08T08:01-04:00", None, [1, 2]),  # same instant, different offset
        ("2026-10-08T12:00:30", None, [1, 2]),  # no offset means UTC
        (datetime(2026, 10, 8, 12, 1), None, [1, 2]),  # naive datetime means UTC
        (datetime(2026, 10, 8, 8, 1, tzinfo=timezone(timedelta(hours=-4))), None, [1, 2]),
        ("2026-10-08", "2026-10-09", [0, 1, 2]),
        ("2026-10-09", None, []),
    ],
)
def test_time_filters(db_path, since, until, expected_minutes):
    for minute in range(3):
        add(db_path, minute, output_tokens=10 ** minute)  # tokens tell us which rows were counted

    summary = db.summarize_requests(db_path, since=since, until=until)

    assert summary["requests"] == len(expected_minutes)
    expected_tokens = sum(10 ** m for m in expected_minutes) if expected_minutes else None
    assert summary["tokens"]["output_tokens"] == expected_tokens


def test_time_filter_with_microseconds(db_path):
    # isoformat() drops ".000000", so make sure text comparison still orders these right
    db.insert_request({"timestamp": "2026-10-08T12:00:00+00:00", "status": db.STATUS_OK}, db_path)
    db.insert_request({"timestamp": "2026-10-08T12:00:00.500000+00:00", "status": db.STATUS_OK}, db_path)

    # 3 fraction digits, since Python 3.10's fromisoformat only takes 3 or 6
    assert db.summarize_requests(db_path, since="2026-10-08T12:00:00.250")["requests"] == 1
    assert db.summarize_requests(db_path, until="2026-10-08T12:00:00.250")["requests"] == 1
    assert db.summarize_requests(db_path, since="2026-10-08T12:00:00")["requests"] == 2


@pytest.mark.parametrize(
    "bad_time",
    ["last tuesday", "0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"],  # last two overflow UTC
)
def test_bad_time_raises(db_path, bad_time):
    with pytest.raises(ValueError):
        db.summarize_requests(db_path, since=bad_time)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-10-08T12:00:00Z",
        "2026-10-08T08:00:00-04:00",
        "2026-10-08T12:00:00",  # no offset means UTC
        datetime(2026, 10, 8, 14, 0, tzinfo=timezone(timedelta(hours=2))),
    ],
)
def test_timestamps_are_stored_in_utc(db_path, timestamp):
    # since/until compare as text, so every row has to be stored in the same format
    db.insert_request({"timestamp": timestamp, "status": db.STATUS_OK}, db_path)

    [row] = db.fetch_requests(db_path)
    assert row["timestamp"] == "2026-10-08T12:00:00+00:00"
    assert db.summarize_requests(db_path, since="2026-10-08T12:00:00Z")["requests"] == 1
    assert db.summarize_requests(db_path, until="2026-10-08T12:00:00.500Z")["requests"] == 1


def test_bad_timestamp_isnt_stored(db_path):
    with pytest.raises(ValueError):
        db.insert_request({"timestamp": "yesterday", "status": db.STATUS_OK}, db_path)
    assert db.fetch_requests(db_path) == []


def test_fetch_requests_filters_still_work(db_path):
    first = add(db_path, 0, model="a-model")
    second = add(db_path, 1, model="a-model", status=db.STATUS_TIMEOUT)
    add(db_path, 2, model="b-model")

    assert [r["request_id"] for r in db.fetch_requests(db_path, model="a-model")] == [first, second]
    assert [r["request_id"] for r in db.fetch_requests(db_path, status=db.STATUS_TIMEOUT)] == [second]
    assert [r["request_id"] for r in db.fetch_requests(db_path, request_id=second)] == [second]
    newest = db.fetch_requests(db_path, model="a-model", newest_first=True, limit=1)
    assert [r["request_id"] for r in newest] == [second]


def add_filter_rows(db_path):
    """Rows that each filter in FILTERS is the only one to drop, plus one row that passes all of them."""
    add(db_path, 0, model="b-model", output_tokens=1)  # before since
    add(db_path, 1, model="b-model", status=db.STATUS_TIMEOUT, output_tokens=10)  # wrong status
    add(db_path, 2, model="b-model", output_tokens=100)  # the one that passes
    add(db_path, 2, model="a-model", output_tokens=1000)  # wrong model
    add(db_path, 3, model="b-model", output_tokens=10000)  # at until, which isn't inclusive


FILTERS = {"model": "b-model", "status": "ok", "since": "2026-10-08T08:01:00-04:00", "until": "2026-10-08T12:03:00Z"}
FILTERED_TOKENS = 100  # output_tokens of the row that passes


# --- CLI ---


def test_cli_json_matches_function(db_path, capsys):
    add(db_path, 0, latency_total_ms=12.5, cost_usd=0.01)
    add(db_path, 1, model="b-model", status=db.STATUS_TIMEOUT)

    db.cli(["summary", "--json"])

    assert json.loads(capsys.readouterr().out) == db.summarize_requests(db_path)


def test_cli_text(db_path, capsys):
    add(db_path, 0, latency_total_ms=12.5, cost_usd=0.01, input_tokens_raw=10, input_tokens_compressed=8,
        total_tokens=1234)
    add(db_path, 1, model=None, status=db.STATUS_TIMEOUT, latency_total_ms=100.0)
    add(db_path, 2, latency_total_ms=1000.0)

    db.cli(["summary"])
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == f"db            {db_path}"
    assert "requests      3  (2026-10-08T12:00:00+00:00 to 2026-10-08T12:02:00+00:00)" in lines
    assert "success rate  66.7%  (ok 2, timeout 1)" in lines
    assert "cost          $0.010000 total, $0.010000 avg per priced request, 2 unpriced" in lines
    assert "compression   2 input tokens saved (20.0%)" in lines
    assert "  total_tokens                    1,234" in lines
    header = lines.index("latency (ms)         avg         p50         p95         max")
    assert lines[header + 1] == "  total            370.8       100.0       910.0     1,000.0"
    for label in ("avg", "p50", "p95", "max"):  # each header ends where its numbers end
        end = lines[header].index(label) + len(label)
        assert lines[header + 1][end - 1].isdigit()
        assert lines[header + 1][end:end + 1] in ("", " ")
    assert "  gemini-3.5-flash: 2 requests, 100.0% ok, 1,234 total tokens, $0.010000, p95 950.6 ms" in lines
    assert "  (no model): 1 request, 0.0% ok, - total tokens, -, p95 100.0 ms" in lines


def test_cli_filters(db_path, capsys):
    add_filter_rows(db_path)

    db.cli(["summary", "--json", "--model", FILTERS["model"], "--status", FILTERS["status"],
            "--since", FILTERS["since"], "--until", FILTERS["until"]])

    summary = json.loads(capsys.readouterr().out)
    assert summary["requests"] == 1
    assert summary["tokens"]["output_tokens"] == FILTERED_TOKENS


def test_cli_empty(db_path, capsys):
    db.cli(["summary"])
    assert "No requests stored" in capsys.readouterr().out


def test_cli_summary_creates_missing_db(tmp_path, monkeypatch, capsys):
    path = tmp_path / "new" / "telemetry.db"
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(path))

    db.cli(["summary", "--json"])

    assert json.loads(capsys.readouterr().out)["requests"] == 0
    assert path.exists()


def test_cli_no_args_still_initializes(tmp_path, monkeypatch, capsys):
    path = tmp_path / "telemetry.db"
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(path))

    db.cli([])

    assert capsys.readouterr().out.strip() == f"Initialized {path}"
    assert db.fetch_requests(path) == []


@pytest.mark.parametrize(
    "bad_args",
    [["--since", "last tuesday"], ["--until", "0001-01-01T00:00+01:00"], ["--status", "great"]],
)
def test_cli_rejects_bad_args(db_path, capsys, bad_args):
    with pytest.raises(SystemExit) as exit_info:
        db.cli(["summary", *bad_args])
    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err


# --- GET /telemetry/summary ---


@pytest.fixture
def client():
    return TestClient(main.app)


def test_endpoint_matches_function(db_path, client):
    add(db_path, 0, latency_total_ms=12.5, cost_usd=0.01, total_tokens=30)
    add(db_path, 1, model=None, status=db.STATUS_UPSTREAM_ERROR, http_status=500)

    response = client.get("/telemetry/summary")

    assert response.status_code == 200
    assert response.json() == db.summarize_requests(db_path)


def test_endpoint_filters(db_path, client):
    add_filter_rows(db_path)

    response = client.get("/telemetry/summary", params=FILTERS)

    assert response.status_code == 200
    assert response.json()["requests"] == 1
    assert response.json()["tokens"]["output_tokens"] == FILTERED_TOKENS


@pytest.mark.parametrize(
    "params",
    [
        {"status": "great"},
        {"status": "okay"},  # has "ok" in it, but isn't a status
        {"since": "last tuesday"},
        {"until": "soon"},
        {"since": "0001-01-01T00:00:00+01:00"},  # a real time, but before year 1 in UTC
        {"until": "9999-12-31T23:59:59-01:00"},
    ],
)
def test_endpoint_rejects_bad_params(db_path, client, params):
    assert client.get("/telemetry/summary", params=params).status_code == 422
