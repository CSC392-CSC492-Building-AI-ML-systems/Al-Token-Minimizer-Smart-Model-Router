"""Check /ask saves a proper db row for good replies and for failures.

Gemini is faked (the good reply copies the shape of a real gemini-3.5-flash
response), so no real API calls are made. Uses a temp db.

Run from the project folder:
    uv run --with pytest pytest tests/test_parsing.py -v
"""

import os
from types import SimpleNamespace

import httpx
import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

from fastapi.testclient import TestClient  # noqa: E402

import db  # noqa: E402
import main  # noqa: E402


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def real_shaped_reply(**overrides):
    reply = SimpleNamespace(
        id="v1_abc",
        model="gemini-3.5-flash",
        status="completed",
        errors=None,
        output_text="Hi, nice to meet you.",
        usage=SimpleNamespace(
            total_input_tokens=7,
            total_output_tokens=7,
            total_thought_tokens=387,
            total_cached_tokens=0,
            total_tokens=401,
        ),
    )
    for key, value in overrides.items():
        setattr(reply, key, value)
    return reply


def fake_gemini(monkeypatch, reply=None, error=None):
    def create(model, input):
        if error:
            raise error
        return reply

    monkeypatch.setattr(main.client.interactions, "create", create)


def ask_and_get_row(client):
    client.get("/ask", params={"prompt": "hello"})
    [row] = db.fetch_requests()
    return row


def test_good_reply_saves_a_full_row(monkeypatch, client):
    fake_gemini(monkeypatch, reply=real_shaped_reply())

    response = client.get("/ask", params={"prompt": "hello"})
    [row] = db.fetch_requests()

    assert response.json() == {"text": "Hi, nice to meet you."}
    assert row["status"] == db.STATUS_OK
    assert row["endpoint"] == "/ask"
    assert row["model"] == "gemini-3.5-flash"
    assert row["interaction_id"] == "v1_abc"
    assert row["input_tokens_reported"] == 7
    assert row["output_tokens"] == 7
    assert row["thought_tokens"] == 387
    assert row["cached_tokens"] == 0
    assert row["total_tokens"] == 401
    assert row["latency_upstream_ms"] is not None
    assert row["latency_total_ms"] >= row["latency_upstream_ms"]
    assert row["error_type"] is None and row["http_status"] is None


def test_prompt_and_reply_text_are_not_stored(monkeypatch, client):
    fake_gemini(monkeypatch, reply=real_shaped_reply())
    row = ask_and_get_row(client)
    stored = " ".join(str(v) for v in row.values())
    assert "hello" not in stored
    assert "nice to meet you" not in stored


def test_missing_usage_does_not_break_the_row(monkeypatch, client):
    fake_gemini(monkeypatch, reply=real_shaped_reply(usage=None))
    row = ask_and_get_row(client)
    assert row["status"] == db.STATUS_OK
    assert row["output_tokens"] is None


def test_not_completed_reply(monkeypatch, client):
    fake_gemini(monkeypatch, reply=real_shaped_reply(status="incomplete"))
    row = ask_and_get_row(client)
    assert row["status"] == db.STATUS_NOT_COMPLETED
    assert row["error_type"] == "incomplete"


def test_reply_with_errors_is_not_ok(monkeypatch, client):
    errors = [SimpleNamespace(code="oops", message="something broke")]
    fake_gemini(monkeypatch, reply=real_shaped_reply(errors=errors))
    row = ask_and_get_row(client)
    assert row["status"] == db.STATUS_NOT_COMPLETED
    assert "something broke" in row["error_message"]


class FakeHttpError(Exception):
    """Same attributes the SDK's GenAiError has."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@pytest.mark.parametrize("code", [429, 503])
def test_http_error_saves_status_and_code(monkeypatch, client, code):
    fake_gemini(monkeypatch, error=FakeHttpError(code, "try again later"))
    row = ask_and_get_row(client)
    assert row["status"] == db.STATUS_UPSTREAM_ERROR
    assert row["http_status"] == code
    assert row["error_type"] == "FakeHttpError"
    assert row["latency_upstream_ms"] is not None


def test_timeout(monkeypatch, client):
    fake_gemini(monkeypatch, error=httpx.ReadTimeout("timed out"))
    assert ask_and_get_row(client)["status"] == db.STATUS_TIMEOUT


def test_cant_connect(monkeypatch, client):
    fake_gemini(monkeypatch, error=httpx.ConnectError("no route"))
    assert ask_and_get_row(client)["status"] == db.STATUS_CONNECTION_ERROR


def test_anything_else_is_internal_error(monkeypatch, client):
    fake_gemini(monkeypatch, error=RuntimeError("bug"))
    assert ask_and_get_row(client)["status"] == db.STATUS_INTERNAL_ERROR


def test_db_failure_does_not_break_the_answer(monkeypatch, client):
    fake_gemini(monkeypatch, reply=real_shaped_reply())

    def broken_insert(record):
        raise RuntimeError("disk full")

    monkeypatch.setattr(main.db, "insert_request", broken_insert)

    response = client.get("/ask", params={"prompt": "hello"})
    assert response.status_code == 200
    assert response.json() == {"text": "Hi, nice to meet you."}
