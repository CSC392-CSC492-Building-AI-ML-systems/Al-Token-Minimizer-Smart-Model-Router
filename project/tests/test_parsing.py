"""Check /ask saves a proper db row for good replies and for failures.

Gemini is faked (the good reply copies the shape of a real gemini-3.5-flash
response), so no real API calls are made. Uses a temp db.

Run from the project folder:
    uv run --with pytest pytest tests/test_parsing.py -v
"""

import json
import os
from types import SimpleNamespace

import httpx
import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

from fastapi.testclient import TestClient  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

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


# The tests below go through the real SDK, with only Gemini's HTTP API faked, so
# they get the exact objects and errors the SDK hands main.py (it never lets
# httpx's own errors out, and it returns a plain dict for a reply that doesn't
# match its schema).


def real_body(**overrides):
    """A gemini-3.5-flash reply as it comes over the wire."""
    body = {
        "id": "v1_abc",
        "model": "gemini-3.5-flash",
        "status": "completed",
        "steps": [
            {"type": "user_input", "content": [{"type": "text", "text": "hello"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "Hi, nice to meet you."}]},
        ],
        "usage": {"total_input_tokens": 7, "total_output_tokens": 7, "total_tokens": 401},
    }
    body.update(overrides)
    return body


def fake_gemini_http(monkeypatch, handler):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    sdk = genai.Client(api_key="fake", http_options=types.HttpOptions(httpx_client=http))
    monkeypatch.setattr(main, "client", sdk)


def test_real_sdk_good_reply(monkeypatch, client):
    fake_gemini_http(monkeypatch, lambda request: httpx.Response(200, json=real_body()))

    response = client.get("/ask", params={"prompt": "hello"})
    [row] = db.fetch_requests()

    assert response.json() == {"text": "Hi, nice to meet you."}
    assert row["status"] == db.STATUS_OK
    assert row["interaction_id"] == "v1_abc"
    assert row["total_tokens"] == 401


def test_real_sdk_timeout(monkeypatch, client):
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    fake_gemini_http(monkeypatch, handler)
    row = ask_and_get_row(client)
    assert row["status"] == db.STATUS_TIMEOUT
    assert row["error_type"] == "APITimeoutError"


def test_real_sdk_cant_connect(monkeypatch, client):
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    fake_gemini_http(monkeypatch, handler)
    row = ask_and_get_row(client)
    assert row["status"] == db.STATUS_CONNECTION_ERROR
    assert row["error_type"] == "APIConnectionError"
    assert row["error_message"] == "no route"


def test_real_sdk_http_error_keeps_geminis_message(monkeypatch, client):
    error = {"error": {"code": 400, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"}}
    fake_gemini_http(monkeypatch, lambda request: httpx.Response(400, json=error))
    row = ask_and_get_row(client)
    assert row["status"] == db.STATUS_UPSTREAM_ERROR
    assert row["http_status"] == 400
    assert row["error_type"] == "BadRequestError"
    assert row["error_message"] == "INVALID_ARGUMENT: Request contains an invalid argument."


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, text=json.dumps(real_body()), headers={"content-type": "text/plain"}),
        httpx.Response(200, text=json.dumps(real_body())[:-5], headers={"content-type": "application/json"}),
        httpx.Response(201, json=real_body()),
    ],
    ids=["not-json-content-type", "cut-off-json", "201"],
)
def test_real_sdk_reply_it_cant_read_isnt_stored(monkeypatch, client, reply):
    # the SDK's error message has the whole reply (prompt and answer) in it
    fake_gemini_http(monkeypatch, lambda request: reply)
    row = ask_and_get_row(client)
    assert row["status"] == db.STATUS_UPSTREAM_ERROR
    assert row["http_status"] == reply.status_code
    assert row["error_message"] is None
    stored = " ".join(str(v) for v in row.values())
    assert "hello" not in stored
    assert "nice to meet you" not in stored


def test_real_sdk_reply_without_status(monkeypatch, client):
    body = real_body()
    del body["status"]
    fake_gemini_http(monkeypatch, lambda request: httpx.Response(200, json=body))

    response = client.get("/ask", params={"prompt": "hello"})
    [row] = db.fetch_requests()

    assert response.status_code == 200
    assert row["status"] == db.STATUS_NOT_COMPLETED
    assert row["error_type"] is None
    assert row["interaction_id"] == "v1_abc"
    assert row["total_tokens"] == 401  # still read from the dict


def test_real_sdk_reply_without_status_or_text(monkeypatch, client):
    body = real_body(steps=real_body()["steps"][:1])  # only the user's input, no answer
    del body["status"]
    fake_gemini_http(monkeypatch, lambda request: httpx.Response(200, json=body))

    response = client.get("/ask", params={"prompt": "hello"})

    assert response.json() == {"text": ""}  # same as for a typed reply with no text


TOKEN_FIELDS = {  # Gemini's usage field -> column
    "total_input_tokens": "input_tokens_reported",
    "total_output_tokens": "output_tokens",
    "total_thought_tokens": "thought_tokens",
    "total_cached_tokens": "cached_tokens",
    "total_tokens": "total_tokens",
}


@pytest.mark.parametrize("field, column", list(TOKEN_FIELDS.items()))
def test_real_sdk_token_count_that_isnt_a_number(monkeypatch, client, field, column):
    usage = {**dict.fromkeys(TOKEN_FIELDS, 7), field: "lots"}
    fake_gemini_http(monkeypatch, lambda request: httpx.Response(200, json=real_body(usage=usage)))

    response = client.get("/ask", params={"prompt": "hello"})
    [row] = db.fetch_requests()

    assert response.json() == {"text": "Hi, nice to meet you."}
    assert row["status"] == db.STATUS_OK
    assert row[column] is None  # not stored, the summary adds these up
    assert all(row[other] == 7 for other in TOKEN_FIELDS.values() if other != column)
    assert db.summarize_requests()["tokens"][column] is None


@pytest.mark.parametrize("bad_id", [["v1_abc"], {"id": "v1_abc"}, 5])
def test_real_sdk_id_that_isnt_a_string(monkeypatch, client, bad_id):
    # sqlite can't store a list or dict, which would lose the whole row
    fake_gemini_http(monkeypatch, lambda request: httpx.Response(200, json=real_body(id=bad_id)))

    response = client.get("/ask", params={"prompt": "hello"})
    [row] = db.fetch_requests()

    assert response.status_code == 200
    assert row["status"] == db.STATUS_OK
    assert row["interaction_id"] is None
    assert row["total_tokens"] == 401


@pytest.mark.parametrize(
    "errors, message",
    [(5, "5"), ({"code": "RESOURCE_EXHAUSTED", "message": "quota"}, "RESOURCE_EXHAUSTED: quota")],
    ids=["number", "one-error-not-in-a-list"],
)
def test_real_sdk_errors_that_arent_a_list(monkeypatch, client, errors, message):
    # not a list, so the SDK hands back a dict
    fake_gemini_http(monkeypatch, lambda request: httpx.Response(200, json=real_body(errors=errors)))

    response = client.get("/ask", params={"prompt": "hello"})
    [row] = db.fetch_requests()

    assert response.status_code == 200
    assert row["status"] == db.STATUS_NOT_COMPLETED
    assert row["error_message"] == message
