"""Check POST /ask, the compression pipeline hook and the db migration for its columns.

Gemini is faked, so no real API calls are made. Uses a temp db.

Run from the project folder:
    uv run --with pytest pytest tests/test_post_ask.py -v
"""

import os
import sqlite3
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

from fastapi.testclient import TestClient  # noqa: E402

import db  # noqa: E402
import main  # noqa: E402
from compressor import pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


@pytest.fixture(autouse=True)
def fake_token_counter(monkeypatch):
    monkeypatch.setattr(main.model_package.tokenizer, "count", lambda text: len(text.split()))


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def ok_reply(text="hi"):
    return SimpleNamespace(
        id="v1_abc",
        status="completed",
        model="gemini-3.5-flash",
        output_text=text,
        errors=None,
        usage=SimpleNamespace(
            total_input_tokens=12,
            total_output_tokens=3,
            total_thought_tokens=20,
            total_cached_tokens=None,
            total_tokens=35,
        ),
    )


class RateLimited(Exception):
    status_code = 429


# ---------------------------------------------------------------- POST /ask


def test_post_ask_returns_text_and_saves_row(monkeypatch, client):
    sent = {}

    def fake_create(**kwargs):
        sent.update(kwargs)
        return ok_reply("Canberra")

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    response = client.post("/ask", json={"prompt": "capital?", "system": "be brief"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Canberra"
    assert sent == {
        "model": "gemini-3.5-flash",
        "input": "capital?",
        "system_instruction": "be brief",
        "generation_config": {"max_output_tokens": main.DEFAULT_MAX_OUTPUT_TOKENS},
    }
    [row] = db.fetch_requests(request_id=body["request_id"])
    assert row["status"] == db.STATUS_OK
    assert row["compression"] == "none"
    assert row["latency_compress_ms"] >= 0
    assert row["latency_count_ms"] >= 0
    assert row["input_tokens_raw"] == 1
    assert row["input_tokens_compressed"] == 1
    assert row["input_tokens_reported"] == 12


def test_post_ask_without_system_doesnt_send_one(monkeypatch, client):
    sent = {}
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: sent.update(kw) or ok_reply())

    client.post("/ask", json={"prompt": "hello"})

    assert "system_instruction" not in sent


def test_post_ask_rejects_unknown_stages(client):
    response = client.post("/ask", json={"prompt": "hello", "stages": ["zip"]})

    assert response.status_code == 422
    assert db.fetch_requests() == []


def test_post_ask_failure_still_gives_request_id(monkeypatch, client):
    def fake_create(**kwargs):
        raise RateLimited("slow down")

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    response = client.post("/ask", json={"prompt": "hello"})

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["status"] == db.STATUS_UPSTREAM_ERROR
    [row] = db.fetch_requests(request_id=detail["request_id"])
    assert row["http_status"] == 429


def test_post_ask_runs_the_stages(monkeypatch, client):
    sent = {}
    monkeypatch.setitem(pipeline.STAGES, "shout", lambda p: pipeline.Prompt(p.text.upper(), p.system))
    monkeypatch.setitem(pipeline.STAGES, "exclaim", lambda p: pipeline.Prompt(p.text + "!", p.system))
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: sent.update(kw) or ok_reply())

    response = client.post("/ask", json={"prompt": "hello", "stages": ["shout", "exclaim"]})

    assert sent["input"] == "HELLO!"
    [row] = db.fetch_requests(request_id=response.json()["request_id"])
    assert row["compression"] == "shout+exclaim"


def capped_reply(text):
    """What Gemini sends back when it stops at max_output_tokens."""
    reply = ok_reply(text)
    reply.status = "incomplete"
    return reply


def test_post_ask_sends_the_clients_cap(monkeypatch, client):
    sent = {}
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: sent.update(kw) or ok_reply())

    response = client.post("/ask", json={"prompt": "hello", "max_output_tokens": 300})

    assert sent["generation_config"] == {"max_output_tokens": 300}
    [row] = db.fetch_requests(request_id=response.json()["request_id"])
    assert row["max_output_tokens"] == 300
    assert row["truncated"] == 0 and row["cap_retry"] == 0


def test_post_ask_null_cap_sends_none(monkeypatch, client):
    sent = {}
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: sent.update(kw) or ok_reply())

    response = client.post("/ask", json={"prompt": "hello", "max_output_tokens": None})

    assert "generation_config" not in sent
    [row] = db.fetch_requests(request_id=response.json()["request_id"])
    assert row["max_output_tokens"] is None


@pytest.mark.parametrize("cap", [0, -5])
def test_post_ask_rejects_a_cap_below_one(client, cap):
    assert client.post("/ask", json={"prompt": "hello", "max_output_tokens": cap}).status_code == 422


def test_get_ask_takes_a_cap(monkeypatch, client):
    sent = {}
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: sent.update(kw) or ok_reply())

    client.get("/ask", params={"prompt": "hello", "max_output_tokens": 50})

    assert sent["generation_config"] == {"max_output_tokens": 50}


def test_cut_off_text_is_sent_back_marked_truncated(monkeypatch, client):
    calls = []
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: calls.append(kw) or capped_reply("Rivers are"))

    response = client.post("/ask", json={"prompt": "essay", "max_output_tokens": 40})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Rivers are" and body["truncated"] is True
    assert len(calls) == 1
    [row] = db.fetch_requests(request_id=body["request_id"])
    assert row["status"] == db.STATUS_OK
    assert row["truncated"] == 1 and row["cap_retry"] == 0
    assert row["error_type"] is None


@pytest.mark.parametrize("cut_off", ["[", '{"answer": "Can', "Here:\n```python\ndef f(", "  ", "<ul><li>"])
def test_cut_off_structured_reply_is_asked_again_uncapped(monkeypatch, client, cut_off):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return capped_reply(cut_off) if len(calls) == 1 else ok_reply('{"answer": "Canberra"}')

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    response = client.post("/ask", json={"prompt": "json please", "max_output_tokens": 40})

    body = response.json()
    assert body["text"] == '{"answer": "Canberra"}' and body["truncated"] is False
    assert calls[0]["generation_config"] == {"max_output_tokens": 40}
    assert "generation_config" not in calls[1]
    [row] = db.fetch_requests(request_id=body["request_id"])
    assert row["status"] == db.STATUS_OK
    assert row["cap_retry"] == 1 and row["truncated"] == 0
    assert row["output_tokens"] == 6 and row["total_tokens"] == 70  # both calls added up


def test_incomplete_without_a_cap_is_still_not_completed(monkeypatch, client):
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: capped_reply("Rivers are"))

    response = client.post("/ask", json={"prompt": "essay", "max_output_tokens": None})

    [row] = db.fetch_requests(request_id=response.json()["request_id"])
    assert row["status"] == db.STATUS_NOT_COMPLETED
    assert row["truncated"] == 0


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        pipeline.compress(pipeline.Prompt("hi"), ["nope"])


# ---------------------------------------------------------------- db migration


def test_init_db_adds_new_columns_to_an_old_db(tmp_path):
    path = tmp_path / "old.db"
    old_columns = {k: v for k, v in db.COLUMNS.items() if k not in ("compression", "latency_compress_ms")}
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE requests ({', '.join(f'{k} {v}' for k, v in old_columns.items())})")
        conn.execute("INSERT INTO requests (request_id, timestamp, status) VALUES ('old', '2026-10-01T00:00:00+00:00', 'ok')")
    db.init_db(path)

    db.insert_request(db.RequestRecord(status=db.STATUS_OK, compression="none", latency_compress_ms=0.1), path)

    rows = db.fetch_requests(path)
    assert [r["compression"] for r in rows] == [None, "none"]
