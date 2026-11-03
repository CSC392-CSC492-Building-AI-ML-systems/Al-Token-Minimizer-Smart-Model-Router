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
    assert sent == {"model": "gemini-3.5-flash", "input": "capital?", "system_instruction": "be brief"}
    [row] = db.fetch_requests(request_id=body["request_id"])
    assert row["status"] == db.STATUS_OK
    assert row["compression"] == "none"
    assert row["latency_compress_ms"] >= 0
    assert row["latency_count_ms"] >= 0
    assert row["input_tokens_raw"] == 3  # user text plus system instruction
    assert row["input_tokens_compressed"] == 3
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


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        pipeline.compress(pipeline.Prompt("hi"), ["nope"])


def test_unknown_preset_rejected(client):
    assert client.post("/ask", json={"prompt": "hi", "compression": "unknown"}).status_code == 422
    assert db.fetch_requests() == []


def test_preset_and_stage_list_are_mutually_exclusive(client):
    response = client.post("/ask", json={"prompt": "hi", "compression": "light", "stages": []})
    assert response.status_code == 422
    assert db.fetch_requests() == []


def test_explicit_empty_stages_override_configured_default(monkeypatch, client):
    sent = []
    monkeypatch.setattr(main, "DEFAULT_COMPRESSION", "light")
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: sent.append(kw) or ok_reply())
    client.post("/ask", json={"prompt": "hi"})
    client.post("/ask", json={"prompt": "hi", "stages": []})
    assert sent[0]["system_instruction"] == pipeline.CONCISE_INSTRUCTION
    assert "system_instruction" not in sent[1]
    assert [row["compression"] for row in db.fetch_requests()] == ["concise", "none"]


def test_equivalent_preset_and_stage_list_share_cache(monkeypatch, client):
    from response_cache import ResponseCache

    monkeypatch.setattr(main, "response_cache", ResponseCache())
    calls = []
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: calls.append(kw) or ok_reply())
    original = client.post("/ask", json={"prompt": "hi", "compression": "full", "cache": "use"})
    repeated = client.post("/ask", json={"prompt": "hi", "stages": ["code", "concise"], "cache": "use"})
    assert original.json()["cache_status"] == "miss"
    assert repeated.json()["cache_status"] == "hit"
    assert len(calls) == 1
    assert all(row["compression"] == "code+concise" for row in db.fetch_requests())


def test_context_is_sent_counted_and_included_in_cache_identity(monkeypatch):
    from response_cache import ResponseCache

    monkeypatch.setattr(main, "response_cache", ResponseCache())
    sent = []
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: sent.append(kw) or ok_reply())
    first = pipeline.Prompt("question", "system", (pipeline.Chunk("first content", "a.txt"),))
    second = pipeline.Prompt("question", "system", (pipeline.Chunk("different content", "a.txt"),))
    for prompt in (first, second, first):
        record = db.RequestRecord(model=main.MODEL)
        main._answer(record, prompt, (), "use")
    assert [request["input"] for request in sent] == [first.render(), second.render()]
    rows = db.fetch_requests()
    assert [row["cache_status"] for row in rows] == ["miss", "miss", "hit"]
    assert rows[0]["input_tokens_raw"] == len(first.render().split()) + 1


def test_equal_rendered_text_with_different_chunk_boundaries_does_not_share_cache(monkeypatch):
    from response_cache import ResponseCache

    monkeypatch.setattr(main, "response_cache", ResponseCache())
    monkeypatch.setattr(main.model_package.tokenizer, "count", len)
    sent = []
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: sent.append(kw) or ok_reply())
    opening = "```python\nx    =    1"
    closing = "```"
    first = pipeline.Prompt("question", context=(pipeline.Chunk(opening + "\n\n" + closing),))
    second = pipeline.Prompt("question", context=(pipeline.Chunk(opening), pipeline.Chunk(closing)))
    assert first.render() == second.render()
    for prompt in (first, second):
        main._answer(db.RequestRecord(model=main.MODEL), prompt, ("code",), "use")
    assert len(sent) == 2
    assert sent[0]["input"] != sent[1]["input"]
    assert [row["cache_status"] for row in db.fetch_requests()] == ["miss", "miss"]


# ---------------------------------------------------------------- db migration


def test_init_db_adds_new_columns_to_an_old_db(tmp_path):
    path = tmp_path / "old.db"
    old_columns = {k: v for k, v in db.COLUMNS.items() if k not in (
        "compression", "latency_compress_ms", "compression_version", "cache_status",
    )}
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE requests ({', '.join(f'{k} {v}' for k, v in old_columns.items())})")
        conn.execute("INSERT INTO requests (request_id, timestamp, status) VALUES ('old', '2026-10-01T00:00:00+00:00', 'ok')")
    db.init_db(path)

    db.insert_request(db.RequestRecord(status=db.STATUS_OK, compression="none", latency_compress_ms=0.1,
                                       compression_version="1", cache_status="hit"), path)

    rows = db.fetch_requests(path)
    assert [r["compression"] for r in rows] == [None, "none"]
    assert [r["cache_status"] for r in rows] == [None, "hit"]
    assert [r["compression_version"] for r in rows] == [None, "1"]
