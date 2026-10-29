"""Check how /ask uses the semantic cache. Redis and the embedding model are
faked (see conftest.py), so this runs without Docker.

Run from the project folder:
    uv run --with pytest pytest tests/test_cache.py -v
"""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

from fastapi.testclient import TestClient  # noqa: E402

import db  # noqa: E402
import main  # noqa: E402
from cache import semantic  # noqa: E402
from compressor.pipeline import Chunk, Prompt  # noqa: E402


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


@pytest.fixture
def stored(monkeypatch):
    """Records every semantic.store call as (prompt text, answer)."""
    calls = []
    monkeypatch.setattr(semantic, "store", lambda prompt, model, answer: calls.append((prompt.text, answer)))
    return calls


def ok_reply(text="from gemini"):
    usage = SimpleNamespace(
        total_input_tokens=12, total_output_tokens=3, total_thought_tokens=0, total_cached_tokens=None, total_tokens=15
    )
    return SimpleNamespace(id="v1_abc", status="completed", model=main.MODEL, output_text=text, errors=None, usage=usage)


def test_a_hit_answers_without_calling_gemini(monkeypatch, client, stored):
    monkeypatch.setattr(semantic, "lookup", lambda prompt, model: "from cache")
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: pytest.fail("Gemini was called"))

    response = client.post("/ask", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["text"] == "from cache"
    [row] = db.fetch_requests(request_id=response.json()["request_id"])
    assert row["cache_hit"] == "semantic"
    assert row["status"] == db.STATUS_OK
    assert row["latency_cache_ms"] >= 0
    assert row["latency_upstream_ms"] is None
    assert stored == []  # already in the cache


def test_a_miss_asks_gemini_and_stores_the_answer(monkeypatch, client, stored):
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: ok_reply())

    response = client.post("/ask", json={"prompt": "hello"})

    assert response.json()["text"] == "from gemini"
    assert stored == [("hello", "from gemini")]
    [row] = db.fetch_requests(request_id=response.json()["request_id"])
    assert row["cache_hit"] is None


def test_the_cache_gets_the_prompt_from_before_compression(monkeypatch, client, stored):
    monkeypatch.setitem(main.pipeline.STAGES, "shout", lambda p: Prompt(p.text.upper(), p.system))
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: ok_reply())

    client.post("/ask", json={"prompt": "hello", "stages": ["shout"]})

    assert stored == [("hello", "from gemini")]


def test_a_failed_reply_is_not_stored(monkeypatch, client, stored):
    failed = ok_reply()
    failed.status = "failed"
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: failed)

    client.post("/ask", json={"prompt": "hello"})

    assert stored == []


def test_cache_false_skips_the_cache(monkeypatch, client, stored):
    monkeypatch.setattr(semantic, "lookup", lambda prompt, model: pytest.fail("cache was checked"))
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: ok_reply())

    response = client.post("/ask", json={"prompt": "hello", "cache": False})

    assert response.json()["text"] == "from gemini"
    assert stored == []


def test_a_broken_cache_still_answers(monkeypatch, client):
    def redis_down(*args):
        raise ConnectionError("Redis is not running")

    monkeypatch.setattr(semantic, "lookup", redis_down)
    monkeypatch.setattr(semantic, "store", redis_down)
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: ok_reply())

    response = client.post("/ask", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["text"] == "from gemini"


# ---------------------------------------------------------------- scope


def test_scope_ignores_the_question():
    assert semantic._scope(Prompt("a", "sys"), "m") == semantic._scope(Prompt("b", "sys"), "m")


@pytest.mark.parametrize(
    "other",
    [
        (Prompt("a", "other sys"), "m"),
        (Prompt("a", "sys"), "other model"),
        (Prompt("a", "sys", (Chunk("a doc"),)), "m"),
    ],
)
def test_scope_changes_with_model_system_and_chunks(other):
    assert semantic._scope(Prompt("a", "sys"), "m") != semantic._scope(*other)


# ---------------------------------------------------------------- startup


def test_startup_warms_up_the_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(semantic, "warm_up", lambda: calls.append(1))

    with TestClient(main.app):  # entering runs the startup code
        pass

    assert calls == [1]


def test_startup_still_works_without_redis(monkeypatch):
    def redis_down():
        raise ConnectionError("Redis is not running")

    monkeypatch.setattr(semantic, "warm_up", redis_down)
    monkeypatch.setattr(main.client.interactions, "create", lambda **kw: ok_reply())

    with TestClient(main.app) as client:
        assert client.post("/ask", json={"prompt": "hello"}).status_code == 200
