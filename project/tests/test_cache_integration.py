"""Exercise the HTTP proxy, cache and telemetry together without Gemini calls."""

import json
import logging
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")

from fastapi.testclient import TestClient  # noqa: E402

import db  # noqa: E402
import main  # noqa: E402
from compressor import pipeline  # noqa: E402
from response_cache import ResponseCache  # noqa: E402


def interaction(text="private answer", *, status="completed", model=None):
    return SimpleNamespace(
        id="original-interaction",
        status=status,
        model=model or main.MODEL,
        output_text=text,
        errors=None,
        usage=SimpleNamespace(total_input_tokens=15, total_output_tokens=3,
                              total_thought_tokens=2, total_cached_tokens=4,
                              total_tokens=20),
    )


@pytest.fixture
def proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "cache-integration.db"))
    db.init_db()
    monkeypatch.setattr(main, "response_cache", ResponseCache())
    calls = {"count": [], "generate": []}

    def count(text):
        calls["count"].append(text)
        return len(text.split())

    def generate(**kwargs):
        calls["generate"].append(kwargs)
        return interaction(model=kwargs["model"])

    monkeypatch.setattr(main.model_package.tokenizer, "count", count)
    monkeypatch.setattr(main.client.interactions, "create", generate)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client, calls


def ask(client, **overrides):
    return client.post("/ask", json={"prompt": "private prompt", "system": "private system",
                                      "compression": "none", "cache": "use", **overrides})


def row_for(response):
    [row] = db.fetch_requests(request_id=response.json()["request_id"])
    return row


def test_cache_hit_skips_model_and_token_counts_but_saves_fresh_telemetry(proxy, monkeypatch):
    client, calls = proxy
    first = ask(client)
    assert first.status_code == 200
    assert first.json()["cache_status"] == "miss"
    original = row_for(first)
    assert original["input_tokens_raw"] == original["input_tokens_compressed"] == 4
    assert original["compression_version"]
    assert len(original["prompt_hash"]) == 64
    assert len(calls["generate"]) == 1
    assert calls["count"]

    def unexpected_call(*args, **kwargs):
        raise AssertionError("a cache hit must not call Gemini or count tokens")

    monkeypatch.setattr(main.client.interactions, "create", unexpected_call)
    monkeypatch.setattr(main.model_package.tokenizer, "count", unexpected_call)
    second = ask(client)
    assert second.status_code == 200
    assert second.json()["cache_status"] == "hit"
    assert second.json()["text"] == first.json()["text"]
    assert second.json()["request_id"] != first.json()["request_id"]
    hit = row_for(second)
    assert hit["status"] == db.STATUS_OK
    assert hit["cache_status"] == "hit"
    assert hit["input_tokens_raw"] == hit["input_tokens_compressed"] == 4
    assert hit["prompt_hash"] == original["prompt_hash"]
    assert hit["compression_version"] == original["compression_version"]
    assert hit["interaction_id"] is None
    for column in ("input_tokens_reported", "output_tokens", "thought_tokens", "cached_tokens",
                   "total_tokens", "cost_usd", "latency_upstream_ms", "latency_count_ms",
                   "latency_compress_ms"):
        assert hit[column] == 0, column
    assert len(db.fetch_requests()) == 2


@pytest.mark.parametrize("change", [
    {"prompt": "private prompt "},
    {"system": "another system"},
    {"system": None},
    {"system": ""},
    {"compression": "light"},
])
def test_different_request_settings_do_not_reuse_answer(proxy, change):
    client, calls = proxy
    assert ask(client).json()["cache_status"] == "miss"
    assert ask(client, **change).json()["cache_status"] == "miss"
    assert ask(client).json()["cache_status"] == "hit"
    assert len(calls["generate"]) == 2


def test_absent_and_empty_system_prompts_have_different_identity(proxy):
    client, calls = proxy
    assert ask(client, system=None).json()["cache_status"] == "miss"
    assert ask(client, system="").json()["cache_status"] == "miss"
    assert len(calls["generate"]) == 2


def test_model_change_invalidates_cache(proxy, monkeypatch):
    client, calls = proxy
    assert ask(client).json()["cache_status"] == "miss"
    monkeypatch.setattr(main, "MODEL", "another-model")
    response = ask(client)
    assert response.json()["cache_status"] == "miss"
    assert row_for(response)["model"] == "another-model"
    assert len(calls["generate"]) == 2


@pytest.mark.parametrize("changed_config", ["version", "stages", "concise_instruction"])
def test_effective_pipeline_configuration_invalidates_cache(proxy, monkeypatch, changed_config):
    client, calls = proxy
    assert ask(client, compression="light").json()["cache_status"] == "miss"
    if changed_config == "version":
        monkeypatch.setattr(pipeline, "PIPELINE_VERSION", "next-version")
    elif changed_config == "stages":
        monkeypatch.setitem(pipeline.PIPELINES, "light", ("code", "concise"))
    else:
        monkeypatch.setattr(pipeline, "CONCISE_INSTRUCTION", "A different style instruction.")
    assert ask(client, compression="light").json()["cache_status"] == "miss"
    assert ask(client, compression="light").json()["cache_status"] == "hit"
    assert len(calls["generate"]) == 2


def test_refresh_updates_cached_answer_and_off_does_not(proxy, monkeypatch):
    client, calls = proxy
    assert ask(client).json()["cache_status"] == "miss"
    monkeypatch.setattr(main.client.interactions, "create", lambda **kwargs: interaction("new answer"))
    refreshed = ask(client, cache="refresh")
    assert refreshed.json()["cache_status"] == "refresh"
    assert refreshed.json()["text"] == "new answer"
    assert row_for(refreshed)["cache_status"] == "refresh"
    assert ask(client).json()["text"] == "new answer"
    monkeypatch.setattr(main.client.interactions, "create", lambda **kwargs: interaction("uncached answer"))
    bypassed = ask(client, cache="off")
    assert bypassed.json()["cache_status"] == "off"
    assert bypassed.json()["text"] == "uncached answer"
    assert row_for(bypassed)["cache_status"] == "off"
    assert ask(client).json()["text"] == "new answer"


def test_default_off_does_not_read_or_populate_cache(proxy):
    client, calls = proxy
    payload = {"prompt": "private prompt", "system": "private system", "compression": "none"}
    for _ in range(2):
        response = client.post("/ask", json=payload)
        assert response.json()["cache_status"] == "off"
    assert len(main.response_cache) == 0
    assert ask(client).json()["cache_status"] == "miss"
    assert len(calls["generate"]) == 3


def test_global_disable_bypasses_requested_cache(proxy, monkeypatch):
    client, calls = proxy
    monkeypatch.setattr(main, "response_cache", ResponseCache(enabled=False))
    for mode in ("use", "use", "refresh"):
        response = ask(client, cache=mode)
        assert response.json()["cache_status"] == "disabled"
        assert row_for(response)["cache_status"] == "disabled"
    assert len(calls["generate"]) == 3
    assert len(main.response_cache) == 0


def test_expired_answer_triggers_new_upstream_request(proxy, monkeypatch):
    client, calls = proxy
    now = [0]
    monkeypatch.setattr(main, "response_cache", ResponseCache(ttl_seconds=1, clock=lambda: now[0]))
    assert ask(client).json()["cache_status"] == "miss"
    assert ask(client).json()["cache_status"] == "hit"
    now[0] = 1
    assert ask(client).json()["cache_status"] == "miss"
    assert len(calls["generate"]) == 2


@pytest.mark.parametrize(("text", "status"), [
    ("", "completed"),
    (" \n", "completed"),
    ("partial answer", "in_progress"),
])
def test_empty_or_incomplete_responses_are_not_cached(proxy, monkeypatch, text, status):
    client, calls = proxy

    def generate(**kwargs):
        calls["generate"].append(kwargs)
        return interaction(text, status=status)

    monkeypatch.setattr(main.client.interactions, "create", generate)
    ask(client)
    ask(client)
    assert len(calls["generate"]) == 2
    assert len(main.response_cache) == 0
    assert all(row["cache_status"] == "miss" for row in db.fetch_requests())


def test_failed_request_not_cached_and_next_success_can_be_cached(proxy, monkeypatch):
    client, calls = proxy

    class RateLimited(Exception):
        status_code = 429
        body = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "slow down"}}

    def fail(**kwargs):
        raise RateLimited("unreadable upstream body")

    monkeypatch.setattr(main.client.interactions, "create", fail)
    response = ask(client)
    assert response.status_code == 429
    [failed] = db.fetch_requests()
    assert failed["cache_status"] == "miss"
    assert failed["status"] == db.STATUS_UPSTREAM_ERROR
    assert len(main.response_cache) == 0
    monkeypatch.setattr(main.client.interactions, "create", lambda **kwargs: interaction())
    assert ask(client).json()["cache_status"] == "miss"
    assert ask(client).json()["cache_status"] == "hit"


def test_cache_and_telemetry_do_not_log_or_persist_text(proxy, caplog):
    client, calls = proxy
    with caplog.at_level(logging.INFO):
        ask(client)
        ask(client)
    persisted = json.dumps(db.fetch_requests())
    for secret in ("private prompt", "private system", "private answer"):
        assert secret not in caplog.text
        assert secret not in persisted


@pytest.mark.parametrize("failure", ["exception", "upstream", "partial"])
def test_error_echoes_do_not_persist_request_or_response_text(proxy, monkeypatch, caplog, failure):
    client, calls = proxy

    class UpstreamError(Exception):
        status_code = 400
        body = {"error": {"status": "INVALID_ARGUMENT", "message": "private prompt / private system"}}

    def generate(**kwargs):
        if failure == "exception":
            raise RuntimeError("failure involving private prompt and private system")
        if failure == "upstream":
            raise UpstreamError("raw upstream body: private prompt")
        result = interaction(status="in_progress")
        result.errors = [{"code": "INVALID", "message": "private prompt / private system / private answer"}]
        return result

    monkeypatch.setattr(main.client.interactions, "create", generate)
    with caplog.at_level(logging.INFO):
        response = ask(client)
    assert len(main.response_cache) == 0
    [row] = db.fetch_requests()
    assert row["status"] != db.STATUS_OK
    persisted = json.dumps(row)
    for secret in ("private prompt", "private system", "private answer"):
        assert secret not in persisted
        assert secret not in caplog.text


def test_telemetry_write_failure_does_not_log_exception_payload(proxy, monkeypatch, caplog):
    client, calls = proxy

    def fail_save(*args, **kwargs):
        raise RuntimeError("private prompt / private system / private answer")

    monkeypatch.setattr(db, "insert_request", fail_save)
    with caplog.at_level(logging.INFO):
        response = ask(client)
    assert response.status_code == 200
    assert "Could not save request" in caplog.text
    for secret in ("private prompt", "private system", "private answer"):
        assert secret not in caplog.text


def test_truncated_error_does_not_leak_fragments_of_a_long_prompt(proxy, monkeypatch, caplog):
    client, calls = proxy
    secret_fragment = "private-long-prompt-fragment"
    prompt = (secret_fragment + " ") * 50

    def fail(**kwargs):
        raise RuntimeError("failed input: " + kwargs["input"])

    monkeypatch.setattr(main.client.interactions, "create", fail)
    with caplog.at_level(logging.INFO):
        response = ask(client, prompt=prompt)
    assert response.status_code == 500
    [row] = db.fetch_requests()
    assert row["error_type"] == "RuntimeError"
    assert secret_fragment not in json.dumps(row)
    assert secret_fragment not in caplog.text


def test_get_provider_error_is_sanitized_before_it_reaches_server_logging(proxy, monkeypatch, caplog):
    def fail(**kwargs):
        raise RuntimeError("provider echoed private prompt / private answer")

    monkeypatch.setattr(main.client.interactions, "create", fail)
    # If GET still lets the provider exception escape, this client raises it,
    # reproducing the exception Uvicorn would otherwise log with a traceback.
    with TestClient(main.app, raise_server_exceptions=True) as client, caplog.at_level(logging.INFO):
        response = client.get("/ask", params={"prompt": "private prompt"})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["status"] == db.STATUS_INTERNAL_ERROR
    [row] = db.fetch_requests(request_id=detail["request_id"])
    assert row["error_type"] == "RuntimeError"
    for secret in ("private prompt", "private answer"):
        assert secret not in json.dumps(row)
        assert secret not in response.text
        assert secret not in caplog.text


def test_invalid_cache_mode_rejected_before_telemetry_or_generation(proxy):
    client, calls = proxy
    response = ask(client, cache="forever")
    assert response.status_code == 422
    assert db.fetch_requests() == []
    assert calls == {"count": [], "generate": []}
