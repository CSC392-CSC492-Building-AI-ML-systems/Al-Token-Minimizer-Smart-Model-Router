"""Check that /ask records total and upstream latency, including on failure.

Gemini is mocked, so no real API calls are made.

Run from the project folder:
    uv run --with pytest pytest tests/test_latency.py -v
"""

import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

from fastapi.testclient import TestClient  # noqa: E402

import db  # noqa: E402
import main  # noqa: E402


@pytest.fixture
def records(monkeypatch):
    """Collects every RequestRecord that /ask creates."""
    created = []
    real_record = db.RequestRecord

    def capture(*args, **kwargs):
        record = real_record(*args, **kwargs)
        created.append(record)
        return record

    monkeypatch.setattr(main.db, "RequestRecord", capture)
    return created


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def test_success_sets_total_and_upstream(monkeypatch, records, client):
    def fake_create(model, input):
        time.sleep(0.01)
        return SimpleNamespace(output_text="hi")

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    response = client.get("/ask", params={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json() == {"text": "hi"}
    [record] = records
    assert record.latency_upstream_ms >= 10
    assert record.latency_total_ms >= record.latency_upstream_ms


def test_gemini_error_still_sets_latency(monkeypatch, records, client):
    def fake_create(model, input):
        time.sleep(0.01)
        raise RuntimeError("gemini is down")

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    response = client.get("/ask", params={"prompt": "hello"})

    assert response.status_code == 500
    [record] = records
    assert record.latency_upstream_ms >= 10
    assert record.latency_total_ms >= record.latency_upstream_ms


def test_count_latency_not_set_yet(monkeypatch, records, client):
    # filled in once token counting (CSC-6) lands in /ask
    monkeypatch.setattr(
        main.client.interactions, "create", lambda model, input: SimpleNamespace(output_text="hi")
    )

    client.get("/ask", params={"prompt": "hello"})

    [record] = records
    assert record.latency_count_ms is None
