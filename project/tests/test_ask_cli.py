"""CLI request contracts with a mocked local HTTP transport."""

import io
import json
from http.client import BadStatusLine
from urllib.error import HTTPError, URLError

import pytest

import ask


@pytest.fixture
def transport(monkeypatch):
    sent = []

    def open_request(request, timeout):
        sent.append((request, timeout))
        return io.BytesIO(json.dumps({"text": "answer", "request_id": "id", "cache_status": "miss"}).encode())

    monkeypatch.setattr(ask, "urlopen", open_request)
    return sent


def test_posts_compression_system_and_cache_options(transport, capsys):
    result = ask.cli(["--compression", "full", "--cache", "use", "--system", "Be brief.",
                      "Explain", "caching"])
    assert result == 0
    [(request, timeout)] = transport
    assert request.full_url == "http://127.0.0.1:8000/ask"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 120
    assert json.loads(request.data) == {"prompt": "Explain caching", "system": "Be brief.",
                                        "compression": "full", "cache": "use"}
    assert json.loads(capsys.readouterr().out)["text"] == "answer"


def test_defaults_honor_server_compression_and_bypass_cache(transport):
    assert ask.cli(["Hello"]) == 0
    [(request, _)] = transport
    assert json.loads(request.data) == {"prompt": "Hello", "cache": "off"}


def test_stdin_code_whitespace_is_preserved(transport, monkeypatch):
    prompt = "\n```python\ndef hello():\n    return 'hi'\n```\n"
    monkeypatch.setattr(ask.sys, "stdin", io.StringIO(prompt))
    assert ask.cli(["--compression", "code", "--cache", "refresh"]) == 0
    [(request, _)] = transport
    assert json.loads(request.data) == {"prompt": prompt, "compression": "code", "cache": "refresh"}


def test_explicit_ordered_stages_are_sent_without_preset(transport):
    assert ask.cli(["Explain this", "--stages", "code", "concise"]) == 0
    [(request, _)] = transport
    assert json.loads(request.data) == {"prompt": "Explain this", "stages": ["code", "concise"], "cache": "off"}


def test_empty_stages_explicitly_overrides_server_default(transport, monkeypatch):
    monkeypatch.setattr(ask.sys, "stdin", io.StringIO("Unchanged prompt\n"))
    assert ask.cli(["--stages"]) == 0
    [(request, _)] = transport
    assert json.loads(request.data) == {"prompt": "Unchanged prompt\n", "stages": [], "cache": "off"}


def test_double_dash_separates_stage_names_from_prompt(transport):
    assert ask.cli(["--stages", "concise", "code", "--", "Explain", "this"]) == 0
    [(request, _)] = transport
    assert json.loads(request.data)["stages"] == ["concise", "code"]
    assert json.loads(request.data)["prompt"] == "Explain this"


def test_usage_requests_existing_summary_endpoint(transport):
    assert ask.cli(["--usage"]) == 0
    [(request, timeout)] = transport
    assert request.full_url == "http://127.0.0.1:8000/telemetry/summary"
    assert request.get_method() == "GET"
    assert request.data is None
    assert timeout == 10


@pytest.mark.parametrize("argv", [
    ["--compression", "unknown", "hello"],
    ["--cache", "forever", "hello"],
    ["--previous", "not-supported", "hello"],
    ["--usage", "hello"],
    ["--usage", "--system", "ignored"],
    ["--usage", "--stages"],
    ["--stages", "unknown-stage"],
    ["--compression", "full", "--stages", "code"],
])
def test_invalid_or_unsupported_options_fail_before_network(transport, argv):
    with pytest.raises(SystemExit) as exc:
        ask.cli(argv)
    assert exc.value.code == 2
    assert transport == []


def test_empty_prompt_fails_before_network(transport, monkeypatch):
    monkeypatch.setattr(ask.sys, "stdin", io.StringIO("  \n"))
    with pytest.raises(SystemExit) as exc:
        ask.cli([])
    assert exc.value.code == 2
    assert transport == []


@pytest.mark.parametrize("error", [
    HTTPError("http://localhost/ask", 429, "private prompt", {}, io.BytesIO(b"private answer")),
    URLError("private prompt / private answer"),
    TimeoutError("private prompt / private answer"),
    BadStatusLine("private prompt / private answer"),
])
def test_request_errors_do_not_echo_prompt_or_response(monkeypatch, capsys, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(ask, "urlopen", fail)
    assert ask.cli(["private prompt"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Request failed:" in captured.err
    assert "private prompt" not in captured.err
    assert "private answer" not in captured.err


def test_invalid_json_response_is_reported_without_body(monkeypatch, capsys):
    monkeypatch.setattr(ask, "urlopen", lambda *args, **kwargs: io.BytesIO(b"private answer"))
    assert ask.cli(["hello"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid JSON" in captured.err
    assert "private answer" not in captured.err
