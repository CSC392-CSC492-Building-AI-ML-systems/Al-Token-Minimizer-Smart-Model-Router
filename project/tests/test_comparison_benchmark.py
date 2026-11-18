import json

import pytest

import comparison_benchmark.run as benchmark_run
from comparison_benchmark.run import (
    _cache_savings,
    _compression_savings,
    _contains_expected_evidence,
    _total_tokens_saved,
)


@pytest.mark.parametrize(
    ("raw", "compressed", "saved", "percent"),
    [
        (100, 75, 25, 25.0),
        (100, 110, -10, -10.0),
        (0, 0, 0, None),
    ],
)
def test_compression_savings_reports_input_token_delta(raw, compressed, saved, percent):
    result = _compression_savings(raw, compressed)

    assert result["input_tokens_uncompressed"] == raw
    assert result["input_tokens_compressed"] == compressed
    assert result["compression_input_tokens_saved"] == saved
    assert result["compression_input_savings_percent"] == percent


def test_save_report_writes_run_metadata_and_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_run, "RESULTS_DIR", tmp_path)
    results = [{"case_id": "sample", "compression_input_tokens_saved": 12}]

    report_path = benchmark_run._save_report(
        results,
        "gemini-test",
        "semantic",
        {"kb_version": "v1"},
    )

    assert report_path.parent.parent == tmp_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model"] == "gemini-test"
    assert report["kb"] == {"version": "v1", "strategy": "semantic"}
    assert report["cases"] == results


def test_cache_savings_estimates_each_hit_from_reference_usage():
    telemetry = {
        "input_tokens_reported": 100,
        "output_tokens": 20,
        "thought_tokens": 10,
        "total_tokens": 130,
    }

    result = _cache_savings(telemetry, exact_hit=True, semantic_hit=True)

    assert result["exact_cache_potential_tokens_saved"] == 130
    assert result["semantic_cache_potential_tokens_saved"] == 130
    assert result["cache_potential_tokens_saved_total"] == 260
    assert result["cache_savings_reference_usage"]["total_tokens"] == 130


def test_cache_savings_is_unknown_if_reference_usage_is_missing():
    telemetry = {
        "input_tokens_reported": None,
        "output_tokens": None,
        "thought_tokens": None,
        "total_tokens": None,
    }

    result = _cache_savings(telemetry, exact_hit=True, semantic_hit=True)

    assert result["exact_cache_potential_tokens_saved"] is None
    assert result["semantic_cache_potential_tokens_saved"] is None
    assert result["cache_potential_tokens_saved_total"] is None


def test_total_tokens_saved_sums_compression_and_cache_savings():
    assert _total_tokens_saved(25, 260) == 285
    assert _total_tokens_saved(25, None) is None


@pytest.mark.parametrize(
    ("answer", "evidence", "matches"),
    [
        ("Run `fastapi dev`; open /docs and /redoc.", ["fastapi dev", "/docs", "/redoc"], True),
        ("Open /docs.", ["/docs", "/redoc"], False),
        ("Open /DOCS.", "/docs", True),
    ],
)
def test_expected_evidence_accepts_one_or_multiple_requirements(answer, evidence, matches):
    assert _contains_expected_evidence(answer, evidence) is matches
