"""Check the benchmark: its cases and checks, and the run/grade/report flow.

Gemini and Jev are faked, so no real API calls are made.

Run from the project folder:
    uv run --with pytest pytest tests/test_benchmark.py -v
"""

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

import db  # noqa: E402
import main  # noqa: E402
from benchmark import cases, run  # noqa: E402
from tests.test_post_ask import RateLimited, ok_reply  # noqa: E402

# a call can also get a system prompt from the soft limit's hint, so the fakes
# spot verbose_json by its own one
VERBOSE_JSON_SYSTEM = cases.TESTS["verbose_json"].system


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")  # run folders go here, not benchmark/results
    db.init_db()


@pytest.fixture(autouse=True)
def fake_token_counter(monkeypatch):
    monkeypatch.setattr(main.model_package.tokenizer, "count", lambda text: len(text.split()))


# ---------------------------------------------------------------- cases


@pytest.mark.parametrize("name", [name for name, case in cases.TESTS.items() if case.judge])
def test_expected_answer_passes_its_check(name):
    case = cases.TESTS[name]
    assert case.check(case.expected)[0]


def test_question_and_expected_only_needed_for_jev():
    def make(**fields):
        return cases.Case(name="x", category="x", prompt="p", system=None,
                          check=lambda reply: (True, ""), **fields)

    assert make(judge=False).expected == ""
    assert make(judge=True, question="q?", expected="a").judge
    with pytest.raises(ValueError):
        make(judge=True, question="q?")


def test_generated_prompts_are_the_same_every_time():
    assert cases.orders_json().prompt == cases.orders_json().prompt
    assert cases.products_html().prompt == cases.products_html().prompt


def test_orders_json_is_long_enough():
    assert cases.TESTS["orders_json"].prompt.count("\n") >= 200


def test_every_bug_is_its_own_test():
    assert [name for name in cases.TESTS if name.startswith("code_bug_")] == [
        f"code_bug_{f['name']}" for f in cases.BUGGY_FUNCTIONS
    ]


def test_wrong_answers_fail():
    assert not cases.TESTS["orders_json"].check("12.00")[0]
    assert not cases.TESTS["products_html"].check('["Copper Kettle"]')[0]
    assert not cases.TESTS["js_parse_config"].check("it returns undefined")[0]
    assert not cases.TESTS["cents_comment"].check("it costs $3752")[0]


@pytest.mark.parametrize("function", cases.BUGGY_FUNCTIONS, ids=lambda f: f["name"])
def test_buggy_function_fails_and_fixed_one_passes(function):
    old, new = function["fixed_line"]
    assert not cases.run_unit_tests(function["buggy"], function["tests"])[0]
    assert cases.run_unit_tests(function["buggy"].replace(old, new), function["tests"])[0]


def test_code_bug_check_reads_the_code_block():
    function = cases.BUGGY_FUNCTIONS[0]
    fixed = function["buggy"].replace(*function["fixed_line"])
    case = cases.TESTS["code_bug_moving_average"]

    assert case.check(f"Here you go:\n```python\n{fixed}```\nThe range was one short.")[0]
    assert not case.check(f"```python\n{function['buggy']}```")[0]
    assert not case.check("Change the range to len(values) - window + 1.")[0]


def test_verbose_json_check():
    check = cases.TESTS["verbose_json"].check
    assert check('{"answer": "Canberra", "confidence": 0.98}')[0]
    assert check('```json\n{"answer": "Canberra", "confidence": 0.98}\n```')[0]
    assert not check('Sure! {"answer": "Canberra", "confidence": 0.98}')[0]
    assert not check('{"answer": "Canberra"}')[0]


def test_bullet_check():
    check = cases.TESTS["verbose_bullets"].check
    assert check("Try these:\n- one\n- two\n* three")[0]
    assert check("Just go to bed earlier.")[0]
    assert not check("1. one\n2. two\n3. three\n4. four")[0]
    assert not check("")[0]


def test_numbers_in():
    assert cases.numbers_in("Total: $1,234.50 (was -3)") == [1234.5, -3.0]


def test_number_check_does_not_accept_a_correct_intermediate_value():
    check = cases.TESTS["control_short"].check
    assert check("72°F is 22.2°C")[0]
    assert not check("22.2 is incorrect; my final answer is 99 degrees.")[0]


# ---------------------------------------------------------------- runner


def run_args(tmp_path, **overrides):
    args = dict(
        tests=["control_short", "verbose_json"],
        repeats=2,
        pipelines=["none"],
        limits=["default"],
        seed=0,
        delay=0,
        retries=1,
        backoff=0,
        no_judge=True,
        threshold=0.8,
        max_rate_limited=3,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_run_grades_and_reports(monkeypatch, tmp_path):
    def fake_create(model, input, system_instruction=None):
        if VERBOSE_JSON_SYSTEM in (system_instruction or ""):  # verbose_json
            return ok_reply('{"answer": "Canberra", "confidence": 0.9}')
        return ok_reply("72°F is 22.2°C")

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    out = run.run(run_args(tmp_path))

    assert out.parent == tmp_path / "results"
    graded = [json.loads(line) for line in (out / "graded.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(graded) == 4
    assert all(row["passed"] for row in graded)
    assert {row["request_id"] for row in graded} == {r["request_id"] for r in db.fetch_requests(out / "telemetry.db")}
    stats = json.loads((out / "report.json").read_text(encoding="utf-8"))["stats"]
    assert stats["ALL/none"]["pass_rate"] == 1
    assert stats["ALL/none"]["input_tokens"] == 12
    assert "| control_short | none |" in (out / "report.md").read_text(encoding="utf-8")


def test_run_compares_limits_against_no_limit(monkeypatch, tmp_path):
    hints_sent = []

    def fake_create(model, input, system_instruction=None):
        hints_sent.append(system_instruction)
        reply = ok_reply("72°F is 22.2°C")
        reply.usage.total_output_tokens = 2  # over a limit of 1
        return reply

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    out = run.run(run_args(tmp_path, tests=["control_short"], repeats=1, limits=[1, "none"]))

    assert sorted(hints_sent, key=str) == [main.length_hint(1), None]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["pipelines"] == ["none limit=1", "none limit=none"]
    assert meta["baseline"] == "none limit=none"
    report_json = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "control_short/none limit=1" in report_json["change_vs_baseline"]
    assert report_json["stats"]["ALL/none limit=1"]["over_limit"] == 1
    assert report_json["stats"]["ALL/none limit=1"]["avg_over_by"] == 1
    assert report_json["stats"]["ALL/none limit=none"]["over_limit"] == 0
    assert "## Change vs `none limit=none`" in (out / "report.md").read_text(encoding="utf-8")


def test_limit_argument():
    assert run._limit("none") == "none" and run._limit("default") == "default" and run._limit("256") == 256
    for bad in ("0", "-3", "lots"):
        with pytest.raises(Exception):
            run._limit(bad)


def test_run_retries_rate_limits(monkeypatch, tmp_path):
    calls = []

    def fake_create(model, input, system_instruction=None):
        calls.append(input)
        if len(calls) == 1:
            raise RateLimited("slow down")
        return ok_reply("22.2")

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    out = run.run(run_args(tmp_path, tests=["control_short"], repeats=1))

    [row] = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert row["attempts"] == 2
    assert row["status"] == db.STATUS_OK
    assert len(db.fetch_requests(out / "telemetry.db")) == 2  # the 429 is kept too


def test_jev_only_asked_about_failed_judgeable_checks(monkeypatch, tmp_path):
    def fake_create(model, input, system_instruction=None):
        if VERBOSE_JSON_SYSTEM in (system_instruction or ""):
            return ok_reply("Canberra")  # fails verbose_json, which Jev must not overrule
        return ok_reply("twenty-two point two degrees")  # fails the number check

    asked = []

    def fake_judge(question, expected, reply, api_key):
        asked.append(reply)
        return {"probability": 0.95, "model": "typesafe/jev-1.13", "cost": 0}

    monkeypatch.setattr(main.client.interactions, "create", fake_create)
    monkeypatch.setattr(run, "judge", fake_judge)
    monkeypatch.setenv("OPEN_ROUTER_KEY", "fake")

    out = run.run(run_args(tmp_path, repeats=1, no_judge=False))

    graded = {
        row["test"]: row
        for row in map(json.loads, (out / "graded.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert asked == ["twenty-two point two degrees"]
    assert graded["control_short"]["passed"] and not graded["control_short"]["check"]
    assert graded["verbose_json"]["passed"] is False


def test_run_stops_when_the_quota_is_used_up(monkeypatch, tmp_path):
    def fake_create(model, input, system_instruction=None):
        raise RateLimited("quota")

    monkeypatch.setattr(main.client.interactions, "create", fake_create)

    out = run.run(run_args(tmp_path, repeats=5, retries=0, max_rate_limited=3))

    assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert "Stopped early" in (out / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("reply", [
    '{"answer": "Sydney", "confidence": 0.9}',
    '{"answer": "Canberra", "confidence": true}',
    '{"answer": "Canberra", "confidence": 1.1}',
    '{"answer": "Canberra", "confidence": 0.9, "extra": 1}',
])
def test_json_check_rejects_wrong_content_and_schema(reply):
    assert not cases.TESTS["verbose_json"].check(reply)[0]


def test_workload_covers_exact_repeats_and_near_matches():
    original, repeated, changed_prompt, changed_system = cases.cache_cases()
    assert (original.prompt, original.system) == (repeated.prompt, repeated.system)
    assert original.prompt != changed_prompt.prompt and original.system == changed_prompt.system
    assert original.prompt == changed_system.prompt and original.system != changed_system.system
    for case in cases.cache_cases() + cases.code_controls() + cases.concise_cases()[:1]:
        assert case.check(case.expected)[0]
        assert not case.check('"incorrect"')[0]


def _rows(out, name="results.jsonl"):
    return [json.loads(line) for line in (out / name).read_text(encoding="utf-8").splitlines()]


def test_cache_evaluation_keeps_phases_separate_and_accounts_for_saved_calls(monkeypatch, tmp_path):
    calls = []

    def create(model, input, system_instruction=None):
        calls.append((input, system_instruction))
        answer = 43 if "26" in input else 42
        if "object" in system_instruction:
            answer = {"answer": answer}
        return ok_reply(json.dumps(answer))

    monkeypatch.setattr(main.client.interactions, "create", create)
    out = run.run(run_args(tmp_path, tests=[c.name for c in cases.cache_cases()],
                           cache_modes=["off", "cold", "warm"]))
    rows = _rows(out)
    assert len(rows) == 24
    assert len(calls) == 19  # 8 off + 8 refresh + 3 unique warm requests
    assert {r["cache_status"] for r in rows if r["cache_mode"] == "off"} == {"off"}
    assert {r["cache_status"] for r in rows if r["cache_mode"] == "cold"} == {"refresh"}
    hits = [r for r in rows if r["cache_status"] == "hit"]
    assert len(hits) == 5
    assert all(r["cache_expected_hit"] and r["cache_reply_matches"] for r in hits)
    assert all(r["total_tokens"] == 0 and r["input_tokens_raw"] > 0 for r in hits)
    assert all(row["passed"] for row in _rows(out, "graded.jsonl"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    warm = report["stats"]["ALL/none/warm"]
    assert warm["cache_hit_rate"] == 5 / 8
    assert warm["total_tokens_sum"] == 105
    assert warm["unknown_usage_attempts"] == 0
    assert warm["unexpected_cache_hits"] == warm["cache_reply_mismatches"] == 0
    assert report["stats"]["ALL/none/off"]["total_tokens_sum"] == 280
    assert report["change_vs_cache_off"]["none/warm"]["total_tokens"] == -5 / 8


def test_benchmark_restores_existing_database_and_cache_even_on_failure(monkeypatch, tmp_path):
    from response_cache import CachedResponse, ResponseCache

    previous_cache = ResponseCache(max_entries=3, ttl_seconds=17, max_entry_bytes=256)
    previous_cache.put("existing", CachedResponse("keep this", 1, 1))
    monkeypatch.setattr(main, "response_cache", previous_cache)
    previous_db = os.environ["TELEMETRY_DB_PATH"]

    def fail(*args, **kwargs):
        assert main.response_cache is not previous_cache
        assert main.response_cache.max_entry_bytes == 256
        raise RuntimeError("interrupted")

    monkeypatch.setattr(run, "_call", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        run.run(run_args(tmp_path))
    assert main.response_cache is previous_cache
    assert main.response_cache.get("existing").text == "keep this"
    assert os.environ["TELEMETRY_DB_PATH"] == previous_db


def test_benchmark_default_does_not_persist_answers_or_check_diagnostics(monkeypatch, tmp_path):
    secret = "PRIVATE_RESPONSE_SENTINEL_7829"
    monkeypatch.setattr(main.client.interactions, "create", lambda **kwargs: ok_reply(secret))
    case = replace(cases.TESTS["control_short"], check=lambda reply: (False, reply), judge=False)
    monkeypatch.setitem(cases.TESTS, "control_short", case)
    out = run.run(run_args(tmp_path, tests=["control_short"], repeats=1))
    assert all("reply" not in row for row in _rows(out))
    for path in out.iterdir():
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    with pytest.raises(SystemExit, match="Replies were not retained"):
        run.grade(out, use_judge=False)
    run.report(out)  # report rebuild remains available


def test_explicit_saved_replies_can_be_regraded(monkeypatch, tmp_path):
    monkeypatch.setattr(main.client.interactions, "create", lambda **kwargs: ok_reply("22.22"))
    monkeypatch.setenv("OPEN_ROUTER_KEY", "fake")
    out = run.run(run_args(tmp_path, tests=["control_short"], repeats=1, save_replies=True, no_judge=False))
    assert _rows(out)[0]["reply"] == "22.22"
    original_meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert original_meta["judge_enabled"] and original_meta["judge_threshold"] == 0.8
    run.grade(out, use_judge=False, threshold=0.95)
    assert _rows(out, "graded.jsonl")[0]["passed"]
    run.report(out)
    meta = json.loads((out / "report.json").read_text(encoding="utf-8"))["meta"]
    assert meta["judge_enabled"] is False
    assert meta["judge_threshold"] == 0.95
    assert meta["graded_at"] >= original_meta["started_at"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_regrade_threshold_is_rejected_before_reading_or_rewriting(tmp_path, threshold):
    with pytest.raises(SystemExit, match="threshold must be between"):
        run.grade(tmp_path / "nonexistent", threshold=threshold)


@pytest.mark.parametrize("option", ["--stages", "--pipelines"])
def test_cli_accepts_stage_combinations_and_preset_aliases(monkeypatch, option):
    calls = []
    monkeypatch.setattr(run, "run", lambda args: calls.append(args))
    run.cli(["run", option, "none", "code+concise", "full", "--no-judge"])
    assert calls[0].pipelines == ["none", "code+concise", "full"]


def test_run_sends_ordered_stage_combinations_and_preserves_report_label(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setitem(run.compression.STAGES, "mark_a", lambda prompt: replace(prompt, text=prompt.text + " a"))
    monkeypatch.setitem(run.compression.STAGES, "mark_b", lambda prompt: replace(prompt, text=prompt.text + " b"))
    monkeypatch.setattr(main.client.interactions, "create", lambda **kwargs: sent.append(kwargs) or ok_reply("22.22"))
    out = run.run(run_args(tmp_path, tests=["control_short"], repeats=1, pipelines=["mark_a+mark_b"]))
    assert sent[0]["input"].endswith(" a b")
    [row] = _rows(out)
    assert row["pipeline"] == "mark_a+mark_b"
    [stored] = db.fetch_requests(out / "telemetry.db")
    assert stored["compression"] == "mark_a+mark_b"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["stats"]["ALL/mark_a+mark_b"]["pass_rate"] == 1


def test_equivalent_preset_and_stage_label_share_expected_cache_identity(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(main.client.interactions, "create", lambda **kwargs: sent.append(kwargs) or ok_reply("22.22"))
    out = run.run(run_args(tmp_path, tests=["control_short"], repeats=1,
                           pipelines=["full", "code+concise"], cache_modes=["warm"]))
    assert len(sent) == 1
    rows = _rows(out)
    [hit] = [row for row in rows if row["cache_status"] == "hit"]
    assert hit["cache_expected_hit"] and hit["cache_reply_matches"]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["pipeline_fingerprints"]["full"] == meta["pipeline_fingerprints"]["code+concise"]


def test_usage_totals_include_retries_and_mark_missing_usage_unknown(monkeypatch, tmp_path):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RateLimited("slow down")
        return ok_reply("22.22")

    monkeypatch.setattr(main.client.interactions, "create", create)
    out = run.run(run_args(tmp_path, tests=["control_short"], repeats=1))
    stats = json.loads((out / "report.json").read_text(encoding="utf-8"))["stats"]["ALL/none"]
    assert stats["attempts"] == 2
    assert stats["total_tokens_sum"] == 35
    assert stats["unknown_usage_attempts"] == 1


@pytest.mark.parametrize("overrides", [
    {"repeats": 0}, {"retries": -1}, {"delay": -1}, {"backoff": -1},
    {"max_rate_limited": 0}, {"threshold": 2}, {"cache_modes": ["warm", "warm"]},
    {"pipelines": ["no_such_stage"]},
])
def test_invalid_run_settings_fail_before_making_requests(tmp_path, overrides):
    with pytest.raises(SystemExit):
        run.run(run_args(tmp_path, **overrides))
    assert not (tmp_path / "results").exists()
