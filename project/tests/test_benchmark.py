"""Check the benchmark: its cases and checks, and the run/grade/report flow.

Gemini and Jev are faked, so no real API calls are made.

Run from the project folder:
    uv run --with pytest pytest tests/test_benchmark.py -v
"""

import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")  # main.py needs one at import

import db  # noqa: E402
import main  # noqa: E402
from benchmark import cases, run  # noqa: E402
from tests.test_post_ask import RateLimited, ok_reply  # noqa: E402


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


def test_numbers_in():
    assert cases.numbers_in("Total: $1,234.50 (was -3)") == [1234.5, -3.0]


# ---------------------------------------------------------------- runner


def run_args(tmp_path, **overrides):
    args = dict(
        tests=["control_short", "verbose_json"],
        repeats=2,
        pipelines=["none"],
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
        if system_instruction:  # verbose_json
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
        if system_instruction:
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
