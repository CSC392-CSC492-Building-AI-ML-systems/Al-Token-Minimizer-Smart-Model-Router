"""Run the compression benchmark, grade the replies and write a report.

From the project folder:
    uv run python -m benchmark run                      # baseline: every test 3 times, no stages ("none")
    uv run python -m benchmark run --stages none code code+concise --repeats 5
    uv run python -m benchmark run --tests orders_json verbose_json --repeats 1
    uv run python -m benchmark grade benchmark/results/<run>   # re-grade (e.g. after changing a check)
    uv run python -m benchmark report benchmark/results/<run>  # rebuild the report
    uv run python -m benchmark list                     # show the tests and their sizes

Each run gets its own folder, named after the time it started, under
benchmark/results/ (gitignored), e.g. benchmark/results/2026-10-15_03-14-44/:
    telemetry.db    the /ask rows for this run only, same schema as the real one
    results.jsonl   one line per call: test, pipeline, repeat and the row's metrics
    graded.jsonl    results.jsonl plus the check result and Jev's verdict
    report.md/json  pass rate, tokens and latency per test and pipeline
    meta.json       the settings the run used

Requests go through POST /ask in this process (FastAPI's TestClient), so they
take the same path as real traffic, compression and telemetry included,
without needing the server running. A "pipeline" here is an ordered combination
of stages, written as a label such as "none", "code" or "code+concise", or a
preset such as "light" or "full". Replies are graded in memory and are only
saved when --save-replies is explicitly set. Each repeat runs every (test, pipeline)
pair in a shuffled order, so pipelines are compared under the same Gemini load
instead of one after the other.
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from benchmark import cases as case_lib
from benchmark.judge import judge
from compressor import pipeline as compression

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE = "none"  # the baseline must not change if the server default changes
CACHE_MODES = {"off": "off", "cold": "refresh", "warm": "use"}

# Copied from the request's db row into results.jsonl.
ROW_COLUMNS = (
    "status",
    "http_status",
    "error_type",
    "error_message",
    "input_tokens_raw",
    "input_tokens_compressed",
    "input_tokens_reported",
    "output_tokens",
    "thought_tokens",
    "cached_tokens",
    "total_tokens",
    "latency_total_ms",
    "latency_upstream_ms",
    "latency_compress_ms",
    "latency_count_ms",
    "cache_status",
)
RETRY_HTTP = {429, 500, 503}  # rate limited or overloaded ("high demand"): wait and try again
RETRY_STATUSES = {"timeout", "connection_error"}


# ---------------------------------------------------------------- run


def run(args):
    try:
        for name in args.pipelines:
            compression.resolve_stages(name)
    except ValueError as exc:
        sys.exit(str(exc))

    modes = getattr(args, "cache_modes", ["off"])
    if not modes or set(modes) - set(CACHE_MODES) or len(modes) != len(set(modes)):
        sys.exit(f"cache modes must be distinct choices from {list(CACHE_MODES)}")
    if args.repeats < 1 or args.retries < 0 or args.delay < 0 or args.backoff < 0 or args.max_rate_limited < 1:
        sys.exit("repeats and max-rate-limited must be positive; retries, delay and backoff must be nonnegative")
    if not args.pipelines or not args.tests or len(args.pipelines) != len(set(args.pipelines)) or len(args.tests) != len(set(args.tests)):
        sys.exit("pipeline and test names must be nonempty and unique")
    if not 0 <= args.threshold <= 1:
        sys.exit("threshold must be between 0 and 1")

    out = RESULTS_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    with _isolated_runtime(out) as (db, main, client):
        _run_calls(out, args, modes, db, main, client)
    report(out)
    return out


@contextmanager
def _isolated_runtime(out):
    """Keep benchmark telemetry and cache contents separate from other in-process work."""
    previous_db = os.environ.get("TELEMETRY_DB_PATH")
    os.environ["TELEMETRY_DB_PATH"] = str(out / "telemetry.db")
    previous_cache = None
    try:
        import db
        import main
        from fastapi.testclient import TestClient
        from response_cache import ResponseCache

        previous_cache = main.response_cache
        main.response_cache = ResponseCache(
            max_entries=previous_cache.max_entries,
            ttl_seconds=previous_cache.ttl_seconds,
            max_entry_bytes=previous_cache.max_entry_bytes,
            enabled=previous_cache.enabled,
        )
        db.init_db()
        with TestClient(main.app, raise_server_exceptions=False) as client:
            yield db, main, client
    finally:
        if previous_cache is not None:
            main.response_cache.clear()
            main.response_cache = previous_cache
        if previous_db is None:
            os.environ.pop("TELEMETRY_DB_PATH", None)
        else:
            os.environ["TELEMETRY_DB_PATH"] = previous_db


def _run_calls(out, args, modes, db, main, client):
    case_list = [case_lib.TESTS[name] for name in args.tests]
    save_replies = getattr(args, "save_replies", False)
    api_key = _judge_key(not args.no_judge)
    meta = {
        "started_at": _now(),
        "model": main.MODEL,
        "pipelines": args.pipelines,
        "pipeline_version": compression.PIPELINE_VERSION,
        "pipeline_fingerprints": {name: compression.fingerprint(name) for name in args.pipelines},
        "tests": args.tests,
        "repeats": args.repeats,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "cache_modes": modes,
        "cache_enabled": main.response_cache.enabled,
        "cache_max_entries": main.response_cache.max_entries,
        "cache_ttl_seconds": main.response_cache.ttl_seconds,
        "save_replies": save_replies,
        "judge_enabled": bool(api_key),
        "judge_threshold": args.threshold,
    }
    _write_json(out / "meta.json", meta)

    order = random.Random(args.seed)
    jobs = []
    for mode in modes:
        for repeat in range(args.repeats):
            batch = [(mode, repeat, case, name) for case in case_list for name in args.pipelines]
            order.shuffle(batch)
            jobs += batch
    rate_limited = 0  # rate-limited calls in a row, after retries
    print(f"{len(jobs)} proxy requests to {main.MODEL} before retries, writing to {out}")
    previous_mode = None
    seen = {}
    with open(out / "results.jsonl", "w", encoding="utf-8") as results, \
            open(out / "graded.jsonl", "w", encoding="utf-8") as graded:
        for done, (mode, repeat, case, name) in enumerate(jobs, start=1):
            if mode != previous_mode:
                main.response_cache.clear()
                seen.clear()
                previous_mode = mode
            # Presets and explicit labels resolving to the same stage order
            # share server cache identity, even when reported under two labels.
            identity = (case.prompt, case.system, tuple(compression.resolve_stages(name)))
            expected_hit = mode == "warm" and identity in seen and main.response_cache.enabled
            row = _call(client, db, case, name, args, cache_mode=mode)
            row.update(
                test=case.name,
                category=case.category,
                pipeline=name,
                repeat=repeat,
                cache_mode=mode,
                cache_expected_hit=expected_hit,
                cache_reply_matches=(row["reply"] == seen.get(identity))
                if row.get("cache_status") == "hit" else None,
            )
            if row["status"] == "ok" and row.get("cache_status") in {"miss", "refresh"}:
                seen[identity] = row["reply"]
            checked = _grade_row(row, api_key=api_key, threshold=args.threshold, retain_details=save_replies)
            if not save_replies:
                row.pop("reply", None)
                checked.pop("reply", None)
            results.write(json.dumps(row) + "\n")
            results.flush()  # so an interrupted run can still be graded
            graded.write(json.dumps(checked) + "\n")
            graded.flush()
            latency = row["latency_total_ms"]
            print(
                f"[{done}/{len(jobs)}] {case.name:<26} {name:<10} {mode:<5} {row['status']:<16}"
                f"{'' if latency is None else f'{latency:8.0f} ms'}"
            )

            # a used-up daily quota answers 429 for hours, so don't spend
            # the rest of the run retrying it
            rate_limited = rate_limited + 1 if row["http_status"] == 429 else 0
            if rate_limited >= args.max_rate_limited:
                meta["stopped_early"] = f"{rate_limited} calls in a row were rate limited (429)"
                print(f"stopping: {meta['stopped_early']}, the Gemini quota is probably used up")
                break
            if args.delay:
                time.sleep(args.delay)

    meta["finished_at"] = _now()
    _write_json(out / "meta.json", meta)


def _call(client, db, case, pipeline_name, args, cache_mode="off"):
    """POST the case to /ask, retrying rate limits and timeouts. Returns the results.jsonl fields."""
    body = {"prompt": case.prompt, "system": case.system,
            "stages": list(compression.resolve_stages(pipeline_name)),
            "cache": CACHE_MODES[cache_mode]}
    attempts = []
    started = time.perf_counter()
    for attempt in range(1, args.retries + 2):
        response = client.post("/ask", json=body)
        payload = response.json()
        detail = payload.get("detail")
        request_id = payload.get("request_id") or (detail.get("request_id") if isinstance(detail, dict) else None)
        if request_id is None:
            raise RuntimeError(f"/ask answered {response.status_code} without a request_id: {payload}")
        [stored] = db.fetch_requests(request_id=request_id)
        attempts.append(stored)

        retryable = stored["http_status"] in RETRY_HTTP or stored["status"] in RETRY_STATUSES
        if response.status_code == 200 or not retryable or attempt > args.retries:
            break
        wait = args.backoff * 2 ** (attempt - 1)
        print(f"    {stored['status']} ({stored['http_status']}), retrying in {wait:g}s")
        time.sleep(wait)

    return {
        "request_id": request_id,
        "attempts": attempt,  # the earlier attempts are still in telemetry.db
        "request_latency_ms": (time.perf_counter() - started) * 1000,
        "attempt_input_tokens": _known_sum(r["input_tokens_reported"] for r in attempts),
        "attempt_output_tokens": _known_sum(r["output_tokens"] for r in attempts),
        "attempt_total_tokens": _known_sum(r["total_tokens"] for r in attempts),
        "unknown_usage_attempts": sum(r["total_tokens"] is None for r in attempts),
        "reply": payload.get("text"),
        **{column: stored.get(column) for column in ROW_COLUMNS},
    }


# ---------------------------------------------------------------- grade


def grade(out, use_judge=True, threshold=0.8):
    """Regrade an explicitly retained run. The default run grades replies in memory."""
    if not 0 <= threshold <= 1:
        sys.exit("threshold must be between 0 and 1")
    out = Path(out)
    api_key = _judge_key(use_judge)
    rows = _read_jsonl(out / "results.jsonl")
    if any("reply" not in row for row in rows):
        sys.exit("Replies were not retained. Use report to rebuild this run's report; "
                 "future runs need --save-replies for regrading.")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    with open(out / "graded.jsonl", "w", encoding="utf-8") as graded:
        for row in rows:
            graded.write(json.dumps(_grade_row(row, api_key=api_key, threshold=threshold)) + "\n")
    meta.update(judge_enabled=bool(api_key), judge_threshold=threshold, graded_at=_now())
    _write_json(out / "meta.json", meta)
    print(f"graded {len(rows)} replies")


def _judge_key(use_judge):
    load_dotenv()
    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if use_judge and not api_key:
        print("OPEN_ROUTER_KEY not set, grading with the checks only")
    return api_key if use_judge else None


def _grade_row(row, *, api_key, threshold, retain_details=True):
    row = dict(row, check=None, check_detail=None, jev=None, passed=None)
    if row["status"] == "ok":
        case = case_lib.TESTS[row["test"]]
        row["check"], row["check_detail"] = case.check(row.get("reply") or "")
        if not retain_details:
            row["check_detail"] = "deterministic check passed" if row["check"] else "deterministic check failed"
        if not row["check"] and case.judge and api_key:
            try:
                row["jev"] = judge(case.question, case.expected, row.get("reply") or "", api_key)
            except Exception as exc:  # exceptions can echo the raw answer; retain only their type
                row["jev"] = {"error": type(exc).__name__}
        probability = (row["jev"] or {}).get("probability")
        row["passed"] = bool(row["check"] or (probability is not None and probability >= threshold))
    return row


# ---------------------------------------------------------------- report


def report(out):
    """Summarize graded.jsonl per test and pipeline. Writes report.md and report.json."""
    out = Path(out)
    rows = _read_jsonl(out / "graded.jsonl")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))

    modes = meta.get("cache_modes", ["off"])
    groups = {}
    for row in rows:
        mode = row.get("cache_mode", "off")
        groups.setdefault((row["test"], row["pipeline"], mode), []).append(row)
        groups.setdefault(("ALL", row["pipeline"], mode), []).append(row)
    tests = [t for t in meta["tests"] if any(key[0] == t for key in groups)] + ["ALL"]
    pipelines = meta["pipelines"]
    stats = {key: _stats(group) for key, group in groups.items()}

    lines = [
        f"# Compression benchmark {out.name}",
        "",
        f"Model `{meta['model']}`, pipelines {', '.join(f'`{p}`' for p in pipelines)}, "
        f"{meta['repeats']} repeat(s) per test, "
        f"commit `{(meta.get('git_commit') or '?')[:7]}`, started {meta['started_at']}.",
        "",
        *([f"**Stopped early: {meta['stopped_early']}.**", ""] if meta.get("stopped_early") else []),
        "Pass rate is over successful replies; errors are separate. Tokens are mean newly consumed "
        "provider usage per successful request, including zero usage on cache hits. Raw/compressed "
        "counts describe the prompt, so they remain nonzero on hits. Times are in ms. "
        "Request latency includes retries and backoff; upstream and compression times describe the final attempt.",
        "",
        "Cache off bypasses lookup and storage. Cold refreshes every request. Warm starts empty, then "
        "reuses exact requests, including deliberate duplicate cases. Warm hit rate includes the first miss. "
        "Expiry and capacity eviction may reduce reuse; unexpected hits or changed cached replies indicate a failure.",
        "",
        "Quality and token reductions require a live run to establish. Deterministic mocked tests validate "
        "the runner and cache accounting, not a model's response quality.",
        "",
        "| test | pipeline | cache | calls | errors | pass | hits | hit rate | raw tok | compressed tok "
        "| input tok | output tok | total tok | request avg | request p95 | upstream avg | compress avg |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for test in tests:
        for name in pipelines:
            for mode in modes:
                s = stats.get((test, name, mode))
                if s is None:
                    continue
                lines.append(
                    f"| {test} | {name} | {mode} | {s['calls']} | {s['errors']} | {_pct(s['pass_rate'])} "
                    f"| {s['cache_hits']} | {_pct(s['cache_hit_rate'])} | {_num(s['input_tokens_raw'])} "
                    f"| {_num(s['input_tokens_compressed'])} | {_num(s['input_tokens'])} | {_num(s['output_tokens'])} "
                    f"| {_num(s['total_tokens'])} | {_num(s['request_latency_avg'])} | {_num(s['request_latency_p95'])} "
                    f"| {_num(s['upstream_avg'])} | {_num(s['compress_avg'], 2)} |"
                )

    lines += [
        "", "## Usage across all attempts", "",
        "Totals include retries and errors with reported usage. Missing usage is unknown, not zero. "
        "A total with unknown attempts is a lower bound. No dollar savings are inferred without model pricing.", "",
        "| pipeline | cache | attempts | input tok | output tok | total tok | unknown usage | unexpected hits | changed replies |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in pipelines:
        for mode in modes:
            s = stats.get(("ALL", name, mode))
            if s:
                lines.append(
                    f"| {name} | {mode} | {s['attempts']} | {_num(s['input_tokens_sum'])} "
                    f"| {_num(s['output_tokens_sum'])} | {_num(s['total_tokens_sum'])} "
                    f"| {s['unknown_usage_attempts']} | {s['unexpected_cache_hits']} | {s['cache_reply_mismatches']} |"
                )

    compared = [p for p in pipelines if p != BASELINE]
    deltas = {}
    if BASELINE in pipelines and compared:
        lines += [
            "",
            f"## Change vs `{BASELINE}`",
            "",
            "Compared within the same cache phase. Token savings on warm phases include response reuse.",
            "", "| test | pipeline | cache | input tok | output tok | total tok | request avg | pass rate |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for test in tests:
            for mode in modes:
                base = stats.get((test, BASELINE, mode))
                for name in compared:
                    s = stats.get((test, name, mode))
                    if base is None or s is None:
                        continue
                    d = {
                        "input_tokens": _change(s["input_tokens"], base["input_tokens"]),
                        "output_tokens": _change(s["output_tokens"], base["output_tokens"]),
                        "total_tokens": _change(s["total_tokens"], base["total_tokens"]),
                        "latency_avg": _change(s["request_latency_avg"], base["request_latency_avg"]),
                        "pass_rate_points": None
                        if s["pass_rate"] is None or base["pass_rate"] is None
                        else (s["pass_rate"] - base["pass_rate"]) * 100,
                    }
                    deltas[_stats_key(test, name, mode, modes)] = d
                    points = d["pass_rate_points"]
                    lines.append(
                        f"| {test} | {name} | {mode} | {_signed_pct(d['input_tokens'])} "
                        f"| {_signed_pct(d['output_tokens'])} | {_signed_pct(d['total_tokens'])} "
                        f"| {_signed_pct(d['latency_avg'])} | {'-' if points is None else f'{points:+.0f} pts'} |"
                    )

    cache_deltas = {}
    if "off" in modes:
        for name in pipelines:
            base = stats.get(("ALL", name, "off"))
            for mode in (m for m in modes if m != "off"):
                s = stats.get(("ALL", name, mode))
                if base and s:
                    cache_deltas[f"{name}/{mode}"] = {
                        "total_tokens": _change(s["total_tokens"], base["total_tokens"]),
                        "request_latency": _change(s["request_latency_avg"], base["request_latency_avg"]),
                    }

    failures = [r for r in rows if r["passed"] is False]
    if failures:
        lines += ["", "## Failed checks", ""]
        for r in failures:
            jev = r["jev"] or {}
            verdict = f", Jev {jev['probability']:.2f}" if "probability" in jev else (
                f", Jev error: {jev['error']}" if "error" in jev else ""
            )
            lines.append(f"- `{r['test']}` `{r['pipeline']}` repeat {r['repeat']}: {r['check_detail']}{verdict}")
    errors = [r for r in rows if r["status"] != "ok"]
    if errors:
        lines += ["", "## Errors", ""]
        for r in errors:
            lines.append(
                f"- `{r['test']}` `{r['pipeline']}` repeat {r['repeat']}: {r['status']}"
                f" {r['http_status'] or ''} {r['error_message'] or ''}".rstrip()
            )

    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    _write_json(
        out / "report.json",
        {"meta": meta,
         "stats": {_stats_key(f, p, m, modes): s for (f, p, m), s in stats.items()},
         "change_vs_baseline": deltas, "change_vs_cache_off": cache_deltas},
    )
    print()
    print(text)


def _stats(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    passed = [r for r in ok if r["passed"]]

    def values(column):
        return [r[column] for r in ok if r.get(column) is not None]

    def mean(column):
        return statistics.fmean(values(column)) if values(column) else None

    return {
        "calls": len(rows),
        "errors": len(rows) - len(ok),
        "pass_rate": len(passed) / len(ok) if ok else None,
        "check_pass_rate": sum(1 for r in ok if r["check"]) / len(ok) if ok else None,
        "jev_passed": sum(1 for r in passed if not r["check"]),
        "cache_hits": sum(r.get("cache_status") == "hit" for r in rows),
        "cache_hit_rate": sum(r.get("cache_status") == "hit" for r in rows) / len(rows) if rows else None,
        "unexpected_cache_hits": sum(r.get("cache_status") == "hit" and not r.get("cache_expected_hit") for r in rows),
        "cache_reply_mismatches": sum(r.get("cache_reply_matches") is False for r in rows),
        "input_tokens_raw": mean("input_tokens_raw"),
        "input_tokens_compressed": mean("input_tokens_compressed"),
        "input_tokens": mean("input_tokens_reported"),
        "output_tokens": mean("output_tokens"),
        "thought_tokens": mean("thought_tokens"),
        "total_tokens": mean("total_tokens"),
        "latency_avg": mean("latency_total_ms"),
        "latency_min": min(values("latency_total_ms"), default=None),
        "latency_max": max(values("latency_total_ms"), default=None),
        "upstream_avg": mean("latency_upstream_ms"),
        "compress_avg": mean("latency_compress_ms"),
        "request_latency_avg": mean("request_latency_ms"),
        "request_latency_p50": _percentile(values("request_latency_ms"), 0.50),
        "request_latency_p95": _percentile(values("request_latency_ms"), 0.95),
        "attempts": sum(r.get("attempts", 1) for r in rows),
        "input_tokens_sum": _known_sum(r.get("attempt_input_tokens", r.get("input_tokens_reported")) for r in rows),
        "output_tokens_sum": _known_sum(r.get("attempt_output_tokens", r.get("output_tokens")) for r in rows),
        "total_tokens_sum": _known_sum(r.get("attempt_total_tokens", r.get("total_tokens")) for r in rows),
        "unknown_usage_attempts": sum(r.get("unknown_usage_attempts", int(r.get("total_tokens") is None)) for r in rows),
    }


def _stats_key(test, pipeline, mode, modes):
    # Retain the original report.json keys for the default cache-off run.
    return f"{test}/{pipeline}" if modes == ["off"] else f"{test}/{pipeline}/{mode}"


def _known_sum(values):
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _change(value, base):
    return None if value is None or not base else (value - base) / base


def _num(value, decimals=0):
    return "-" if value is None else f"{value:,.{decimals}f}"


def _pct(value):
    return "-" if value is None else f"{value:.0%}"


def _signed_pct(value):
    return "-" if value is None else f"{value:+.1%}"


# ---------------------------------------------------------------- helpers


def _now():
    return datetime.now(timezone.utc).isoformat()


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_jsonl(path):
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def list_tests():
    for case in case_lib.TESTS.values():
        size = len(case.prompt) + len(case.system or "")
        print(f"{case.name:<26} {case.category:<22} {size:>7,} chars  judge={case.judge}  expected {case.expected[:60]}")


def cli(argv=None):
    parser = argparse.ArgumentParser(prog="python -m benchmark", description="Compression benchmark.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_grade_args(p):
        p.add_argument("--no-judge", action="store_true", help="grade with the checks only, no Jev calls")
        p.add_argument("--threshold", type=float, default=0.8,
                       help="Jev probability at which a failed check counts as passed (default 0.8)")

    run_parser = commands.add_parser("run", help="run the benchmark, then grade and report")
    run_parser.add_argument("--tests", nargs="+", choices=list(case_lib.TESTS), default=list(case_lib.TESTS),
                            help="tests to run (default: all, see `list`)")
    add_grade_args(run_parser)
    run_parser.add_argument("--stages", "--pipelines", dest="pipelines", metavar="COMBO", nargs="+", default=[BASELINE],
                            help=f"stage combinations or presets, e.g. none code code+concise full "
                                 f"(default: {BASELINE}; stages: {', '.join(sorted(compression.STAGES))}; "
                                 f"presets: {', '.join(compression.PIPELINES)})")
    run_parser.add_argument("--cache-modes", nargs="+", choices=list(CACHE_MODES), default=["off"],
                            help="cache phases: off bypasses caching, cold refreshes every request, "
                                 "warm starts empty then reuses exact repeats (default off)")
    run_parser.add_argument("--save-replies", action="store_true",
                            help="explicitly save response text to enable later regrading; omitted by default")
    run_parser.add_argument("--repeats", type=int, default=3, help="times each test is sent per combination (default 3)")
    run_parser.add_argument("--seed", type=int, default=0, help="seed for the call order (default 0)")
    run_parser.add_argument("--delay", type=float, default=0, help="seconds to wait between calls (default 0)")
    run_parser.add_argument("--retries", type=int, default=3, help="retries for 429/5xx/timeouts (default 3)")
    run_parser.add_argument("--backoff", type=float, default=10, help="first retry wait in seconds, doubles (default 10)")
    run_parser.add_argument("--max-rate-limited", type=int, default=3,
                            help="stop after this many rate-limited calls in a row (default 3)")

    grade_parser = commands.add_parser("grade", help="grade a finished run again")
    grade_parser.add_argument("run_dir")
    add_grade_args(grade_parser)

    report_parser = commands.add_parser("report", help="rebuild a run's report")
    report_parser.add_argument("run_dir")

    commands.add_parser("list", help="show the tests")

    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "grade":
        grade(args.run_dir, use_judge=not args.no_judge, threshold=args.threshold)
        report(args.run_dir)
    elif args.command == "report":
        report(args.run_dir)
    else:
        list_tests()
