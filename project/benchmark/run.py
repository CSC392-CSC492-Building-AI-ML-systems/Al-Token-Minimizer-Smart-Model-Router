"""Run the compression benchmark, grade the replies and write a report.

From the project folder:
    uv run python -m benchmark run                      # baseline: every test 3 times, no stages ("none")
    uv run python -m benchmark run --stages none a a+b --repeats 5   # compare no stages, stage a, and a then b
    uv run python -m benchmark run --tests orders_json verbose_json --repeats 1
    uv run python -m benchmark run --caps none 1024 4096     # compare output caps against no cap
    uv run python -m benchmark grade benchmark/results/<run>   # re-grade (e.g. after changing a check)
    uv run python -m benchmark report benchmark/results/<run>  # rebuild the report
    uv run python -m benchmark list                     # show the tests and their sizes

Each run gets its own folder, named after the time it started, under
benchmark/results/ (gitignored), e.g. benchmark/results/2026-10-15_03-14-44/:
    telemetry.db    the /ask rows for this run only, same schema as the real one
    results.jsonl   one line per call: test, pipeline, repeat, reply text and the row's metrics
    graded.jsonl    results.jsonl plus the check result and Jev's verdict
    report.md/json  pass rate, tokens and latency per test and pipeline
    meta.json       the settings the run used

Requests go through POST /ask in this process (FastAPI's TestClient), so they
take the same path as real traffic, compression and telemetry included,
without needing the server running. A "pipeline" here is a combination of
compression stages written as a label: "none", "a" or "a+b" (stage a, then b).
With --caps it's also an output cap, e.g. "none cap=1024" (max_output_tokens
1024) or "none cap=none" (no cap); without it every call gets the server's
default cap and pipelines are named by their stages alone.
Each repeat runs every (test, pipeline)
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
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from benchmark import cases as case_lib
from benchmark.judge import judge
from compressor import pipeline as compression

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE = "none"  # no stages; every other combination is compared against it
DEFAULT_CAP = "default"  # --caps value that leaves max_output_tokens to the server
NO_CAP = "none"  # --caps value that sends max_output_tokens: null

# Copied from the request's db row into results.jsonl.
ROW_COLUMNS = (
    "status",
    "http_status",
    "error_type",
    "error_message",
    "input_tokens_reported",
    "output_tokens",
    "thought_tokens",
    "cached_tokens",
    "total_tokens",
    "max_output_tokens",
    "truncated",
    "cap_retry",
    "latency_total_ms",
    "latency_upstream_ms",
    "latency_compress_ms",
)
RETRY_HTTP = {429, 500, 503}  # rate limited or overloaded ("high demand"): wait and try again
RETRY_STATUSES = {"timeout", "connection_error"}


# ---------------------------------------------------------------- run


def run(args):
    unknown = {s for name in args.pipelines for s in compression.unknown_stages(compression.parse_label(name))}
    if unknown:
        sys.exit(f"unknown stages {sorted(unknown)}, expected some of {sorted(compression.STAGES)}")

    out = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # local time
    out.mkdir(parents=True, exist_ok=False)
    # main.py creates its db when imported, so point it at this run's folder first
    os.environ["TELEMETRY_DB_PATH"] = str(out / "telemetry.db")
    import db
    import main
    from fastapi.testclient import TestClient

    db.init_db()  # in case main was already imported (e.g. in tests) with another db
    client = TestClient(main.app, raise_server_exceptions=False)
    case_list = [case_lib.TESTS[name] for name in args.tests]
    variants = [(stages, cap) for stages in args.pipelines for cap in args.caps]
    names = {variant: _variant_name(*variant, args.caps) for variant in variants}
    baseline_cap = NO_CAP if NO_CAP in args.caps else args.caps[0]
    meta = {
        "started_at": _now(),
        "model": main.MODEL,
        "pipelines": list(names.values()),
        "baseline": _variant_name(BASELINE, baseline_cap, args.caps),
        "stages": args.pipelines,
        "caps": args.caps,
        "default_max_output_tokens": main.DEFAULT_MAX_OUTPUT_TOKENS,
        "tests": args.tests,
        "repeats": args.repeats,
        "seed": args.seed,
        "git_commit": _git_commit(),
    }
    _write_json(out / "meta.json", meta)

    order = random.Random(args.seed)
    jobs = []
    for repeat in range(args.repeats):
        batch = [(repeat, case, variant) for case in case_list for variant in variants]
        order.shuffle(batch)
        jobs += batch
    rate_limited = 0  # rate-limited calls in a row, after retries
    print(f"{len(jobs)} calls to {main.MODEL}, writing to {out}")
    with open(out / "results.jsonl", "w", encoding="utf-8") as results:
        for done, (repeat, case, variant) in enumerate(jobs, start=1):
            name = names[variant]
            row = _call(client, db, case, *variant, args)
            row.update(
                test=case.name,
                category=case.category,
                pipeline=name,
                repeat=repeat,
            )
            results.write(json.dumps(row) + "\n")
            results.flush()  # so an interrupted run can still be graded
            latency = row["latency_total_ms"]
            print(
                f"[{done}/{len(jobs)}] {case.name:<26} {name:<10} {row['status']:<16}"
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
    grade(out, use_judge=not args.no_judge, threshold=args.threshold)
    report(out)
    return out


def _variant_name(stages, cap, caps):
    """How a (stages, cap) pair is shown: just the stages when only the default cap is run."""
    return stages if caps == [DEFAULT_CAP] else f"{stages} cap={cap}"


def _call(client, db, case, stages, cap, args):
    """POST the case to /ask, retrying rate limits and timeouts. Returns the results.jsonl fields."""
    body = {"prompt": case.prompt, "system": case.system, "stages": compression.parse_label(stages)}
    if cap != DEFAULT_CAP:
        body["max_output_tokens"] = None if cap == NO_CAP else cap
    for attempt in range(1, args.retries + 2):
        response = client.post("/ask", json=body)
        payload = response.json()
        detail = payload.get("detail")
        request_id = payload.get("request_id") or (detail.get("request_id") if isinstance(detail, dict) else None)
        if request_id is None:
            raise RuntimeError(f"/ask answered {response.status_code} without a request_id: {payload}")
        [stored] = db.fetch_requests(request_id=request_id)

        retryable = stored["http_status"] in RETRY_HTTP or stored["status"] in RETRY_STATUSES
        if response.status_code == 200 or not retryable or attempt > args.retries:
            break
        wait = args.backoff * 2 ** (attempt - 1)
        print(f"    {stored['status']} ({stored['http_status']}), retrying in {wait:g}s")
        time.sleep(wait)

    return {
        "request_id": request_id,
        "attempts": attempt,  # the earlier attempts are still in telemetry.db
        "reply": payload.get("text"),
        **{column: stored[column] for column in ROW_COLUMNS},
    }


# ---------------------------------------------------------------- grade


def grade(out, use_judge=True, threshold=0.8):
    """Check every ok reply; ask Jev about failed ones on judge=True cases. Writes graded.jsonl."""
    load_dotenv()
    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if use_judge and not api_key:
        print("OPEN_ROUTER_KEY not set, grading with the checks only")
        use_judge = False

    rows = _read_jsonl(Path(out) / "results.jsonl")
    judged = 0
    with open(Path(out) / "graded.jsonl", "w", encoding="utf-8") as graded:
        for row in rows:
            row = dict(row, check=None, check_detail=None, jev=None, passed=None)
            if row["status"] == "ok":
                case = case_lib.TESTS[row["test"]]
                row["check"], row["check_detail"] = case.check(row["reply"] or "")
                if not row["check"] and case.judge and use_judge:
                    judged += 1
                    try:
                        row["jev"] = judge(case.question, case.expected, row["reply"] or "", api_key)
                    except Exception as exc:  # grade the rest even if OpenRouter is down
                        row["jev"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                probability = (row["jev"] or {}).get("probability")
                row["passed"] = row["check"] or (probability is not None and probability >= threshold)
            graded.write(json.dumps(row) + "\n")
    print(f"graded {len(rows)} replies, asked Jev about {judged}")


# ---------------------------------------------------------------- report


def report(out):
    """Summarize graded.jsonl per test and pipeline. Writes report.md and report.json."""
    out = Path(out)
    rows = _read_jsonl(out / "graded.jsonl")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))

    groups = {}
    for row in rows:
        groups.setdefault((row["test"], row["pipeline"]), []).append(row)
        groups.setdefault(("ALL", row["pipeline"]), []).append(row)
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
        "Pass rate is over the calls that got a reply (errors are counted separately). "
        "\"Jev\" is how many failed checks Jev passed. Tokens are means as reported by Gemini. "
        "\"Cut off\" is how many replies stopped at the output cap and were sent back short, "
        "\"re-asked\" how many hit it as JSON/code and were asked again without a cap (their tokens are both calls). "
        "Times are in ms, over the calls that got a reply: \"total\" is the whole /ask request "
        "(what a user waits for), \"upstream\" is the Gemini call alone, \"compress\" is the compression pipeline.",
        "",
        "| test | pipeline | calls | errors | pass | Jev | cut off | re-asked | input tok | output tok | thought tok "
        "| total tok | total avg | fastest | slowest | upstream avg | compress avg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for test in tests:
        for name in pipelines:
            s = stats.get((test, name))
            if s is None:
                continue
            lines.append(
                f"| {test} | {name} | {s['calls']} | {s['errors']} | {_pct(s['pass_rate'])} | {s['jev_passed']} "
                f"| {s['truncated']} | {s['cap_retries']} "
                f"| {_num(s['input_tokens'])} | {_num(s['output_tokens'])} | {_num(s['thought_tokens'])} "
                f"| {_num(s['total_tokens'])} | {_num(s['latency_avg'])} | {_num(s['latency_min'])} "
                f"| {_num(s['latency_max'])} | {_num(s['upstream_avg'])} | {_num(s['compress_avg'], 2)} |"
            )

    baseline = meta.get("baseline", BASELINE)  # runs from before --caps don't have it
    compared = [p for p in pipelines if p != baseline]
    deltas = {}
    if baseline in pipelines and compared:
        lines += [
            "",
            f"## Change vs `{baseline}`",
            "",
            "| test | pipeline | input tok | output tok | thought tok | total tok | total avg | pass rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for test in tests:
            base = stats.get((test, baseline))
            for name in compared:
                s = stats.get((test, name))
                if base is None or s is None:
                    continue
                d = {
                    "input_tokens": _change(s["input_tokens"], base["input_tokens"]),
                    "output_tokens": _change(s["output_tokens"], base["output_tokens"]),
                    "thought_tokens": _change(s["thought_tokens"], base["thought_tokens"]),
                    "total_tokens": _change(s["total_tokens"], base["total_tokens"]),
                    "latency_avg": _change(s["latency_avg"], base["latency_avg"]),
                    "pass_rate_points": None
                    if s["pass_rate"] is None or base["pass_rate"] is None
                    else (s["pass_rate"] - base["pass_rate"]) * 100,
                }
                deltas[f"{test}/{name}"] = d
                points = d["pass_rate_points"]
                lines.append(
                    f"| {test} | {name} | {_signed_pct(d['input_tokens'])} | {_signed_pct(d['output_tokens'])} "
                    f"| {_signed_pct(d['thought_tokens'])} | {_signed_pct(d['total_tokens'])} "
                    f"| {_signed_pct(d['latency_avg'])} | {'-' if points is None else f'{points:+.0f} pts'} |"
                )

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
        {"meta": meta, "stats": {f"{f}/{p}": s for (f, p), s in stats.items()}, "change_vs_baseline": deltas},
    )
    print()
    print(text)


def _stats(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    passed = [r for r in ok if r["passed"]]

    def values(column):
        return [r[column] for r in ok if r[column] is not None]

    def mean(column):
        return statistics.fmean(values(column)) if values(column) else None

    return {
        "calls": len(rows),
        "errors": len(rows) - len(ok),
        "pass_rate": len(passed) / len(ok) if ok else None,
        "check_pass_rate": sum(1 for r in ok if r["check"]) / len(ok) if ok else None,
        "jev_passed": sum(1 for r in passed if not r["check"]),
        "truncated": sum(1 for r in ok if r.get("truncated")),  # .get: runs from before the cap don't have it
        "cap_retries": sum(1 for r in ok if r.get("cap_retry")),
        "input_tokens": mean("input_tokens_reported"),
        "output_tokens": mean("output_tokens"),
        "thought_tokens": mean("thought_tokens"),
        "total_tokens": mean("total_tokens"),
        "latency_avg": mean("latency_total_ms"),
        "latency_min": min(values("latency_total_ms"), default=None),
        "latency_max": max(values("latency_total_ms"), default=None),
        "upstream_avg": mean("latency_upstream_ms"),
        "compress_avg": mean("latency_compress_ms"),
    }


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


def _cap(text):
    if text in (DEFAULT_CAP, NO_CAP):
        return text
    if not text.isdigit() or int(text) == 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, {NO_CAP} or {DEFAULT_CAP}, got {text!r}")
    return int(text)


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
    run_parser.add_argument("--stages", dest="pipelines", metavar="COMBO", nargs="+", default=[BASELINE],
                            help=f"stage combinations to compare, e.g. none a a+b (default: {BASELINE}; "
                                 f"stages: {', '.join(sorted(compression.STAGES))})")
    run_parser.add_argument("--caps", nargs="+", type=_cap, default=[DEFAULT_CAP],
                            help=f"output caps (max_output_tokens) to compare: numbers, {NO_CAP} for no cap or "
                                 f"{DEFAULT_CAP} for the server's cap (default: {DEFAULT_CAP}); "
                                 f"{NO_CAP} is the baseline if listed")
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
