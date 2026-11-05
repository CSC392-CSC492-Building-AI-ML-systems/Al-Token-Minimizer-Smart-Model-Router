"""Command-line client for the local compression proxy."""

import argparse
import json
import sys
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from compressor.pipeline import PIPELINES, resolve_stages

BASE = "http://127.0.0.1:8000"


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Send a prompt to the FastAPI Gemini proxy")
    parser.add_argument("prompt", nargs="*", help="prompt text (or pipe via stdin)")
    parser.add_argument("--system", help="optional system instruction")
    compression = parser.add_mutually_exclusive_group()
    compression.add_argument("--compression", choices=sorted(PIPELINES), help="default: server configuration")
    compression.add_argument("--stages", nargs="*", help="ordered stage names; use -- before a following prompt")
    parser.add_argument("--cache", choices=("off", "use", "refresh"), default="off",
                        help="reuse exact replies, refresh one, or bypass the cache (default: off)")
    parser.add_argument("--usage", action="store_true", help="show stored telemetry totals instead")
    args = parser.parse_args(argv)
    if args.stages is not None:
        try:
            args.stages = list(resolve_stages(args.stages))
        except ValueError as exc:
            parser.error(str(exc))

    if args.usage:
        if (args.prompt or args.system is not None or args.compression is not None
                or args.stages is not None or args.cache != "off"):
            parser.error("--usage cannot be combined with a prompt, --system, --compression, --stages or --cache")
        request = Request(f"{BASE}/telemetry/summary", headers={"Accept": "application/json"})
        timeout = 10
    else:
        # Preserve stdin whitespace, which can be significant in source code and
        # is part of the exact-match cache identity.
        prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read()
        if not prompt.strip():
            parser.error("No prompt given.")
        body = {"prompt": prompt, "cache": args.cache}
        if args.system is not None:
            body["system"] = args.system
        if args.stages is not None:
            body["stages"] = args.stages
        elif args.compression is not None:
            body["compression"] = args.compression
        request = Request(
            f"{BASE}/ask",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        timeout = 120

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        # Error bodies and reasons can echo prompts or answers. Print only the
        # status, never provider bodies or exception text.
        print(f"Request failed: proxy returned HTTP {exc.code}.", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError, HTTPException):
        print("Request failed: could not reach the proxy.", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, UnicodeDecodeError):
        print("Request failed: proxy returned invalid JSON.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
