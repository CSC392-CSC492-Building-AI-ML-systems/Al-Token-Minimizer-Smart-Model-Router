AI TOKEN MINIMIZER / COMPRESSION PROXY

This project reduces model usage in three ways: compact code in input prompts,
encourage concise complete answers, and reuse exact answers when requested.
It currently proxies one Gemini model, gemini-3.5-flash. Model selection/routing
and verified dollar pricing are not implemented yet.

From the project folder, install dependencies and start the server:

    uv sync --locked
    cp .env.example .env
    # Put your Gemini API key in .env, which is gitignored.
    uv run --frozen fastapi dev

Interactive API documentation: http://127.0.0.1:8000/docs
If a machine's uv configuration redirects PyPI, use UV_NO_CONFIG=1 and
--default-index https://pypi.org/simple for dependency installation.

HOW THE CODE FITS TOGETHER

    ask.py / HTTP caller
      -> main.py: validate request, check optional exact-match cache
      -> compressor/pipeline.py: ordered stages selected directly or by preset
      -> compressor/compressor_service.py: count original and transformed input
      -> Gemini interactions.create
      -> gemini_parse.py: parse response and classify failures
      -> db.py: one metadata-only SQLite telemetry row per accepted request

response_cache.py holds bounded, expiring answers in process memory.
benchmark/ runs the same POST /ask path with isolated telemetry and cache state.
model_data/ contains the tokenizer and a pricing abstraction. Its pricing
formulas are placeholders; the service leaves uncached cost_usd unknown.
compressor/compressor.py is an older unused placeholder; pipeline.py is the
active extension point. log_setup.py handles metadata-only application logs.

ASK API AND CLI

POST http://127.0.0.1:8000/ask with JSON:

    {"prompt": "Explain this code: ...", "system": "Optional caller instruction",
     "compression": "full", "cache": "use"}

Alternatively send "stages": ["code", "concise"] in place of "compression".
The stage list runs in order; an explicit [] disables compression. The two
configuration fields are mutually exclusive. Omitting both uses the server default.

Returns text, request_id, cache_status, and over_limit (see SOFT OUTPUT LIMIT). On upstream errors, POST returns
an HTTP error and detail containing status and request_id. Partial model
responses remain marked not_completed in telemetry and are never cached.
Invalid request settings return 422 before provider calls.
GET /ask?prompt=... remains compatible, uses the configured default compression,
and bypasses the response cache. Prefer POST to keep prompts out of URLs.

    uv run --frozen python ask.py --compression full --cache use "Explain this"
    cat snippet.txt | uv run --frozen python ask.py --compression code
    cat snippet.txt | uv run --frozen python ask.py --stages code concise
    uv run --frozen python ask.py --usage

COMPRESSION LEVELS (CSC-14, CSC-16, CSC-17)

    none   Unchanged user and system prompts (apart from the soft output
           limit's length line, see below). Default and benchmark baseline.
    light  Add a concise-answer system instruction.
    code   Compact fenced Python and JavaScript structural whitespace.
    full   Run code compression, then concise-answer guidance.

COMPRESSION_LEVEL in .env chooses the server default. Each POST can override it.
These levels are presets for the native ordered-stage API. Stages live in
compressor/stages/ and register with @stage; new stage modules are loaded
automatically. The passthrough example remains available. Prompt.render() retains
knowledge-base chunks and their sources when internal callers supply context.
The registry, preset definitions, and version live in compressor/pipeline.py.
Bump PIPELINE_VERSION when stage behavior changes. The version, effective stages,
and concise instruction are included in cache identity and benchmark metadata.
Telemetry stores the effective stage label, such as code+concise, for either
request form. Equivalent presets and explicit stage lists can share cached answers.

The concise instruction requests the shortest complete answer while retaining
facts, caveats, requested detail, code, and formatting. Caller system instructions
are retained verbatim after this default guidance. There is no arbitrary output
length cap. This adds input tokens and aims to reduce output tokens, so assess
both total usage and quality against the none baseline.

Code compression recognizes python/py/javascript/js fences using backticks or
tildes. It preserves prose, comments, docstrings, strings, regexes, and template
literals. Python uses ast/tokenize; JavaScript uses Tree-sitter. A rewrite must
be shorter and preserve the parsed structure and literal tokens. Unknown
languages, invalid/incomplete code, and blocks over 256 KB pass through unchanged.
Tool directives are treated conservatively. Source coordinates and source
reflection are not preserved by whitespace compression; use none for tasks that
depend on exact source layout. No prompt code is executed by the compressor.

Counts include both user and system text. Identical text is counted once per
request and reused. These are sums of content token counts, not a claim about
provider framing overhead. Gemini-reported usage is stored separately. A negative
input-token saving is valid when concise guidance adds more input than code saves.
If code compaction increases the actual model token count, the service restores
the original user text while retaining any selected concise-answer guidance.

SOFT OUTPUT LIMIT (CSC-19)

There is no hard cap on replies, so nothing is ever cut off. Instead, after the
compression stages run, one line is added to the end of the system prompt
asking Gemini to aim for under 1024 output tokens by default
(DEFAULT_SOFT_OUTPUT_LIMIT in main.py) and to always finish its answer anyway.
Send "soft_output_limit": 500 to ask for a different length, or
"soft_output_limit": null for no limit (GET /ask takes it as a query parameter).
The limit is on the answer only; thinking tokens aren't counted.

A reply that goes over is still sent back whole, with "over_limit": true. Every
row in the db records soft_output_limit, over_limit (1/0) and over_limit_by
(how many output tokens over), and the log line for it is a WARNING.
uv run python db.py summary shows how many replies went over and by how much.
The limit is part of the cache identity, so an answer written for one limit is
not reused for another. Cache hits generate nothing new and have no over-limit
flags.

EXACT-MATCH CACHE (CSC-18)

    cache="off"      Default. Skip lookup and storage.
    cache="use"      Reuse an exact answer or generate and populate on a miss.
    cache="refresh"  Generate a new answer and replace a successful cached entry.

Identity includes original text, context chunks and sources, system text, model, stages,
and effective pipeline version/settings. Whitespace and None/empty differences
are significant. Only completed, nonblank replies are cached. Failed, partial,
and blank replies are not stored. A failed refresh leaves an older successful
entry usable until its original expiry.

Defaults: CACHE_ENABLED=true, CACHE_MAX_ENTRIES=256, CACHE_TTL_SECONDS=300.
CACHE_ENABLED=false disables storage and lookup regardless of request settings.
Each answer is limited to 1 MiB; LRU eviction bounds entry count. TTL uses a
monotonic clock and is not extended by reads. Expired entries are removed lazily.
Each worker has its own cache; restart clears it. Concurrent misses may each call
Gemini. This cache is for the current stateless API; future tenant context,
conversation history, model options, or tools must be part of cache identity.

A hit skips compression, token counting, and generation. It still gets a new
request ID and telemetry row. Raw/compressed prompt counts are reused, while
newly consumed provider tokens and upstream cost are zero. cache_status is one
of off, disabled, miss, hit, refresh. Gemini's cached_tokens describes provider
prompt caching and is separate from this response cache.

TELEMETRY AND PRIVACY

    uv run --frozen python db.py summary
    uv run --frozen python db.py summary --json

The same summary is at GET /telemetry/summary. Both support model, status, since
(inclusive), and until (exclusive) filters. Naive dates are interpreted as UTC.
Startup and the summary CLI migrate existing databases without deleting rows.
Cache summaries count each status and calculate hit rate over hits plus misses;
off, disabled, and explicit refresh requests do not count as lookups.

New /ask rows and application logs do not store raw prompts or answers. Telemetry
stores a request digest, counts, timings, status, HTTP code, and error type.
Free-form error messages are omitted because providers can echo request/response
content in them. Old log/database content is not rewritten. Answers are retained
only in the opt-in memory cache unless benchmark --save-replies is explicitly used.

BENCHMARK WORKLOAD AND CACHE EVALUATION (CSC-22)

    uv run --frozen python -m benchmark list
    uv run --frozen python -m benchmark run --repeats 1 --no-judge
    uv run --frozen python -m benchmark run --pipelines none light code full --cache-modes off cold warm --repeats 2 --no-judge
    uv run --frozen python -m benchmark run --stages none code code+concise --repeats 1 --no-judge
    uv run --frozen python -m benchmark run --limits none 512 1024 --repeats 1 --no-judge

--stages and --pipelines select combinations; both accept stage labels and presets.

The deterministic workload has 19 cases covering JSON/HTML, Python bugs,
JavaScript behavior, literal/comment preservation, strict output formats,
concise answers, exact duplicates, changed prompts, and changed system rules.
Default: 19 cases x 3 repeats = 57 proxy requests before retries. The full
comparison above makes 456 proxy requests before retries. Running the benchmark
uses the real configured Gemini API and can incur usage. Tests use mocked calls.

Cache phases are reported separately: off bypasses the cache; cold refreshes
every request; warm starts empty and reuses repetitions. The first miss is part
of warm-phase totals. Reports show quality checks, errors, cache hits, unexpected
hits, changed cached replies, prompt counts, consumed tokens, and latency.
All-attempt totals include retries; missing usage is reported as unknown.
No dollar savings are inferred from the placeholder pricing formulas.

Runs write benchmark/results/<UTC timestamp>/ with telemetry.db, results.jsonl,
graded.jsonl, meta.json, report.md and report.json. Raw answers and potentially
sensitive check diagnostics are omitted by default; grading happens in memory.
Add --save-replies only when retaining answers for later regrading is intended.

    uv run --frozen python -m benchmark report benchmark/results/<run>
    uv run --frozen python -m benchmark grade benchmark/results/<run> --no-judge

report works without retained answers; grade requires --save-replies on the
original run. Without --no-judge, failed eligible checks may send the question,
expected answer, and reply to Jev via OpenRouter when OPEN_ROUTER_KEY is set.
The run metadata records settings and pipeline fingerprints for reproducibility.

VALIDATION

    uv run --frozen pytest -q

All tests use temporary telemetry/log paths and fake provider calls. They cover
legacy behavior, parsing, schema migration, compression semantics, cache identity,
TTL/LRU/limits, zero-usage hits, privacy, CLI payloads, and benchmark phases.
Mocked tests establish implementation behavior. Real model quality, output-token
savings, and live latency still need a benchmark run with valid credentials.
