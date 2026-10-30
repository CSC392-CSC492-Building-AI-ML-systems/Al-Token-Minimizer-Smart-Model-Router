install uv https://docs.astral.sh/uv/getting-started/installation/
create your own api key on gemini, then in the project folder copy .env.example to .env and put your key in it (.env is gitignored so it won't get committed)

after cloning run: 

uv sync

to set up the sqlite db (creates telemetry.db in the project folder) run:

uv run python db.py

every /ask request (failed ones too) is saved as one row in that db. to see a
summary of what's stored (requests per status, tokens, cost, latency) run:

uv run python db.py summary

add --json for raw output, or filter with --model, --status, --since and
--until (see uv run python db.py summary --help). while the server is running
the same summary is at http://127.0.0.1:8000/telemetry/summary

To run the server:

uv run fastapi dev


go to http://127.0.0.1:8000/docs after running

POST /ask takes a JSON body instead of a query string (use it for long prompts):

{"prompt": "...", "system": "optional system prompt", "stages": ["passthrough"]}

it returns {"text": ..., "request_id": ..., "truncated": false}. stages are the compression stages
to run, in order (leave it out for none). each stage is a file in
compressor/stages/, see compressor/pipeline.py for how to add one.

OUTPUT CAP

replies are capped at 4096 tokens by default (DEFAULT_MAX_OUTPUT_TOKENS in main.py).
send "max_output_tokens": 500 to use your own cap, or "max_output_tokens": null
for none (GET /ask takes it as a query parameter). gemini's thinking counts
toward the cap, so small caps can get used up before the answer starts.

when a reply hits the cap:
- plain text comes back cut short, with "truncated": true
- JSON, HTML, code blocks or an empty reply would be broken cut short, so the
  server asks gemini again with no cap and returns the full reply (the db row
  has cap_retry = 1 and its token counts cover both calls)

every row in the db records max_output_tokens, truncated and cap_retry.


BENCHMARK

from the project folder:

uv run python -m benchmark run                            (baseline, 30 calls)
uv run python -m benchmark run --repeats 1                (barebones, 10 calls)
uv run python -m benchmark run --stages none a a+b         (compare no stages, stage a, and a then b)
uv run python -m benchmark run --caps none 1024 4096       (compare output caps against no cap)

results go in benchmark/results/<time>/report.md. --help for all options.
For reading the .md file VS code should be able to render the table in markdown if you righclick on the file and click Open preview 