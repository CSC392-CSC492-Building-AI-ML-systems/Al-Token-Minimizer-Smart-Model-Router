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

{"prompt": "...", "system": "optional system prompt", "compression": "none"}

it returns {"text": ..., "request_id": ...}. compression picks a pipeline from
compressor/pipeline.py (only "none" exists for now).


BENCHMARK

from the project folder:

uv run python -m benchmark run                            (baseline, 30 calls)
uv run python -m benchmark run --repeats 1                (barebones, 10 calls)
uv run python -m benchmark run --pipelines <EVENTUAL PIPELINE STAGE HERE> <EVENTUAL PIPELINE OTHER STAGE>    (compare with a pipeline)

results go in benchmark/results/<time>/report.md. --help for all options.
For reading the .md file VS code should be able to render the table in markdown if you righclick on the file and click Open preview 