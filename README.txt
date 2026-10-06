install uv https://docs.astral.sh/uv/getting-started/installation/
create your own api key on gemini, then in the project folder copy .env.example to .env and put your key in it (.env is gitignored so it won't get committed)

after cloning run: 

uv sync

to set up the sqlite db (creates telemetry.db in the project folder) run:

uv run python db.py

To run the server:

uv run fastapi dev


go to http://127.0.0.1:8000/docs after running