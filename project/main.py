import logging
import os
import time
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from google import genai
from pydantic import BaseModel, field_validator

import db
from compressor import pipeline
from gemini_parse import fill_from_error, fill_from_interaction, reply_text
from log_setup import setup_logging

load_dotenv()  # reads GEMINI_API_KEY from project/.env (not committed)
setup_logging()  # writes to project/logs/app.log, see log_setup.py
db.init_db()  # creates telemetry.db + table if they're not there yet
logger = logging.getLogger(__name__)

app = FastAPI()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set, copy .env.example to .env and add your key")
client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash"


def _ms_since(start):
    return (time.perf_counter() - start) * 1000


# NOTE ON KEEPING TRACK OF STUFF TO WRITE TO THE DB:co to keep track of all the different fields before writing to the sqlite for a
# specific request, use the RequestRecord object (not sure if it's worth having
# this, maybe just a dict is enough?). it's one record per request, save it once
# at the end, even if the request failed:
#   record = db.RequestRecord(endpoint="/ask", model="gemini-3.5-flash")
#   record.input_tokens_raw = 120
#   record.status = db.STATUS_OK 
#   .........
#   db.insert_request(record) <-- this actually saves to the sqlite
@app.get("/ask")
def ask(prompt: str):
    record = db.RequestRecord(endpoint="/ask", model=MODEL)
    return {"text": _answer(record, pipeline.Prompt(prompt))}


class AskRequest(BaseModel):
    prompt: str
    system: str | None = None  # sent as Gemini's system_instruction
    compression: str = pipeline.DEFAULT_PIPELINE  # a name from compressor.pipeline.PIPELINES

    @field_validator("compression")
    @classmethod
    def _known_pipeline(cls, name):
        if name not in pipeline.PIPELINES:
            raise ValueError(f"must be one of {sorted(pipeline.PIPELINES)}")
        return name


# HTTP code POST /ask answers with when the request fails, by db status.
# 429 is passed through as is, so a caller (like the benchmark) knows to slow down.
_HTTP_FOR_STATUS = {
    db.STATUS_UPSTREAM_ERROR: 502,
    db.STATUS_CONNECTION_ERROR: 502,
    db.STATUS_NOT_COMPLETED: 502,
    db.STATUS_TIMEOUT: 504,
    db.STATUS_INTERNAL_ERROR: 500,
}


@app.post("/ask")
def ask_post(body: AskRequest):
    """Like GET /ask, but takes a system prompt and a compression pipeline, and
    returns the request_id of the row saved for it (also on failure, in `detail`)."""
    record = db.RequestRecord(endpoint="/ask", model=MODEL)
    try:
        text = _answer(record, pipeline.Prompt(body.prompt, body.system), body.compression)
    except Exception:
        code = 429 if record.http_status == 429 else _HTTP_FOR_STATUS.get(record.status, 500)
        raise HTTPException(code, {"status": record.status, "request_id": record.request_id}) from None
    return {"text": text, "request_id": record.request_id}


def _answer(record, prompt, compression=pipeline.DEFAULT_PIPELINE):
    """Compress the prompt, send it to Gemini and return the reply text.

    Fills `record` along the way and saves it at the end, also when something
    raises (the error is re-raised after that).
    """
    start = time.perf_counter()
    record.compression = compression
    try:
        # token counting (CSC-6) goes here, timed the same way into
        # record.latency_count_ms

        compress_start = time.perf_counter()
        try:
            prompt = pipeline.compress(prompt, compression)
        finally:
            record.latency_compress_ms = _ms_since(compress_start)

        logger.info("Prompt sent to %s: %r", MODEL, prompt.text)
        kwargs = {"model": MODEL, "input": prompt.text}
        if prompt.system is not None:
            kwargs["system_instruction"] = prompt.system
        upstream_start = time.perf_counter()
        try:
            interaction = client.interactions.create(**kwargs)
        finally:
            record.latency_upstream_ms = _ms_since(upstream_start)
        fill_from_interaction(record, interaction)  # tokens, status, ids (gemini_parse.py)
        return reply_text(interaction)
    except Exception as exc:
        fill_from_error(record, exc)  # sets status + error fields, then the error still goes out
        raise
    finally:
        record.latency_total_ms = _ms_since(start)
        _save(record)


def _save(record):
    """Write the request's row. A db problem is logged, never shown to the user."""
    try:
        if record.status is None:  # shouldn't happen, but insert_request needs one
            record.status = db.STATUS_INTERNAL_ERROR
        db.insert_request(record)
    except Exception:
        logger.exception("Could not save request %s to the db", record.request_id)


@app.get("/telemetry/summary")
def telemetry_summary(
    model: str | None = None,
    # must be one of db.STATUSES, anything else gets a 422
    status: Annotated[str | None, Query(pattern=f"^({'|'.join(db.STATUSES)})$")] = None,
    since: datetime | None = None,  # inclusive, UTC if no offset
    until: datetime | None = None,  # not inclusive
):
    """Counts per status, tokens, cost and latency for the stored requests."""
    try:
        return db.summarize_requests(model=model, status=status, since=since, until=until)
    except ValueError as exc:  # since/until can't be converted to UTC (e.g. 0001-01-01T00:00+01:00)
        raise HTTPException(422, str(exc)) from exc
