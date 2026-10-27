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
from compressor.compressor_service import CompressorService
from gemini_parse import fill_from_error, fill_from_interaction, reply_text
from log_setup import setup_logging
from model_data.model_package import ModelPackage
from model_data.token_counter import GeminiTokenCounter

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
model_package = ModelPackage(pricing=None, tokenizer=GeminiTokenCounter(client))


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
    return {"text": _answer(record, pipeline.Prompt(prompt), [])}


class AskRequest(BaseModel):
    prompt: str
    system: str | None = None  # sent as Gemini's system_instruction
    stages: list[str] = []  # compression stages to run, in order (names from compressor/stages/)

    @field_validator("stages")
    @classmethod
    def _known_stages(cls, stages):
        if unknown := pipeline.unknown_stages(stages):
            raise ValueError(f"unknown stages {unknown}, expected some of {sorted(pipeline.STAGES)}")
        return stages


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
    """Like GET /ask, but takes a system prompt and the compression stages to run, and
    returns the request_id of the row saved for it (also on failure, in `detail`)."""
    record = db.RequestRecord(endpoint="/ask", model=MODEL)
    try:
        text = _answer(record, pipeline.Prompt(body.prompt, body.system), body.stages)
    except Exception:
        code = 429 if record.http_status == 429 else _HTTP_FOR_STATUS.get(record.status, 500)
        raise HTTPException(code, {"status": record.status, "request_id": record.request_id}) from None
    return {"text": text, "request_id": record.request_id}


def _answer(record: db.RequestRecord, prompt: pipeline.Prompt, stages: list[str]) -> str:
    """Compress the prompt, send it to Gemini and return the reply text.

    `stages` are the names of the compression stages to run, in order, e.g.
    ["passthrough"]; an empty list sends the prompt unchanged. Fills
    `record` along the way and saves it at the end, also when something raises
    (the error is re-raised after that).
    """
    start = time.perf_counter()
    record.compression = pipeline.label(stages)  # e.g. "a+b", "none" for no stages
    try:
        # Run the stages the request listed, in order. Each stage takes the
        # Prompt and returns a new, shorter one. E.g. with
        # {"stages": ["passthrough"]} this calls passthrough
        # (compressor/stages/example.py), which hands the prompt back as is; a
        # real stage would shorten it here. No stages sends the prompt unchanged.
        compress_start = time.perf_counter()
        try:
            compressor_service = CompressorService(prompt, model_package, stages=stages)
        finally:
            record.latency_compress_ms = _ms_since(compress_start)

        count_start = time.perf_counter()
        try:
            token_info = compressor_service.get_token_information()
            record.input_tokens_raw = token_info["uncompressed_input_tokens"]
            record.input_tokens_compressed = token_info["compressed_input_tokens"]
        finally:
            record.latency_count_ms = _ms_since(count_start)

        prompt = compressor_service.compressed_prompt
        logger.info("Prompt sent to %s: %r", MODEL, prompt.render())
        kwargs = {"model": MODEL, "input": prompt.render()}
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
