import logging
import os
import time
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from google import genai
from pydantic import BaseModel, Field, field_validator

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
# soft_output_limit when the client doesn't send one: the reply length (in
# output tokens, thinking not included) the prompt asks Gemini to aim for.
# Nothing is cut off, a reply that goes over is only flagged (see _call_gemini).
DEFAULT_SOFT_OUTPUT_LIMIT = 1024


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
def ask(prompt: str, soft_output_limit: Annotated[int, Query(gt=0)] = DEFAULT_SOFT_OUTPUT_LIMIT):
    record = db.RequestRecord(endpoint="/ask", model=MODEL)
    return {"text": _answer(record, pipeline.Prompt(prompt), [], soft_output_limit)}


class AskRequest(BaseModel):
    prompt: str
    system: str | None = None  # sent as Gemini's system_instruction
    stages: list[str] = []  # compression stages to run, in order (names from compressor/stages/)
    # reply length in output tokens the model is asked to stay under. null for no limit
    soft_output_limit: int | None = Field(DEFAULT_SOFT_OUTPUT_LIMIT, gt=0)

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
    returns the request_id of the row saved for it (also on failure, in `detail`).
    `over_limit` says whether the reply went over soft_output_limit."""
    record = db.RequestRecord(endpoint="/ask", model=MODEL)
    try:
        text = _answer(record, pipeline.Prompt(body.prompt, body.system), body.stages, body.soft_output_limit)
    except Exception:
        code = 429 if record.http_status == 429 else _HTTP_FOR_STATUS.get(record.status, 500)
        raise HTTPException(code, {"status": record.status, "request_id": record.request_id}) from None
    return {"text": text, "request_id": record.request_id, "over_limit": bool(record.over_limit)}


def _answer(
    record: db.RequestRecord,
    prompt: pipeline.Prompt,
    stages: list[str],
    soft_output_limit: int | None = DEFAULT_SOFT_OUTPUT_LIMIT,
) -> str:
    """Compress the prompt, send it to Gemini and return the reply text.

    `stages` are the names of the compression stages to run, in order, e.g.
    ["passthrough"]; an empty list sends the prompt unchanged.
    `soft_output_limit` is the reply length Gemini is asked to aim for (None for
    no limit), see _call_gemini. Fills `record` along the way and saves it at
    the end, also when something raises (the error is re-raised after that).
    """
    start = time.perf_counter()
    record.compression = pipeline.label(stages)  # e.g. "a+b", "none" for no stages
    record.soft_output_limit = soft_output_limit
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
        upstream_start = time.perf_counter()
        try:
            text = _call_gemini(record, prompt, soft_output_limit)
        finally:
            record.latency_upstream_ms = _ms_since(upstream_start)
        logger.log(
            logging.WARNING if record.over_limit else logging.INFO,
            "Reply %s: %s output + %s thought tokens, soft limit %s%s",
            record.request_id,
            record.output_tokens,
            record.thought_tokens,
            soft_output_limit,
            f" (OVER by {record.over_limit_by})" if record.over_limit else "",
        )
        return text
    except Exception as exc:
        fill_from_error(record, exc)  # sets status + error fields, then the error still goes out
        raise
    finally:
        record.latency_total_ms = _ms_since(start)
        _save(record)


def _call_gemini(record, prompt, soft_output_limit):
    """Send the prompt, asking Gemini to keep the reply under soft_output_limit
    tokens, and return the reply text.

    There's no hard cap, so a reply is never cut off (cut-off JSON or code would
    be broken). One that goes over anyway is sent back whole and flagged in the
    record: over_limit = 1 and over_limit_by = how many output tokens over.
    """
    system = prompt.system
    if soft_output_limit is not None:
        hint = length_hint(soft_output_limit)
        system = f"{system}\n\n{hint}" if system else hint
    kwargs = {"model": MODEL, "input": prompt.render()}
    if system is not None:
        kwargs["system_instruction"] = system

    interaction = client.interactions.create(**kwargs)
    fill_from_interaction(record, interaction)  # tokens, status, ids (gemini_parse.py)
    if soft_output_limit is not None and record.output_tokens is not None:
        record.over_limit_by = max(record.output_tokens - soft_output_limit, 0)
        record.over_limit = int(record.over_limit_by > 0)
    return reply_text(interaction)


def length_hint(soft_output_limit):
    """The line added to the system prompt that asks for a reply under the limit."""
    words = soft_output_limit * 3 // 4  # roughly 0.75 English words per token
    return (
        f"Aim to keep your answer under {soft_output_limit} tokens (about {words} words). "
        "This is a guideline, not a hard limit: always finish your answer, and never "
        "cut code, JSON or other structured output short to stay under it."
    )


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
