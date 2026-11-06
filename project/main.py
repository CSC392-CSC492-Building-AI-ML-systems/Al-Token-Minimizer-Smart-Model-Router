import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from google import genai
from pydantic import BaseModel, Field, field_validator, model_validator

import db
from cache import semantic
from compressor import pipeline
from compressor.compressor_service import CompressorService
from gemini_parse import fill_from_error, fill_from_interaction, reply_text
from log_setup import setup_logging
from model_data.model_package import ModelPackage
from model_data.token_counter import GeminiTokenCounter
from response_cache import CachedResponse, ResponseCache, make_cache_key

load_dotenv()  # reads GEMINI_API_KEY from project/.env (not committed)
setup_logging()  # writes to project/logs/app.log, see log_setup.py
db.init_db()  # creates telemetry.db + table if they're not there yet
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Runs when the server starts: loads the semantic cache's embedding model
    (a few seconds), so the first request doesn't wait for it."""
    if not SEMANTIC_CACHE_ENABLED:
        yield
        return
    try:
        semantic.warm_up()
    except Exception:
        logger.exception("Semantic cache not ready (is Redis running? docker compose up -d), requests will skip it")
    yield


app = FastAPI(lifespan=lifespan)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set, copy .env.example to .env and add your key")
client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash"
model_package = ModelPackage(pricing=None, tokenizer=GeminiTokenCounter(client))
# soft_output_limit when the client doesn't send one. The limit is an opt-in
# add-on: off by default, so compression "none" stays an unchanged prompt. When
# a client sends one, the prompt asks Gemini to aim for that many output tokens
# (thinking not included). Nothing is cut off, a reply that goes over is only
# flagged (see _call_gemini).
DEFAULT_SOFT_OUTPUT_LIMIT = None
DEFAULT_COMPRESSION = pipeline.validate_pipeline(os.environ.get("COMPRESSION_LEVEL", pipeline.DEFAULT_PIPELINE))


def _env_bool(name, default):
    value = os.environ.get(name, str(default)).lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"{name} must be true, false, 1, or 0")
    return value in ("true", "1")


# CACHE_ENABLED=false turns off both caches; EXACT_CACHE_ENABLED and
# SEMANTIC_CACHE_ENABLED turn off one each.
CACHE_ENABLED = _env_bool("CACHE_ENABLED", True)
SEMANTIC_CACHE_ENABLED = CACHE_ENABLED and _env_bool("SEMANTIC_CACHE_ENABLED", True)
response_cache = ResponseCache(
    enabled=CACHE_ENABLED and _env_bool("EXACT_CACHE_ENABLED", True),
    max_entries=int(os.environ.get("CACHE_MAX_ENTRIES", "256")),
    ttl_seconds=float(os.environ.get("CACHE_TTL_SECONDS", "300")),
)


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
def ask(prompt: str, soft_output_limit: Annotated[int | None, Query(gt=0)] = DEFAULT_SOFT_OUTPUT_LIMIT):
    record = db.RequestRecord(endpoint="/ask", model=MODEL)
    try:
        text = _answer(
            record, pipeline.Prompt(prompt), pipeline.resolve_stages(DEFAULT_COMPRESSION),
            soft_output_limit=soft_output_limit,
        )
    except Exception:
        raise _request_error(record) from None
    return {"text": text}


class AskRequest(BaseModel):
    prompt: str
    system: str | None = None  # sent as Gemini's system_instruction
    # Presets are shortcuts for ordered stages; callers can choose either form.
    compression: str | None = None
    stages: list[str] | None = None
    cache: Literal["off", "use", "refresh"] = "off"
    # reply length in output tokens the model is asked to stay under. Leave it out
    # (or null) for no limit
    soft_output_limit: int | None = Field(DEFAULT_SOFT_OUTPUT_LIMIT, gt=0)

    @field_validator("compression")
    @classmethod
    def _known_pipeline(cls, name):
        return pipeline.validate_pipeline(name) if name is not None else None

    @field_validator("stages")
    @classmethod
    def _known_stages(cls, names):
        if names is not None:
            pipeline.resolve_stages(names)
        return names

    @model_validator(mode="after")
    def _one_configuration(self):
        if self.compression is not None and self.stages is not None:
            raise ValueError("choose either compression or stages, not both")
        return self

    def selected_stages(self):
        return pipeline.resolve_stages(
            self.stages if self.stages is not None else self.compression or DEFAULT_COMPRESSION
        )


# HTTP code POST /ask answers with when the request fails, by db status.
# 429 is passed through as is, so a caller (like the benchmark) knows to slow down.
_HTTP_FOR_STATUS = {
    db.STATUS_UPSTREAM_ERROR: 502,
    db.STATUS_CONNECTION_ERROR: 502,
    db.STATUS_NOT_COMPLETED: 502,
    db.STATUS_TIMEOUT: 504,
    db.STATUS_INTERNAL_ERROR: 500,
}


def _request_error(record):
    """Sanitized failures for both HTTP entry points, without provider messages."""
    code = 429 if record.http_status == 429 else _HTTP_FOR_STATUS.get(record.status, 500)
    return HTTPException(code, {"status": record.status, "request_id": record.request_id})


@app.post("/ask")
def ask_post(body: AskRequest):
    """Like GET /ask, but takes a system prompt and compression stages or a preset, and
    returns the request_id of the row saved for it (also on failure, in `detail`).
    `over_limit` says whether the reply went over soft_output_limit."""
    record = db.RequestRecord(endpoint="/ask", model=MODEL)
    try:
        text = _answer(
            record, pipeline.Prompt(body.prompt, body.system), body.selected_stages(), body.cache,
            body.soft_output_limit,
        )
    except Exception:
        raise _request_error(record) from None
    return {
        "text": text,
        "request_id": record.request_id,
        "cache_status": record.cache_status,
        "cache_hit": record.cache_hit,  # which cache answered: "exact", "semantic", or None
        "over_limit": bool(record.over_limit),
    }


def _answer(
    record: db.RequestRecord,
    prompt: pipeline.Prompt,
    stages=(),
    cache="off",
    soft_output_limit: int | None = DEFAULT_SOFT_OUTPUT_LIMIT,
) -> str:
    """Compress the prompt, send it to Gemini and return the reply text.

    `soft_output_limit` is the reply length Gemini is asked to aim for (None for
    no limit), see _call_gemini. Fills `record` along the way and saves it at
    the end, also when something raises (the error is re-raised after that).
    """
    start = time.perf_counter()
    record.compression = pipeline.label(stages)
    record.compression_version = pipeline.PIPELINE_VERSION
    record.cache_status = "off" if cache == "off" else "disabled"
    record.soft_output_limit = soft_output_limit
    try:
        key = make_cache_key(
            model=MODEL, prompt=prompt.text, system=prompt.system,
            context=tuple((chunk.text, chunk.source) for chunk in prompt.context),
            compression=record.compression, pipeline_version=pipeline.fingerprint(stages),
            soft_output_limit=soft_output_limit,
        )
        record.prompt_hash = key
        # Anything besides the prompt that changes the answer, so the semantic
        # cache only reuses answers made with the same stages and output limit.
        cache_settings = f"{pipeline.fingerprint(stages)}\0{soft_output_limit}"
        use_cache = cache != "off" and (response_cache.enabled or SEMANTIC_CACHE_ENABLED)
        if use_cache:
            record.cache_status = "refresh" if cache == "refresh" else "miss"
            cached = response_cache.get(key) if cache == "use" else None
            if cached is not None:
                _mark_cache_hit(record, "exact")
                record.input_tokens_raw = cached.input_tokens_raw
                record.input_tokens_compressed = cached.input_tokens_compressed
                return cached.text
            # Exact miss: try an earlier prompt that means the same
            if cache == "use" and SEMANTIC_CACHE_ENABLED:
                cache_start = time.perf_counter()
                similar = _cache_lookup(prompt, cache_settings)
                record.latency_cache_ms = _ms_since(cache_start)
                if similar is not None:
                    _mark_cache_hit(record, "semantic")
                    return similar

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

        original_prompt = prompt  # the semantic cache works on the prompt before compression
        prompt = compressor_service.compressed_prompt
        logger.info("Request %s sent to %s (compression=%s)", record.request_id, MODEL, record.compression)
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
        if use_cache and record.status == db.STATUS_OK and text.strip():
            response_cache.put(key, CachedResponse(
                text=text, input_tokens_raw=record.input_tokens_raw,
                input_tokens_compressed=record.input_tokens_compressed,
                output_tokens=record.output_tokens,
            ))
            if SEMANTIC_CACHE_ENABLED:
                _cache_store(original_prompt, cache_settings, text)
        return text
    except Exception as exc:
        fill_from_error(record, exc)  # sets status + error fields, then the error still goes out
        raise
    finally:
        # Free-form provider/exception messages can echo prompts and answers,
        # even when truncated. Persist status, HTTP code and error type only.
        record.error_message = None
        record.latency_total_ms = _ms_since(start)
        _save(record)


def _mark_cache_hit(record, kind):
    """Fill `record` for an answer that came from the `kind` cache ("exact" or "semantic")."""
    record.cache_status = "hit"
    record.cache_hit = kind
    record.status = db.STATUS_OK
    # These fields measure newly consumed upstream usage, not the
    # original generation replayed from memory.
    for column in ("input_tokens_reported", "output_tokens", "thought_tokens", "cached_tokens", "total_tokens"):
        setattr(record, column, 0)
    record.cost_usd = 0.0
    record.latency_compress_ms = record.latency_count_ms = record.latency_upstream_ms = 0.0


def _cache_lookup(prompt, settings):
    """The semantic cache's answer, or None. A cache problem (e.g. Redis not
    running) is logged and counts as a miss, so the request still goes to Gemini."""
    try:
        return semantic.lookup(prompt, MODEL, settings)
    except Exception as exc:
        # Only the error type: messages can include request data
        logger.error("Semantic cache lookup failed (%s)", type(exc).__name__)
        return None


def _cache_store(prompt, settings, text):
    """Save the answer in the semantic cache. A cache problem is logged, never shown to the user."""
    try:
        semantic.store(prompt, MODEL, settings, text)
    except Exception as exc:
        logger.error("Could not save the answer to the semantic cache (%s)", type(exc).__name__)


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
        # Exception messages can include request data. Keep logs metadata-only.
        logger.error("Could not save request %s to the db", record.request_id)


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
