"""Copy what Gemini tells us into a RequestRecord (see db.py).

fill_from_interaction: for a reply that came back.
fill_from_error: for a call that raised.

Neither touches the database. main.py calls one of them, then saves the record.
Field names are from a real gemini-3.5-flash reply: interaction.status / id /
model / errors, and interaction.usage.total_input_tokens / total_output_tokens /
total_thought_tokens / total_cached_tokens / total_tokens. Any of them can be
missing, so everything is read with getattr and left None if it isn't there.
"""

import httpx

import db

MAX_ERROR_MESSAGE = 300  # keep error text short, and it shouldn't carry the prompt


def _short(text):
    text = " ".join(str(text).split())
    return text[:MAX_ERROR_MESSAGE] if text else None


def fill_from_interaction(record, interaction):
    """Fill the record from a reply that came back from interactions.create."""
    usage = getattr(interaction, "usage", None)

    record.interaction_id = getattr(interaction, "id", None) or None
    record.model = str(getattr(interaction, "model", None) or record.model)
    record.input_tokens_reported = getattr(usage, "total_input_tokens", None)
    record.output_tokens = getattr(usage, "total_output_tokens", None)
    record.thought_tokens = getattr(usage, "total_thought_tokens", None)
    record.cached_tokens = getattr(usage, "total_cached_tokens", None)
    record.total_tokens = getattr(usage, "total_tokens", None)

    gemini_status = getattr(interaction, "status", None)
    errors = getattr(interaction, "errors", None) or []
    if gemini_status == "completed" and not errors:
        record.status = db.STATUS_OK
    else:
        # came back, but not "completed" (or it listed errors)
        record.status = db.STATUS_NOT_COMPLETED
        record.error_type = str(gemini_status) if gemini_status else None
        record.error_message = _short(
            "; ".join(f"{getattr(e, 'code', '')}: {getattr(e, 'message', '')}" for e in errors)
        )


def fill_from_error(record, exc):
    """Fill the record from an exception raised while calling Gemini."""
    record.error_type = type(exc).__name__

    status_code = getattr(exc, "status_code", None)
    if isinstance(exc, httpx.TimeoutException):
        record.status = db.STATUS_TIMEOUT
    elif isinstance(exc, (httpx.TransportError, httpx.NetworkError)) or (
        type(exc).__name__ == "NoResponseError"
    ):
        record.status = db.STATUS_CONNECTION_ERROR
    elif isinstance(status_code, int):
        # the SDK's HTTP errors (4xx / 5xx, including 429)
        record.status = db.STATUS_UPSTREAM_ERROR
        record.http_status = status_code
    else:
        record.status = db.STATUS_INTERNAL_ERROR

    record.error_message = _short(getattr(exc, "message", None) or exc)
