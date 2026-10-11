"""Copy what Gemini tells us into a RequestRecord (see db.py).

fill_from_interaction: for a reply that came back.
fill_from_error: for a call that raised.
reply_text: the answer to send back to the user.

Neither fill_ touches the database. main.py calls one of them, then saves the record.
Field names are from a real gemini-3.5-flash reply: interaction.status / id /
model / errors, and interaction.usage.total_input_tokens / total_output_tokens /
total_thought_tokens / total_cached_tokens / total_tokens. Any of them can be
missing, so everything is read with _get and left None if it isn't there.
When a reply doesn't match the SDK's schema (e.g. it has no status, or a token
count isn't a number), create() hands back a plain dict instead, which _get
reads too.
"""

import httpx

import db

MAX_ERROR_MESSAGE = 300  # keep error text short, and it shouldn't carry the prompt


def _short(text):
    text = " ".join(str(text).split())
    return text[:MAX_ERROR_MESSAGE] if text else None


def _join(*parts):
    return ": ".join(str(part) for part in parts if part)


def _get(obj, name):
    """obj.name, or obj[name] for a dict. None if it's missing."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int_or_none(value):
    # a dict reply can have anything here, and the summary adds these up
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def reply_text(interaction):
    text = _get(interaction, "output_text")
    return text if isinstance(text, str) else ""  # a dict reply has no output_text if there's no text


def fill_from_interaction(record, interaction):
    """Fill the record from a reply that came back from interactions.create."""
    usage = _get(interaction, "usage")

    interaction_id = _get(interaction, "id")
    # a dict reply can have anything here too, and a list or dict id couldn't be saved at all
    record.interaction_id = interaction_id if isinstance(interaction_id, str) and interaction_id else None
    record.model = str(_get(interaction, "model") or record.model)
    record.input_tokens_reported = _int_or_none(_get(usage, "total_input_tokens"))
    record.output_tokens = _int_or_none(_get(usage, "total_output_tokens"))
    record.thought_tokens = _int_or_none(_get(usage, "total_thought_tokens"))
    record.cached_tokens = _int_or_none(_get(usage, "total_cached_tokens"))
    record.total_tokens = _int_or_none(_get(usage, "total_tokens"))

    gemini_status = _get(interaction, "status")
    errors = _get(interaction, "errors") or []
    if not isinstance(errors, list):
        errors = [errors]
    if gemini_status == "completed" and not errors:
        record.status = db.STATUS_OK
    else:
        # came back, but not "completed" (or it listed errors)
        record.status = db.STATUS_NOT_COMPLETED
        record.error_type = str(gemini_status) if gemini_status else None
        record.error_message = _short(
            "; ".join(_join(_get(e, "code"), _get(e, "message")) or str(e) for e in errors)
        )


def fill_from_error(record, exc):
    """Fill the record from an exception raised while calling Gemini."""
    record.error_type = type(exc).__name__
    record.error_message = _short(getattr(exc, "message", None) or exc)

    # the SDK doesn't let httpx's errors out, it raises its own (APITimeoutError
    # is a kind of APIConnectionError, so it's checked first)
    names = {cls.__name__ for cls in type(exc).__mro__}
    status_code = getattr(exc, "status_code", None)
    if isinstance(exc, httpx.TimeoutException) or "APITimeoutError" in names:
        record.status = db.STATUS_TIMEOUT
    elif isinstance(exc, (httpx.TransportError, httpx.NetworkError)) or (
        names & {"APIConnectionError", "NoResponseError"}
    ):
        record.status = db.STATUS_CONNECTION_ERROR
    elif isinstance(status_code, int):
        # the SDK's HTTP errors (4xx / 5xx, including 429), or a 2xx reply it couldn't read
        record.status = db.STATUS_UPSTREAM_ERROR
        record.http_status = status_code
        # its message has the whole response body in it, which for a 2xx is the
        # answer itself, so only keep Gemini's own error ({"error": {...}}) if there is one
        error = _get(getattr(exc, "body", None), "error")
        record.error_message = _short(_join(_get(error, "status"), _get(error, "message")))
    else:
        record.status = db.STATUS_INTERNAL_ERROR
