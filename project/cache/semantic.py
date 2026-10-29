"""Semantic cache: answers saved in Redis, looked up by what a prompt means
rather than its exact words, so "how can I reset my password" finds the
answer stored for "How do I reset my password?".

    lookup(prompt, model)          -> the cached answer, or None
    store(prompt, model, answer)   -> save an answer for next time

Only the question (prompt.text) is embedded and compared. A hit also needs the
same model, system prompt and knowledge-base chunks, so a changed document or
system prompt never returns an old answer. Pass the prompt from before
compression, so changing the stages doesn't make every lookup miss.

Needs Redis Stack running (docker compose up -d in this folder). REDIS_URL in
.env points somewhere else; the default is redis://localhost:6379.
"""

import hashlib
import os

from compressor.pipeline import Prompt

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DISTANCE_THRESHOLD = 0.1  # cosine distance, about similarity 0.9 and up; tune with the benchmark
TTL_SECONDS = 7 * 24 * 60 * 60  # entries expire after a week

_cache = None
_vectorizer = None  # kept apart from _cache, so a failed Redis connection doesn't reload the model


def _get_vectorizer():
    """The embedding model, loaded on first use (a few seconds)."""
    global _vectorizer
    if _vectorizer is None:
        from redisvl.utils.vectorize import HFTextVectorizer

        _vectorizer = HFTextVectorizer(EMBEDDING_MODEL)
    return _vectorizer


def _get_cache():
    """The SemanticCache, made on first use, so importing this file doesn't
    need Redis running or load the embedding model."""
    global _cache
    if _cache is None:
        from redisvl.extensions.cache.llm import SemanticCache

        _cache = SemanticCache(
            name="semantic_cache",
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
            distance_threshold=DISTANCE_THRESHOLD,
            ttl=TTL_SECONDS,
            vectorizer=_get_vectorizer(),
            filterable_fields=[{"name": "scope", "type": "tag"}],
        )
    return _cache


def warm_up() -> None:
    """Load the model and connect to Redis now, so the first request doesn't
    wait for it. Raises if Redis can't be reached (the model stays loaded)."""
    _get_vectorizer().embed("warm up")  # the first embedding is slower than the rest
    _get_cache()


def _question(prompt: Prompt) -> str:
    return " ".join(prompt.text.split())


def _scope(prompt: Prompt, model: str) -> str:
    """Hash of everything but the question that changes the answer."""
    parts = [model, prompt.system or ""] + [f"{c.source}\n{c.text}" for c in prompt.context]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def lookup(prompt: Prompt, model: str) -> str | None:
    """The answer stored for a prompt that means the same, or None."""
    from redisvl.query.filter import Tag

    hits = _get_cache().check(
        prompt=_question(prompt),
        filter_expression=Tag("scope") == _scope(prompt, model),
    )
    return hits[0]["response"] if hits else None


def store(prompt: Prompt, model: str, answer: str) -> None:
    """Save `answer` for `prompt`. Only store successful replies."""
    _get_cache().store(
        prompt=_question(prompt),
        response=answer,
        filters={"scope": _scope(prompt, model)},
    )
