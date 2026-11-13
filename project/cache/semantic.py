"""Semantic cache: answers saved in Redis, looked up by what a prompt means
rather than its exact words, so "how can I reset my password" finds the
answer stored for "How do I reset my password?".

    lookup(prompt, model, settings)          -> the cached answer, or None
    store(prompt, model, settings, answer)   -> save an answer for next time

Only the question (prompt.text) is embedded and compared. A hit also needs the
same model, system prompt, knowledge-base chunks and `settings` (a string for
anything else that changes the answer, like the compression stages and the
output limit), so a changed document or system prompt never returns an old
answer. Pass the prompt from before compression.

Needs Redis Stack running (docker compose up -d in this folder). REDIS_URL in
.env points somewhere else; the default is redis://localhost:6379.
In .env: SEMANTIC_CACHE_TTL_SECONDS sets how long entries last (default 5
hours), SEMANTIC_CACHE_DISTANCE_THRESHOLD how close in meaning a question has
to be for a hit (default 0.1), SEMANTIC_CACHE_ENABLED=false turns it off.

To see what's in the cache, from the project folder:
    uv run python -m cache.semantic list              # newest entries first
    uv run python -m cache.semantic check "question"  # closest entries and their distance
    uv run python -m cache.semantic clear             # delete every entry (asks first)
"""

import argparse
import hashlib
import os
import time

from compressor.pipeline import Prompt

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DISTANCE_THRESHOLD = 0.1  # cosine distance, about similarity 0.9 and up; tune with the benchmark
DEFAULT_TTL_SECONDS = 5 * 60 * 60  # entries expire after 5 hours, unless .env says otherwise
CACHE_NAME = "semantic_cache"  # Redis index name, and the prefix of every key

_cache = None
_vectorizer = None  # kept apart from _cache, so a failed Redis connection doesn't reload the model


def _redis_url():
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


# The two settings below are read when the cache is made, after main.py loads .env.


def _ttl_seconds():
    """SEMANTIC_CACHE_TTL_SECONDS from .env. Only affects entries stored from then on."""
    ttl = int(os.environ.get("SEMANTIC_CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS))
    if ttl <= 0:
        raise ValueError("SEMANTIC_CACHE_TTL_SECONDS must be a positive number of seconds")
    return ttl


def _distance_threshold():
    """SEMANTIC_CACHE_DISTANCE_THRESHOLD from .env: 0 only matches the same
    meaning exactly, higher allows looser matches (cosine distance, 0 to 2)."""
    threshold = float(os.environ.get("SEMANTIC_CACHE_DISTANCE_THRESHOLD", DEFAULT_DISTANCE_THRESHOLD))
    if not 0 <= threshold <= 2:
        raise ValueError("SEMANTIC_CACHE_DISTANCE_THRESHOLD must be between 0 and 2")
    return threshold


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
            name=CACHE_NAME,
            redis_url=_redis_url(),
            distance_threshold=_distance_threshold(),
            ttl=_ttl_seconds(),
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


def _scope(prompt: Prompt, model: str, settings: str = "") -> str:
    """Hash of everything but the question that changes the answer."""
    parts = [model, settings, prompt.system or ""] + [f"{c.source}\n{c.text}" for c in prompt.context]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def lookup(prompt: Prompt, model: str, settings: str = "") -> str | None:
    """The answer stored for a prompt that means the same, or None."""
    from redisvl.query.filter import Tag

    hits = _get_cache().check(
        prompt=_question(prompt),
        filter_expression=Tag("scope") == _scope(prompt, model, settings),
    )
    return hits[0]["response"] if hits else None


def store(prompt: Prompt, model: str, settings: str, answer: str) -> None:
    """Save `answer` for `prompt`. Only store successful replies."""
    _get_cache().store(
        prompt=_question(prompt),
        response=answer,
        filters={"scope": _scope(prompt, model, settings)},
    )


# ---------------------------------------------------------------- command line


def _short(text, width):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _ago(seconds):
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size:.0f}{unit}"
    return f"{seconds:.0f}s"


def list_entries(limit=None):
    """Every entry as a dict (prompt, response, scope, inserted_at, ttl, key), newest first.
    Reads Redis directly, so it doesn't load the embedding model."""
    import redis

    client = redis.Redis.from_url(_redis_url(), decode_responses=True)
    entries = []
    for key in client.scan_iter(f"{CACHE_NAME}:*"):
        prompt, response, scope, inserted_at = client.hmget(key, "prompt", "response", "scope", "inserted_at")
        if prompt is None:  # not a cache entry
            continue
        entries.append({
            "key": key,
            "prompt": prompt,
            "response": response,
            "scope": scope,
            "inserted_at": float(inserted_at or 0),
            "ttl": client.ttl(key),
        })
    entries.sort(key=lambda entry: entry["inserted_at"], reverse=True)
    return entries[:limit]


def _cli_list(args):
    entries = list_entries(args.limit)
    if not entries:
        print("The cache is empty.")
        return
    now = time.time()
    print(f"{'age':>5} {'expires':>8}  {'scope':<8}  {'prompt':<50}  response")
    for entry in entries:
        expires = _ago(entry["ttl"]) if entry["ttl"] >= 0 else "never"
        print(
            f"{_ago(now - entry['inserted_at']):>5} {expires:>8}  {(entry['scope'] or '')[:8]:<8}  "
            f"{_short(entry['prompt'], 50):<50}  {_short(entry['response'], 60)}"
        )
    print(f"\n{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} shown. Same scope = same model, system prompt and chunks.")


def _cli_check(args):
    question = " ".join(args.question.split())
    hits = _get_cache().check(prompt=question, num_results=args.limit, distance_threshold=2.0)
    if not hits:
        print("The cache is empty.")
        return
    threshold = _distance_threshold()
    print(f"Closest entries to {question!r} (a hit needs distance <= {threshold} and the same scope):\n")
    print(f"{'distance':>8}  {'':<6}  {'scope':<8}  {'prompt':<50}  response")
    for hit in hits:
        distance = float(hit["vector_distance"])
        mark = "close" if distance <= threshold else ""
        print(
            f"{distance:>8.3f}  {mark:<6}  {hit.get('scope', '')[:8]:<8}  "
            f"{_short(hit['prompt'], 50):<50}  {_short(hit['response'], 60)}"
        )


def _cli_clear(args):
    count = len(list_entries())
    if not count:
        print("The cache is already empty.")
        return
    if not args.yes and input(f"Delete all {count} cache entries? [y/N] ").strip().lower() != "y":
        print("Nothing deleted.")
        return
    _get_cache().clear()
    print(f"Deleted {count} entries.")


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Look at or empty the semantic cache in Redis.")
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list", help="show the cached entries, newest first")
    list_parser.add_argument("--limit", type=int, help="show at most this many")
    list_parser.set_defaults(run=_cli_list)
    check_parser = commands.add_parser("check", help="show the entries closest in meaning to a question")
    check_parser.add_argument("question")
    check_parser.add_argument("--limit", type=int, default=5, help="how many to show (default 5)")
    check_parser.set_defaults(run=_cli_check)
    clear_parser = commands.add_parser("clear", help="delete every entry")
    clear_parser.add_argument("--yes", action="store_true", help="don't ask first")
    clear_parser.set_defaults(run=_cli_clear)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # REDIS_URL and HF_HUB_OFFLINE from .env
    cli()
