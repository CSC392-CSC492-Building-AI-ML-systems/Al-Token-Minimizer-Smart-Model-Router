"""Bounded, process-local exact-match response cache.

Only digests are used as keys. Answers live in memory until eviction, cleanup or
process exit; nothing in this module writes request or response text to disk.
The caller decides whether an upstream result is complete and safe to cache.
"""

import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock


def make_cache_key(
    *,
    model: str,
    prompt: str,
    system: str | None,
    compression: str,
    pipeline_version: str,
    context: tuple[tuple[str, str], ...] = (),
    soft_output_limit: int | None = None,
) -> str:
    """Hash the complete request identity, without normalizing any text.

    ``pipeline_version`` must identify the effective stages and their settings.
    Bump it when transformation behavior changes. If generation options are
    added to the API later, include those options in this identity too.
    ``soft_output_limit`` is one: it changes the instruction Gemini gets, so an
    answer written for one limit isn't reused for another.
    """
    # Keep chunk boundaries and source labels: equal rendered text can pass
    # through code stages differently when fences span separate chunks.
    identity = ["exact-response-v1", model, prompt, system, context, compression, pipeline_version, soft_output_limit]
    encoded = json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    """Immutable answer and reusable counts, not new upstream consumption.

    On a hit the caller must record zero newly consumed upstream tokens. The
    pre/post compression counts describe the prompt, and ``output_tokens``
    describes the original answer rather than a second generation.
    """

    text: str = field(repr=False)
    input_tokens_raw: int
    input_tokens_compressed: int
    output_tokens: int | None = None


@dataclass(frozen=True)
class _Entry:
    response: CachedResponse = field(repr=False)
    expires_at: float


class ResponseCache:
    """Thread-safe LRU cache with a fixed TTL measured by a monotonic clock.

    A hit updates recency but does not extend an answer's lifetime. A disabled
    cache never retains answers. Separate worker processes have separate caches.
    Expired answers are never returned and are removed lazily on cache access.
    Concurrent misses may each call the upstream model; this cache does not
    serialize generation.
    """

    def __init__(
        self,
        max_entries: int = 256,
        ttl_seconds: float = 300,
        *,
        enabled: bool = True,
        max_entry_bytes: int = 1024 * 1024,
        clock: Callable[[], float] = time.monotonic,
    ):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("cache max_entries must be a positive integer")
        if isinstance(ttl_seconds, bool) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("cache ttl_seconds must be finite and positive")
        if isinstance(max_entry_bytes, bool) or not isinstance(max_entry_bytes, int) or max_entry_bytes <= 0:
            raise ValueError("cache max_entry_bytes must be a positive integer")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self.max_entry_bytes = max_entry_bytes
        self._clock = clock
        self._lock = Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def get(self, key: str) -> CachedResponse | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self._clock():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.response

    def put(self, key: str, response: CachedResponse) -> None:
        if not self.enabled:
            return
        if not isinstance(response, CachedResponse):
            raise TypeError("cache response must be a CachedResponse")
        with self._lock:
            # Bound retained answer bytes as well as entry count. A refreshed
            # oversized answer invalidates an older entry for the same key.
            if len(response.text.encode("utf-8", errors="surrogatepass")) > self.max_entry_bytes:
                self._entries.pop(key, None)
                return
            now = self._clock()
            self._purge_expired(now)
            self._entries[key] = _Entry(response, now + self.ttl_seconds)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired(self._clock())
            return len(self._entries)

    def _purge_expired(self, now: float) -> None:
        # Recency and expiry order differ because reads do not renew the TTL.
        for key, entry in list(self._entries.items()):
            if entry.expires_at <= now:
                del self._entries[key]
