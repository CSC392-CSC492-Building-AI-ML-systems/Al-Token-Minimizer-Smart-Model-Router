"""Exact identity, expiry, LRU bounds and concurrent access without an API key."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

from response_cache import CachedResponse, ResponseCache, make_cache_key


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.cache = ResponseCache(max_entries=2, ttl_seconds=10, clock=lambda: self.now)
        self.answer = CachedResponse("private answer", 12, 8, 3)

    def test_exact_key_covers_each_request_setting(self):
        identity = dict(model="gemini-test", prompt="private prompt", system=None,
                        compression="none", pipeline_version="v1")
        key = make_cache_key(**identity)
        self.assertEqual(key, make_cache_key(**dict(reversed(list(identity.items())))))
        self.assertRegex(key, r"^[a-f0-9]{64}$")
        variants = {"model": "another-model", "prompt": "private prompt ", "system": "",
                    "compression": "light", "pipeline_version": "v2"}
        for name, value in variants.items():
            with self.subTest(name=name):
                self.assertNotEqual(key, make_cache_key(**{**identity, name: value}))

    def test_key_does_not_conflate_concatenation_or_unicode(self):
        identity = dict(model="test", compression="none", pipeline_version="v1")
        self.assertNotEqual(
            make_cache_key(**identity, prompt="ab", system="c"),
            make_cache_key(**identity, prompt="a", system="bc"),
        )
        self.assertNotEqual(
            make_cache_key(**identity, prompt="é", system=None),
            make_cache_key(**identity, prompt="e\u0301", system=None),
        )

    def test_hit_returns_immutable_answer_and_hides_text_in_repr(self):
        self.cache.put("key", self.answer)
        hit = self.cache.get("key")
        self.assertEqual(hit, self.answer)
        with self.assertRaises(FrozenInstanceError):
            hit.text = "changed"
        self.assertEqual(self.cache.get("key").text, "private answer")
        self.assertNotIn("private answer", repr(hit))
        self.assertNotIn("private answer", repr(self.cache))

    def test_expiry_is_absolute_even_after_hit(self):
        self.cache.put("key", self.answer)
        self.now = 109.999
        self.assertIsNotNone(self.cache.get("key"))
        self.now = 110
        self.assertIsNone(self.cache.get("key"))
        self.assertEqual(len(self.cache), 0)

    def test_read_updates_lru_order(self):
        self.cache.put("first", self.answer)
        self.cache.put("second", self.answer)
        self.cache.get("first")
        self.cache.put("third", self.answer)
        self.assertEqual(len(self.cache), 2)
        self.assertIsNone(self.cache.get("second"))
        self.assertIsNotNone(self.cache.get("first"))
        self.assertIsNotNone(self.cache.get("third"))

    def test_expired_recent_entry_does_not_evict_valid_old_entry(self):
        self.cache.put("expires-first", self.answer)
        self.now = 105
        self.cache.put("still-valid", self.answer)
        self.cache.get("expires-first")
        self.now = 110
        self.cache.put("new", self.answer)
        self.assertIsNone(self.cache.get("expires-first"))
        self.assertIsNotNone(self.cache.get("still-valid"))
        self.assertIsNotNone(self.cache.get("new"))

    def test_overwrite_replaces_response_and_resets_expiry(self):
        self.cache.put("key", self.answer)
        self.now = 105
        replacement = CachedResponse("new answer", 15, 9)
        self.cache.put("key", replacement)
        self.now = 110
        self.assertEqual(self.cache.get("key"), replacement)
        self.assertEqual(len(self.cache), 1)
        self.now = 115
        self.assertIsNone(self.cache.get("key"))

    def test_disabled_cache_never_retains_response(self):
        cache = ResponseCache(enabled=False)
        cache.put("key", self.answer)
        self.assertIsNone(cache.get("key"))
        self.assertEqual(len(cache), 0)

    def test_oversized_utf8_answer_is_not_retained(self):
        cache = ResponseCache(max_entry_bytes=4)
        cache.put("key", CachedResponse("okay", 1, 1))
        self.assertIsNotNone(cache.get("key"))
        cache.put("key", CachedResponse("ééé", 1, 1))
        self.assertIsNone(cache.get("key"))
        self.assertEqual(len(cache), 0)

    def test_clear_and_len_remove_stale_entries(self):
        self.cache.put("key", self.answer)
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)
        self.assertIsNone(self.cache.get("key"))
        self.cache.put("other", self.answer)
        self.now = 120
        self.assertEqual(len(self.cache), 0)

    def test_invalid_bounds_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(max_entries=value), self.assertRaises(ValueError):
                ResponseCache(max_entries=value)
            with self.subTest(max_entry_bytes=value), self.assertRaises(ValueError):
                ResponseCache(max_entry_bytes=value)
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(ttl_seconds=value), self.assertRaises(ValueError):
                ResponseCache(ttl_seconds=value)

    def test_concurrent_access_preserves_capacity_and_value_integrity(self):
        cache = ResponseCache(max_entries=8)

        def use_cache(number):
            key = str(number % 12)
            value = CachedResponse(key, number, number)
            cache.put(key, value)
            hit = cache.get(key)
            if hit is not None:
                self.assertEqual(hit.text, key)
                self.assertEqual(hit.input_tokens_raw, hit.input_tokens_compressed)
            self.assertLessEqual(len(cache), 8)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(use_cache, range(1000)))
        self.assertLessEqual(len(cache), 8)


if __name__ == "__main__":
    unittest.main()
