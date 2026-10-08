"""Check that the db client can write a row, read it back, and delete it.

Uses the real telemetry.db (created if missing) and only touches the one row it
inserts, so it's safe to run on a db that already has data.

Run from the project folder:
    uv run python -m unittest tests.db_tests -v
"""

import sqlite3
import unittest

import db


class TestWriteReadDelete(unittest.TestCase):
    def test_write_then_check_then_delete(self):
        db.init_db()

        record = db.RequestRecord(
            endpoint="/db-test",
            model="test-model",
            prompt_hash="test-hash",
            input_tokens_raw=120,
            latency_total_ms=12.5,
            status=db.STATUS_OK,
        )
        request_id = db.insert_request(record)

        try:
            # it's in the db, with the values we wrote
            rows = db.fetch_requests(request_id=request_id)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["endpoint"], "/db-test")
            self.assertEqual(row["input_tokens_raw"], 120)
            self.assertEqual(row["latency_total_ms"], 12.5)
            self.assertEqual(row["status"], db.STATUS_OK)
            self.assertIsNone(row["output_tokens"])  # never set, so NULL
        finally:
            # always clean up, even if a check above failed
            deleted = db.delete_request(request_id)

        self.assertEqual(deleted, 1)
        self.assertEqual(db.fetch_requests(request_id=request_id), [])


class TestBadRowsAreRejected(unittest.TestCase):
    """None of these get as far as writing a row, so the real db is left as it was."""

    def setUp(self):
        db.init_db()

    def test_status_not_in_the_list(self):
        with self.assertRaises(sqlite3.IntegrityError):
            db.insert_request({"status": "great"})

    def test_status_missing(self):
        with self.assertRaises(ValueError):
            db.insert_request({"endpoint": "/db-test"})

    def test_unknown_column(self):  # e.g. the raw prompt, which is never stored
        with self.assertRaises(ValueError):
            db.insert_request({"status": db.STATUS_OK, "prompt": "hello"})


if __name__ == "__main__":
    unittest.main()
