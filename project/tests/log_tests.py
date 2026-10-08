"""Check that a logged message ends up in the log file.

Writes to a temporary folder, not the real logs/, and cleans up after itself.

Run from the project folder:
    uv run python -m unittest tests.log_tests -v
"""

import logging
import tempfile
import unittest

from log_setup import log_message, setup_logging


class TestLogToFile(unittest.TestCase):
    def test_message_lands_in_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_file = setup_logging(log_dir=tmp)
            try:
                log_message("hello from the log test")

                for handler in logging.getLogger().handlers:
                    handler.flush()
                contents = log_file.read_text(encoding="utf-8")

                self.assertIn("hello from the log test", contents)
                self.assertIn("INFO", contents)
            finally:
                # close the file handlers so Windows lets the temp folder be deleted
                root = logging.getLogger()
                for handler in list(root.handlers):
                    root.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    unittest.main()
