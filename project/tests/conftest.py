"""Loaded by pytest before the test files, so it runs before they import main.

main.py calls setup_logging() when it's imported, which would send every test's
prompts and tracebacks to the real logs/app.log, and anything that doesn't pick
a db would use the real telemetry.db. Point both at a temporary folder instead.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import log_setup

_tmp = Path(tempfile.mkdtemp(prefix="token-minimizer-tests-"))
atexit.register(shutil.rmtree, _tmp, ignore_errors=True)  # don't leave a folder behind per run
log_setup.LOG_DIR = _tmp / "logs"
os.environ.setdefault("TELEMETRY_DB_PATH", str(_tmp / "telemetry.db"))


import pytest  # noqa: E402

from cache import semantic  # noqa: E402


@pytest.fixture(autouse=True)
def no_semantic_cache(monkeypatch):
    """Tests never touch Redis or load the embedding model: every lookup
    misses, nothing is stored and server startup doesn't load the model. Tests of the cache itself patch these again."""
    monkeypatch.setattr(semantic, "lookup", lambda prompt, model: None)
    monkeypatch.setattr(semantic, "store", lambda prompt, model, answer: None)
    monkeypatch.setattr(semantic, "warm_up", lambda: None)
