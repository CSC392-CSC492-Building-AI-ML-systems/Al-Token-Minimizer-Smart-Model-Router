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
