"""General app logging (separate from the per-request rows in the db).

Call setup_logging() once at startup (main.py does this). After that, log from
any file with either:
    log_message("something happened")          # quick helper below
or the standard way:
    logger = logging.getLogger(__name__)
    logger.info("something happened")

Lines look like:
    2026-10-07 14:32:01 INFO main: something happened

Logs go to project/logs/app.log. At midnight the file is renamed to
app.log.YYYY-MM-DD and a fresh app.log is started, keeping the last 14 days.
logs/ is gitignored.
"""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE_NAME = "app.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DAYS_TO_KEEP = 14

# marks the handlers we added so calling setup_logging() again doesn't duplicate them
_HANDLER_TAG = "_token_minimizer_handler"

_app_logger = logging.getLogger("app")


def setup_logging(log_dir=None, level=logging.INFO):
    """Send all log messages to logs/app.log (rotated daily) and the terminal.

    Returns the path of the log file. log_dir is only there so tests can write
    somewhere temporary instead of the real logs/ folder.
    """
    log_dir = Path(log_dir) if log_dir else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME

    root = logging.getLogger()
    root.setLevel(level)

    # already set up for this file: nothing to do
    ours = [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]
    if any(getattr(h, "baseFilename", None) == str(log_file.resolve()) for h in ours):
        return log_file

    # set up before with a different folder (e.g. in tests): replace those handlers
    for handler in ours:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=DAYS_TO_KEEP, encoding="utf-8"
    )
    console_handler = logging.StreamHandler()

    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)

    # the http libraries log every request at INFO, which drowns out our own lines
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # fastapi dev's file watcher logs "1 change detected" for every file change,
    # including writes to logs/app.log itself, so it loops forever without this
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    return log_file


def log_message(message, level=logging.INFO):
    """Log one line; date, time and level are added automatically."""
    _app_logger.log(level, message)
