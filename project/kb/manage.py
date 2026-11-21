"""Add and remove knowledge-base pages, and rebuild the index after a change.

Used by the /kb endpoints in main.py. Base pages (kb/corpus/, in git) can be
listed but not changed here; pages added by users go to kb/user_corpus/
(gitignored). A page is stored as Markdown, so plain text uploads work too.

After a change, rebuild() re-embeds the corpus for the default strategy and for
any other strategy that already has an index, and swaps the new index into
kb.retrieve, so /ask uses it without a server restart.
"""

import logging
import os
import re
import threading
import time
from pathlib import Path

from kb import embed, load, retrieve
from kb.chunk import DEFAULT_STRATEGY
from kb.load import kb_version, load_corpus

logger = logging.getLogger(__name__)

MAX_PAGE_CHARS = 200_000  # about 40 pages of text; bigger files are rejected
MAX_NAME_CHARS = 80
_UNSAFE = re.compile(r"[^a-z0-9._-]+")


class PageError(ValueError):
    """A page that can't be added or removed. The message is safe to show the user."""


def clean_name(name: str) -> str:
    """A safe file name for a page: lower case, letters, digits, '.', '_' and '-',
    ending in .md. Folders in the name are dropped, so it can't leave the corpus."""
    stem = re.split(r"[\\/]", name or "")[-1].strip().lower()
    for ext in (".md", ".markdown", ".txt"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = _UNSAFE.sub("-", stem).strip("-.")[:MAX_NAME_CHARS].strip("-.")
    if not stem:
        raise PageError("the page needs a name made of letters or digits")
    return stem + ".md"


def list_pages():
    """Every page with where it comes from: "base" (read-only) or "user"."""
    user = {page.name for page in load_corpus(load.USER_CORPUS_DIR)}
    return [
        {"name": page.name, "source": "user" if page.name in user else "base",
         "chars": len(page.text), "sha256": page.sha256}
        for page in load_corpus()
    ]


def add_page(name: str, text: str) -> dict:
    """Save a user page. Adding a name that's already a user page replaces it."""
    filename = clean_name(name)
    if not text or not text.strip():
        raise PageError("the page is empty")
    if len(text) > MAX_PAGE_CHARS:
        raise PageError(f"the page is over {MAX_PAGE_CHARS:,} characters")
    if (Path(load.CORPUS_DIR) / filename).exists():
        raise PageError(f"{filename} is a built-in page, pick another name")
    folder = Path(load.USER_CORPUS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    replaced = path.exists()
    tmp = folder / (filename + ".tmp")  # not *.md, so load_corpus never reads it half written
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return {"name": filename, "replaced": replaced}


def delete_page(name: str) -> str:
    """Remove a user page. Base pages can't be removed."""
    filename = clean_name(name)
    if filename != name:
        raise PageError(f"no page called {name!r}")
    path = Path(load.USER_CORPUS_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    path.unlink()
    return filename


# ------------------------------------------------------------------ rebuilding

_build_lock = threading.Lock()  # one rebuild at a time
_status_lock = threading.Lock()
_status = {"state": "ready", "pending": 0, "error": None, "built_at": None, "seconds": None}


def status():
    """Whether the index is being rebuilt, and whether it matches the pages on disk."""
    with _status_lock:
        current = dict(_status)
    current["state"] = "rebuilding" if current.pop("pending") else current["state"]
    current["kb_version"] = kb_version(load_corpus())
    try:
        meta, _ = retrieve.get_index(DEFAULT_STRATEGY)
        current["index_version"] = meta["kb_version"]
        current["chunks"] = len(meta["chunks"])
    except FileNotFoundError:
        current["index_version"] = None
        current["chunks"] = 0
    current["up_to_date"] = current["index_version"] == current["kb_version"]
    return current


def mark_pending():
    """Call when a rebuild is scheduled, so status() says "rebuilding" straight away."""
    with _status_lock:
        _status["pending"] += 1


def _strategies(index_dir):
    built = {p.parent.name for p in Path(index_dir).glob("*/index.json")}
    return [DEFAULT_STRATEGY] + sorted(built - {DEFAULT_STRATEGY})


def rebuild(index_dir=None):
    """Re-embed the corpus and swap the new index in. Runs in the background after
    every change; changes made during a rebuild get their own rebuild after it."""
    index_dir = embed.INDEX_DIR if index_dir is None else index_dir
    start = time.perf_counter()
    try:
        with _build_lock:
            for strategy in _strategies(index_dir):
                meta, vectors = embed.build_index(strategy, index_dir)
                retrieve._indexes[strategy] = meta, vectors  # /ask uses it from now on
        with _status_lock:
            _status.update(state="ready", error=None, built_at=time.time(),
                           seconds=round(time.perf_counter() - start, 1))
        logger.info("Knowledge-base index rebuilt in %.1fs", time.perf_counter() - start)
    except Exception as exc:
        with _status_lock:
            _status.update(state="error", error=type(exc).__name__)
        logger.exception("Knowledge-base rebuild failed")
    finally:
        with _status_lock:
            _status["pending"] = max(_status["pending"] - 1, 0)
