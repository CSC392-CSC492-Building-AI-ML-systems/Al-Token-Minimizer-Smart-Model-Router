"""Put the code examples into the FastAPI pages.

The pages link to example files with lines like
{* ../../docs_src/body/tutorial001_py310.py hl[2] *}. This swaps each one for the
file's code. Run it once after adding pages.

Run from the project folder:
    uv run python -m kb.inline_code
"""

import re
import urllib.request
from pathlib import Path

from kb.load import CORPUS_DIR

# pinned so the examples don't change if FastAPI updates them
FASTAPI_COMMIT = "5cb6272ed7e3e07eaf6978e1e65b94aca76fa8e0"  # master on 2026-11-02
RAW_URL = f"https://raw.githubusercontent.com/fastapi/fastapi/{FASTAPI_COMMIT}/"

_INCLUDE = re.compile(r"\{\*\s*(\S+)[^}]*?\*\}")  # group 1 is the file path
_LANGUAGES = {".py": "python", ".js": "javascript", ".toml": "toml", ".json": "json", ".sh": "bash"}


def _fetch(path, cache):
    """Download a file from the repo. `path` looks like ../../docs_src/x.py."""
    repo_path = re.sub(r"^(\.\./)+", "", path)
    if repo_path not in cache:
        with urllib.request.urlopen(RAW_URL + repo_path, timeout=30) as response:
            cache[repo_path] = response.read().decode("utf-8")
    return cache[repo_path]


def inline_page(text, cache):
    def replace(match):
        path = match.group(1)
        language = _LANGUAGES.get(Path(path).suffix, "")
        return f"```{language}\n{_fetch(path, cache).rstrip()}\n```"

    return _INCLUDE.sub(replace, text)


if __name__ == "__main__":
    cache = {}
    for page in sorted(Path(CORPUS_DIR).glob("*.md")):
        text = page.read_text(encoding="utf-8")
        found = len(_INCLUDE.findall(text))
        if found:
            page.write_text(inline_page(text, cache), encoding="utf-8")
        print(f"{found:3d} examples inlined  {page.name}")
    print(f"\n{len(cache)} example files downloaded from fastapi@{FASTAPI_COMMIT[:7]}")
