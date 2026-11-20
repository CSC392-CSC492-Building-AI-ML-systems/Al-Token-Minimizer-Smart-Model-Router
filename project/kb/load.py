"""Load the corpus pages and hash them.

Run from the project folder:
    uv run python -m kb.load
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"  # the base pages, in git
USER_CORPUS_DIR = Path(__file__).parent / "user_corpus"  # pages added through /kb, gitignored


@dataclass(frozen=True)
class Page:
    name: str  # file name, e.g. tutorial-body.md
    text: str
    sha256: str


def load_corpus(folders=None):
    """Every .md file in the folder(s), sorted by name. A missing folder has no pages.
    By default the base pages and the pages users added."""
    if folders is None:
        folders = (CORPUS_DIR, USER_CORPUS_DIR)
    elif isinstance(folders, (str, Path)):
        folders = (folders,)
    pages = []
    for folder in folders:
        for path in Path(folder).glob("*.md"):
            text = path.read_text(encoding="utf-8")
            pages.append(Page(path.name, text, hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return sorted(pages, key=lambda page: page.name)


def kb_version(pages):
    """One hash for the whole corpus. It changes if any page changes."""
    combined = "\n".join(f"{page.name} {page.sha256}" for page in pages)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    pages = load_corpus()
    for page in pages:
        print(f"{page.sha256[:12]}  {page.name}  ({len(page.text)} chars)")
    print(f"\n{len(pages)} pages, KB version {kb_version(pages)[:12]}")
