"""Load the corpus pages and hash them.

Run from the project folder:
    uv run python -m kb.load
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"


@dataclass(frozen=True)
class Page:
    name: str  # file name, e.g. tutorial-body.md
    text: str
    sha256: str


def load_corpus(folder=CORPUS_DIR):
    """Every .md file in the folder, sorted by name."""
    pages = []
    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        pages.append(Page(path.name, text, hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return pages


def kb_version(pages):
    """One hash for the whole corpus. It changes if any page changes."""
    combined = "\n".join(f"{page.name} {page.sha256}" for page in pages)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    pages = load_corpus()
    for page in pages:
        print(f"{page.sha256[:12]}  {page.name}  ({len(page.text)} chars)")
    print(f"\n{len(pages)} pages, KB version {kb_version(pages)[:12]}")
