"""Split pages into chunks. There are three strategies: token, sliding and semantic.

Run from the project folder to compare them:
    uv run python -m kb.chunk
"""

from dataclasses import dataclass
from typing import Protocol

from kb.load import load_corpus


@dataclass(frozen=True)
class Chunk:
    id: str  # page name plus position, e.g. tutorial-body.md#2
    page: str
    page_sha256: str
    text: str
    tokens: int


class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...


def count_tokens(text: str) -> int:
    # counts words for now, close enough until we need exact Gemini counts
    return len(text.split())


def _join(pieces):
    return "\n\n".join(pieces)


def blocks(text: str) -> list[tuple[str, bool]]:
    """Split text into (block, is_code) pairs. Code blocks stay whole even with blank lines."""
    result, current, in_code = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if not in_code and current:  # opening fence ends the paragraph before it
                result.append(("\n".join(current), False))
                current = []
            current.append(line)
            if in_code:  # closing fence
                result.append(("\n".join(current), True))
                current = []
            in_code = not in_code
        elif in_code or line.strip():
            current.append(line)
        elif current:  # blank line ends a paragraph
            result.append(("\n".join(current), False))
            current = []
    if current:
        result.append(("\n".join(current), in_code))
    return result


def units(text: str, max_tokens: int) -> list[str]:
    """The pieces all three strategies build chunks from.

    Code blocks are never split, even if they're over max_tokens. Long paragraphs
    are split on words. Headings get attached to the block after them.
    """
    result, heading = [], None
    for block, is_code in blocks(text):
        if not is_code and block.startswith("#") and "\n" not in block:
            heading = block if heading is None else _join([heading, block])
            continue
        if is_code or count_tokens(block) <= max_tokens:
            pieces = [block]
        else:
            words = block.split()
            pieces = [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]
        if heading is not None:
            pieces[0] = _join([heading, pieces[0]])
            heading = None
        result += pieces
    if heading is not None:  # heading at the very end of the page
        result.append(heading)
    return result


class TokenChunker:
    """Fills each chunk up to max_tokens, with no overlap."""

    def __init__(self, max_tokens: int = 200):
        self.max_tokens = max_tokens

    def split(self, text: str) -> list[str]:
        chunks, current = [], []
        for piece in units(text, self.max_tokens):
            if current and count_tokens(_join(current + [piece])) > self.max_tokens:
                chunks.append(_join(current))
                current = []
            current.append(piece)
        if current:
            chunks.append(_join(current))
        return chunks


class SlidingWindowChunker:
    """Like TokenChunker, but each chunk repeats the end of the one before it."""

    def __init__(self, max_tokens: int = 200, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split(self, text: str) -> list[str]:
        chunks, current = [], []
        for piece in units(text, self.max_tokens):
            if current and count_tokens(_join(current + [piece])) > self.max_tokens:
                chunks.append(_join(current))
                # carry over the last few pieces, up to overlap_tokens
                tail = []
                for unit in reversed(current):
                    if count_tokens(_join([unit] + tail)) > self.overlap_tokens:
                        break
                    tail.insert(0, unit)
                current = tail if count_tokens(_join(tail + [piece])) <= self.max_tokens else []
            current.append(piece)
        if current:
            chunks.append(_join(current))
        return chunks


class SemanticChunker:
    """Starts a new chunk where the topic changes.

    Neighbouring pieces with low similarity (bottom `percentile` on the page) mark a
    topic change. Chunks are at least min_tokens and at most max_tokens.
    """

    def __init__(self, max_tokens: int = 200, percentile: float = 25, min_tokens: int = 50):
        self.max_tokens = max_tokens
        self.percentile = percentile
        self.min_tokens = min_tokens

    def split(self, text: str) -> list[str]:
        import numpy as np

        from kb.embed import get_model  # imported here because kb.embed imports this file

        pieces = units(text, self.max_tokens)
        if len(pieces) < 2:
            return pieces
        vectors = get_model().encode(pieces, normalize_embeddings=True)
        similarity = (vectors[:-1] * vectors[1:]).sum(axis=1)  # piece i vs piece i + 1
        cutoff = np.percentile(similarity, self.percentile)

        chunks, current = [], [pieces[0]]
        for i, piece in enumerate(pieces[1:], start=1):
            topic_change = similarity[i - 1] < cutoff and count_tokens(_join(current)) >= self.min_tokens
            too_big = count_tokens(_join(current + [piece])) > self.max_tokens
            if topic_change or too_big:
                chunks.append(_join(current))
                current = []
            current.append(piece)
        chunks.append(_join(current))
        return chunks


CHUNKERS = {"token": TokenChunker, "sliding": SlidingWindowChunker, "semantic": SemanticChunker}


def chunk_pages(pages, chunker: Chunker) -> list[Chunk]:
    chunks = []
    for page in pages:
        for number, text in enumerate(chunker.split(page.text), start=1):
            chunks.append(
                Chunk(
                    id=f"{page.name}#{number}",
                    page=page.name,
                    page_sha256=page.sha256,
                    text=text,
                    tokens=count_tokens(text),
                )
            )
    return chunks


if __name__ == "__main__":
    pages = load_corpus()
    for name, chunker in CHUNKERS.items():
        chunks = chunk_pages(pages, chunker())
        sizes = [chunk.tokens for chunk in chunks]
        print(
            f"{name:<9} {len(chunks):3d} chunks, tokens per chunk: "
            f"min {min(sizes)}, max {max(sizes)}, average {sum(sizes) // len(sizes)}, total {sum(sizes)}"
        )
