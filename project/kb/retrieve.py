"""Find the chunks that best match a question.

Run from the project folder:
    uv run python -m kb.retrieve "how do I read a query parameter?"
    uv run python -m kb.retrieve --strategy token "how do I read a query parameter?"
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kb.chunk import CHUNKERS, DEFAULT_STRATEGY
from kb.embed import INDEX_DIR, get_model
from kb.load import kb_version, load_corpus

logger = logging.getLogger(__name__)

_indexes = {}  # strategy -> (meta, vectors), filled by get_index


def load_index(strategy=DEFAULT_STRATEGY, index_dir=INDEX_DIR):
    folder = Path(index_dir) / strategy
    if not (folder / "index.json").exists():
        raise FileNotFoundError(
            f"no {strategy} index in {index_dir}, build it with: uv run python -m kb.embed --strategy {strategy}"
        )
    meta = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    vectors = np.load(folder / "index.npy")
    return meta, vectors


def get_index(strategy=DEFAULT_STRATEGY):
    """The saved index, loaded once per process. Warns if the corpus changed since it was built."""
    if strategy not in _indexes:
        meta, vectors = load_index(strategy)
        if meta["kb_version"] != kb_version(load_corpus()):
            logger.warning(
                "The %s index is out of date, rebuild it with: uv run python -m kb.embed --strategy %s",
                strategy, strategy,
            )
        _indexes[strategy] = meta, vectors
    return _indexes[strategy]


def retrieve(question, k=3, strategy=DEFAULT_STRATEGY, index=None):
    """The top k chunks, best first. Each has its id, page, text and score."""
    meta, vectors = index or get_index(strategy)
    query = get_model().encode([question], normalize_embeddings=True)[0]
    scores = vectors @ query  # cosine similarity, higher is closer
    best = np.argsort(scores)[::-1][:k]
    return [{**meta["chunks"][i], "score": float(scores[i])} for i in best]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the chunks closest to a question.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--strategy", choices=list(CHUNKERS), default=DEFAULT_STRATEGY)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    question = " ".join(args.question) or "How do I read a query parameter?"
    print(f"Q: {question}  ({args.strategy} chunks)\n")
    for hit in retrieve(question, k=args.k, strategy=args.strategy):
        print(f"{hit['score']:.3f}  {hit['id']}  ({hit['tokens']} tokens)")
        print("      " + hit["text"][:150].replace("\n", " ") + "...\n")
