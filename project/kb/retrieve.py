"""Find the chunks that best match a question.

Run from the project folder:
    uv run python -m kb.retrieve "how do I read a query parameter?"
    uv run python -m kb.retrieve --strategy semantic "how do I read a query parameter?"
"""

import argparse
import json
from pathlib import Path

import numpy as np

from kb.chunk import CHUNKERS
from kb.embed import INDEX_DIR, get_model


def load_index(strategy="token", index_dir=INDEX_DIR):
    folder = Path(index_dir) / strategy
    meta = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    vectors = np.load(folder / "index.npy")
    return meta, vectors


def retrieve(question, k=3, strategy="token", index=None):
    """The top k chunks, best first. Each has its id, page, text and score."""
    meta, vectors = index or load_index(strategy)
    query = get_model().encode([question], normalize_embeddings=True)[0]
    scores = vectors @ query  # cosine similarity, higher is closer
    best = np.argsort(scores)[::-1][:k]
    return [{**meta["chunks"][i], "score": float(scores[i])} for i in best]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the chunks closest to a question.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--strategy", choices=list(CHUNKERS), default="token")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    question = " ".join(args.question) or "How do I read a query parameter?"
    print(f"Q: {question}  ({args.strategy} chunks)\n")
    for hit in retrieve(question, k=args.k, strategy=args.strategy):
        print(f"{hit['score']:.3f}  {hit['id']}  ({hit['tokens']} tokens)")
        print("      " + hit["text"][:150].replace("\n", " ") + "...\n")
