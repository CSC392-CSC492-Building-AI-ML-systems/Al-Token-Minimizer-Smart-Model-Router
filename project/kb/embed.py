"""Embed the chunks and save an index for each chunking strategy.

Run from the project folder:
    uv run python -m kb.embed                      # token strategy
    uv run python -m kb.embed --strategy semantic  # or sliding, or all
"""

import argparse
import json
from pathlib import Path

import numpy as np

from kb.chunk import CHUNKERS, chunk_pages
from kb.load import kb_version, load_corpus

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # same model as the semantic cache
INDEX_DIR = Path(__file__).parent / "index"

_model = None


def get_model():
    """Load the embedding model once. The first load takes a few seconds."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_index(strategy="token", index_dir=INDEX_DIR):
    """Save vectors to index.npy and chunk details to index.json, in index/<strategy>/."""
    pages = load_corpus()
    chunker = CHUNKERS[strategy]()
    chunks = chunk_pages(pages, chunker)
    # normalized, so a dot product is the cosine similarity
    vectors = get_model().encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)

    out = Path(index_dir) / strategy
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "index.npy", np.asarray(vectors, dtype=np.float32))
    meta = {
        "kb_version": kb_version(pages),
        "model": EMBEDDING_MODEL,
        "strategy": strategy,
        "settings": vars(chunker),
        "chunks": [
            {"id": c.id, "page": c.page, "page_sha256": c.page_sha256, "tokens": c.tokens, "text": c.text}
            for c in chunks
        ],
    }
    (out / "index.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta, vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the knowledge-base index.")
    parser.add_argument("--strategy", choices=[*CHUNKERS, "all"], default="token")
    args = parser.parse_args()
    for strategy in CHUNKERS if args.strategy == "all" else [args.strategy]:
        meta, vectors = build_index(strategy)
        print(f"{strategy:<9} {len(meta['chunks'])} chunks embedded, KB version {meta['kb_version'][:12]}")
    print(f"saved to {INDEX_DIR}/")
