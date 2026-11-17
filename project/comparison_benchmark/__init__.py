"""Setup for a single-turn RAG baseline-versus-proxy benchmark.

The benchmark initially uses the repository's fixed FastAPI tutorial corpus
and a curated set of standalone questions with expected answers.

For each question:

Retrieve or select the relevant passages once, then use the identical
question, context, system instructions, model, and generation settings for the
direct baseline and proxy paths. This keeps retrieval differences from
confounding the comparison.

The comparison has four paths:
    * Direct model call, bypassing the proxy and its caches.
    * Proxy call with caching and compression disabled.
    * Proxy call with compression enabled and caching disabled.
    * Proxy calls with caching enabled: first request, exact repeat, and a
      paraphrase paired with the same retrieved context.

Include changed-context cases as negative controls: they should not reuse an
answer scoped to different context. Compare answer quality for direct,
uncached-proxy, exact-cache, and semantic-cache results.

Report:
    * Model and request settings, KB corpus version, and passage IDs.
    * Compression stages and cache settings.
    * Answer-quality results and cache outcomes.
    * Provider-reported input, output, and total tokens.
    * Direct and proxy latency.
    * Compression token savings separately from cache savings, to avoid
      double-counting.
    * Estimated tokens and latency avoided by a cache hit, based on an
      equivalent uncached call. Label these as estimates because a hit makes
      no model call.
    * Retrieval separately if it is later measured; retrieval still runs
      before the proxy's cache lookup.

Live provider calls should be explicitly run rather than part of the ordinary
unit-test suite. Use the same model on both paths when attributing differences
to the proxy, and record the corpus, model, and benchmark configuration with
each result.

The benchmark uses Gemini 3.5 Flash-Lite for both paths. ``setup_gemini()``
loads the API key and sets ``GEMINI_MODEL`` before the proxy is imported, then
returns a separate client for direct baseline calls.
"""

FREE_GEMINI_MODEL = "gemini-3.5-flash-lite"


def setup_kb(strategy=None):
    """Build the KB index if it is missing or out of date, then return its metadata.

    Currently uses the Markdown files found in kb/corpus as the source documents.

    Call this explicitly before benchmark retrieval. Importing this package
    does not load the embedding model or build an index.
    """
    from kb.chunk import CHUNKERS, DEFAULT_STRATEGY
    from kb.embed import build_index
    from kb.load import kb_version, load_corpus
    from kb.retrieve import load_index

    strategy = DEFAULT_STRATEGY if strategy is None else strategy
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown KB strategy {strategy!r}; expected one of {sorted(CHUNKERS)}")

    version = kb_version(load_corpus())
    try:
        metadata, _ = load_index(strategy)
    except FileNotFoundError:
        metadata = None

    if metadata is not None and metadata.get("kb_version") == version:
        return metadata

    metadata, _ = build_index(strategy)
    return metadata


def setup_gemini():
    """Prepare a direct Gemini client and configure the proxy model for this process.

    Call before importing ``main``. The app reads ``GEMINI_MODEL`` during
    import, so both direct and proxy calls use the same model without changing
    the app's normal default or mutating its model at runtime.
    """
    import os
    import sys

    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set; add it to project/.env")

    loaded_main = sys.modules.get("main")
    if loaded_main is not None and (
        loaded_main.MODEL != FREE_GEMINI_MODEL
        or loaded_main.model_package.tokenizer.model != FREE_GEMINI_MODEL
    ):
        raise RuntimeError(
            "Call setup_gemini() before importing main so both benchmark paths use one model"
        )

    os.environ["GEMINI_MODEL"] = FREE_GEMINI_MODEL
    return genai.Client(api_key=api_key), FREE_GEMINI_MODEL

