"""Set up the comparison benchmark's KB index and Gemini clients."""


FREE_GEMINI_MODEL = "gemini-3.5-flash-lite"


def setup_kb(strategy=None):
    """Build the KB index if it is missing or out of date, then return its metadata.

    Currently uses the Markdown files found in kb/corpus as the source documents.
    Call once before benchmark retrieval.
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
    import, so both direct and proxy calls use the same model.
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
