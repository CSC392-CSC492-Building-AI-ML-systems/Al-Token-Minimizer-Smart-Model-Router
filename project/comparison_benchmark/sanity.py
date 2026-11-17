"""Make one live direct-versus-proxy RAG request using the built-in KB.

Run from the project folder:
    uv run python -m comparison_benchmark.sanity

This sends one request directly to Gemini and one through POST /ask. Both use
the same retrieved passages, model, and system instruction. It needs a valid
GEMINI_API_KEY and uses the Gemini API free tier when available.
"""

import os
import logging
import tempfile
from pathlib import Path

from comparison_benchmark import setup_gemini, setup_kb

CASE = {
    "question": "Where do I find the interactive API documentation?",
    "expected_evidence": "/docs",
    "system": "Answer in one short sentence using only the supplied context.",
}


def run_sanity_case():
    direct_client, model = setup_gemini()
    kb_metadata = setup_kb()
    strategy = kb_metadata["strategy"]

    os.environ["SEMANTIC_CACHE_ENABLED"] = "false"
    with tempfile.TemporaryDirectory(prefix="rag-sanity-") as temp_dir:
        import log_setup

        previous_log_dir = log_setup.LOG_DIR
        log_setup.LOG_DIR = Path(temp_dir) / "logs"
        previous_db_path = os.environ.get("TELEMETRY_DB_PATH")
        os.environ["TELEMETRY_DB_PATH"] = str(Path(temp_dir) / "telemetry.db")

        try:
            result = _run_calls(direct_client, model, strategy, kb_metadata)
        finally:
            root = logging.getLogger()
            temp_log = (log_setup.LOG_DIR / log_setup.LOG_FILE_NAME).resolve()
            for handler in list(root.handlers):
                if Path(getattr(handler, "baseFilename", "")).resolve() == temp_log:
                    root.removeHandler(handler)
                    handler.close()
            log_setup.LOG_DIR = previous_log_dir
            if previous_db_path is None:
                os.environ.pop("TELEMETRY_DB_PATH", None)
            else:
                os.environ["TELEMETRY_DB_PATH"] = previous_db_path
        return result


def _run_calls(direct_client, model, strategy, kb_metadata):
    import main
    from fastapi.testclient import TestClient

    from compressor.pipeline import Chunk, Prompt
    from gemini_parse import reply_text
    from kb.retrieve import retrieve

    if main.MODEL != model:
        raise RuntimeError(f"Proxy model is {main.MODEL}, expected {model}")

    passages = retrieve(CASE["question"], k=main.KB_TOP_K, strategy=strategy)
    if not passages:
        raise RuntimeError("The KB returned no passages for the sanity question")
    passage_ids = [passage["id"] for passage in passages]
    context = tuple(Chunk(passage["text"], passage["id"]) for passage in passages)
    prompt = Prompt(CASE["question"], CASE["system"], context)

    direct_interaction = direct_client.interactions.create(
        model=model,
        input=prompt.render(),
        system_instruction=CASE["system"],
    )
    direct_text = reply_text(direct_interaction)

    proxy_passages = []
    original_retrieve = main.retrieve

    def capture_proxy_retrieval(question, k, strategy):
        hits = original_retrieve(question, k=k, strategy=strategy)
        proxy_passages.extend(hits)
        return hits

    main.retrieve = capture_proxy_retrieval
    try:
        with TestClient(main.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/ask",
                json={
                    "prompt": CASE["question"],
                    "system": CASE["system"],
                    "kb_strategy": strategy,
                    "stages": [],
                    "cache": "off",
                },
            )
    finally:
        main.retrieve = original_retrieve

    if response.status_code != 200:
        raise RuntimeError(f"Proxy request failed with HTTP {response.status_code}: {response.json()}")

    proxy_text = response.json()["text"]
    proxy_ids = [passage["id"] for passage in proxy_passages]
    if proxy_ids != passage_ids:
        raise AssertionError(f"Direct and proxy retrieval differed: {passage_ids} != {proxy_ids}")

    expected = CASE["expected_evidence"].casefold()
    direct_passed = expected in direct_text.casefold()
    proxy_passed = expected in proxy_text.casefold()

    print(f"Model: {model}")
    print(f"KB version: {kb_metadata['kb_version']}")
    print(f"Retrieved passages: {', '.join(passage_ids)}")
    print(f"Direct answer: {direct_text}")
    print(f"Proxy answer: {proxy_text}")
    print(f"Direct contains expected evidence: {direct_passed}")
    print(f"Proxy contains expected evidence: {proxy_passed}")
    if not direct_passed or not proxy_passed:
        raise AssertionError("At least one answer did not include the expected evidence")
    return {
        "model": model,
        "kb_version": kb_metadata["kb_version"],
        "passage_ids": passage_ids,
        "direct_text": direct_text,
        "proxy_text": proxy_text,
        "cache_status": response.json()["cache_status"],
    }


if __name__ == "__main__":
    run_sanity_case()
