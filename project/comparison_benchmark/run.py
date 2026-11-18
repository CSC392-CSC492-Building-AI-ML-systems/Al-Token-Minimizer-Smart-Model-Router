"""Run live direct-versus-proxy RAG and answer-cache checks.

Run from the project folder:
    uv run python -m comparison_benchmark

Each case calls Gemini directly for its original question and paraphrase,
then calls the proxy to generate, replay the exact question, and try the
paraphrase. Both proxy prompts use the original question's retrieved KB
passages so the semantic-cache check varies wording, not context. Compression
token savings come from the proxy's CompressorService and model-specific
GeminiTokenCounter telemetry.

The ``answer_checks`` verify that each direct, generated, and cached answer
contains the case's expected evidence as a case-insensitive substring. They
check for required evidence, not overall answer quality or equality between
answers; a failed check raises an assertion and prevents the report from being
written. Cache-hit types are checked separately.

Requires a valid GEMINI_API_KEY and Redis Stack running for semantic-cache
checks. Gemini calls use the free tier when available and consume API quota.

Note (remove this later): Right now each case would make three Gemini calls which would get inefficient as 
more cases are added, to fix this we can run Gemini once for every case and save the output
as a dictionary and utilize that as the baseline, which avoids making Gemini calls for every test case.
I'll implement that later, once I know that the current setup is decent.
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from comparison_benchmark.cases import CASES
from comparison_benchmark.setup import FREE_GEMINI_MODEL, setup_gemini, setup_kb

RESULTS_DIR = Path(__file__).parent / "results"


def _direct_answer(client, model, question, system, context):
    from compressor.pipeline import Prompt
    from gemini_parse import reply_text

    prompt = Prompt(question, system, context)
    interaction = client.interactions.create(
        model=model,
        input=prompt.render(),
        system_instruction=system,
    )
    return reply_text(interaction)


def _proxy_request(client, case, question, strategy, cache_mode):
    response = client.post(
        "/ask",
        json={
            "prompt": question,
            "system": case["system"],
            "kb_strategy": strategy,
            "stages": case.get("compression_stages", []),
            "cache": cache_mode,
        },
    )
    if response.status_code != 200:
        raise RuntimeError(f"Proxy request failed with HTTP {response.status_code}: {response.json()}")
    return response.json()


def _compression_savings(uncompressed_tokens, compressed_tokens):
    """Return input-token savings; negative values mean compression grew input."""
    saved = uncompressed_tokens - compressed_tokens
    percent = saved / uncompressed_tokens * 100 if uncompressed_tokens else None
    return {
        "input_tokens_uncompressed": uncompressed_tokens,
        "input_tokens_compressed": compressed_tokens,
        "compression_input_tokens_saved": saved,
        "compression_input_savings_percent": percent,
    }


def _cache_savings(reference_telemetry, exact_hit, semantic_hit):
    """Estimate the Gemini tokens each successful cache replay could have used."""
    reference_total = reference_telemetry["total_tokens"]

    # Exact requests have the same cache key and therefore the same model input
    # settings as the fresh generation. A semantic paraphrase can have different
    # input/output usage, so this uses the fresh generation as an estimate and
    # assumes comparable context and response length. These are gross tokens
    # potentially avoided on the replay requests, based on Gemini's total_tokens
    # for the fresh generation. This excludes local KB retrieval and direct
    # baseline calls, and is not net savings for the run because the
    # cache-filling generation was still performed.
    exact_saved = reference_total if exact_hit and reference_total is not None else (
        None if exact_hit else 0
    )
    semantic_saved = reference_total if semantic_hit and reference_total is not None else (
        None if semantic_hit else 0
    )
    total_saved = (
        exact_saved + semantic_saved
        if exact_saved is not None and semantic_saved is not None
        else None
    )

    return {
        "cache_savings_reference_usage": {
            "input_tokens": reference_telemetry["input_tokens_reported"],
            "output_tokens": reference_telemetry["output_tokens"],
            "thought_tokens": reference_telemetry["thought_tokens"],
            "total_tokens": reference_total,
        },
        "exact_cache_potential_tokens_saved": exact_saved,
        "semantic_cache_potential_tokens_saved": semantic_saved,
        "cache_potential_tokens_saved_total": total_saved,
    }


def _total_tokens_saved(compression_tokens_saved, cache_tokens_saved):
    # This combines compression input savings with gross cache-replay savings.
    # Keep the total unknown if cache usage is unknown rather than reporting a
    # misleading partial sum.
    return (
        compression_tokens_saved + cache_tokens_saved
        if cache_tokens_saved is not None
        else None
    )


def _contains_expected_evidence(answer, evidence):
    expected = (evidence,) if isinstance(evidence, str) else evidence
    answer = answer.casefold()
    return all(item.casefold() in answer for item in expected)


def run_case(case, direct_client, model, strategy, metadata, proxy_client, main):
    """Run one case against direct Gemini and the proxy's exact/semantic caches."""
    from compressor.pipeline import Chunk
    from kb.retrieve import retrieve

    passages = retrieve(case["question"], k=main.KB_TOP_K, strategy=strategy)
    if not passages:
        raise RuntimeError(f"KB returned no passages for case {case['id']!r}")

    passage_ids = [passage["id"] for passage in passages]
    context = tuple(Chunk(passage["text"], passage["id"]) for passage in passages)
    retrieved_text = "\n".join(chunk.text for chunk in context)
    if not _contains_expected_evidence(retrieved_text, case["expected_evidence"]):
        raise RuntimeError(
            f"Retrieved passages for case {case['id']!r} lack expected evidence "
            f"{case['expected_evidence']!r}; passages={passage_ids}"
        )

    baseline = _direct_answer(
        direct_client, model, case["question"], case["system"], context
    )
    paraphrase_baseline = _direct_answer(
        direct_client, model, case["paraphrase"], case["system"], context
    )

    original_retrieve = main.retrieve
    retrieved_for_proxy = []

    def fixed_context_retrieve(question, k, strategy):
        hits = passages[:k]
        retrieved_for_proxy.extend(hit["id"] for hit in hits)
        return hits

    main.retrieve = fixed_context_retrieve
    try:
        generated = _proxy_request(
            proxy_client, case, case["question"], strategy, "refresh"
        )
        exact = _proxy_request(
            proxy_client, case, case["question"], strategy, "use"
        )
        semantic = _proxy_request(
            proxy_client, case, case["paraphrase"], strategy, "use"
        )
    finally:
        main.retrieve = original_retrieve

    telemetry_rows = main.db.fetch_requests(request_id=generated["request_id"])
    if len(telemetry_rows) != 1:
        raise RuntimeError(
            f"Expected telemetry for generated request {generated['request_id']!r}, "
            f"found {len(telemetry_rows)} rows"
        )
    generated_telemetry = telemetry_rows[0]
    raw_tokens = generated_telemetry["input_tokens_raw"]
    compressed_tokens = generated_telemetry["input_tokens_compressed"]
    if raw_tokens is None or compressed_tokens is None:
        raise RuntimeError("Proxy telemetry is missing CompressorService input-token counts")

    # These counts come from CompressorService using GeminiTokenCounter for the
    # selected model. They include the rendered question+KB context and system
    # prompt. Their difference estimates input-only compression savings under
    # the assumption that all other request settings are unchanged; it does not
    # count output/thought tokens or any savings from cache hits.
    compression_metrics = _compression_savings(raw_tokens, compressed_tokens)

    answer_checks = {
        "direct": _contains_expected_evidence(baseline, case["expected_evidence"]),
        "direct_paraphrase": _contains_expected_evidence(
            paraphrase_baseline, case["expected_evidence"]
        ),
        "proxy_generated": _contains_expected_evidence(
            generated["text"], case["expected_evidence"]
        ),
        "exact_hit": _contains_expected_evidence(exact["text"], case["expected_evidence"]),
        "semantic_hit": _contains_expected_evidence(
            semantic["text"], case["expected_evidence"]
        ),
    }

    if retrieved_for_proxy != passage_ids * 3:
        raise AssertionError(
            f"Proxy context differed from baseline context: {retrieved_for_proxy} != {passage_ids * 3}"
        )
    if exact["cache_hit"] != "exact":
        raise AssertionError(f"Expected exact cache hit, got {exact['cache_hit']!r}")
    if semantic["cache_hit"] != "semantic":
        raise AssertionError(f"Expected semantic cache hit, got {semantic['cache_hit']!r}")
    if generated["cache_status"] != "refresh":
        raise AssertionError(f"Expected fresh proxy generation, got {generated['cache_status']!r}")
    if not all(answer_checks.values()):
        answer_samples = {
            "direct": baseline[:300],
            "direct_paraphrase": paraphrase_baseline[:300],
            "proxy_generated": generated["text"][:300],
            "exact_hit": exact["text"][:300],
            "semantic_hit": semantic["text"][:300],
        }
        raise AssertionError(
            f"Answer evidence check failed: {answer_checks}; "
            f"expected={case['expected_evidence']!r}; "
            f"passages={passage_ids}; answers={answer_samples!r}"
        )

    cache_savings = _cache_savings(
        generated_telemetry,
        exact_hit=exact["cache_hit"] == "exact",
        semantic_hit=semantic["cache_hit"] == "semantic",
    )
    total_tokens_saved = _total_tokens_saved(
        compression_metrics["compression_input_tokens_saved"],
        cache_savings["cache_potential_tokens_saved_total"],
    )

    result = {
        "case_id": case["id"],
        "model": model,
        "kb_version": metadata["kb_version"],
        "passage_ids": passage_ids,
        "compression_stages": case.get("compression_stages", []),
        **compression_metrics,
        **cache_savings,
        "total_tokens_saved": total_tokens_saved,
        "baseline_output": baseline,
        "paraphrased_baseline_output": paraphrase_baseline,
        "proxy_output": generated["text"],
        "exact_cache_hits": exact["cache_hit"],
        "semantic_cache_hits": semantic["cache_hit"],
        "answer_checks": answer_checks,
    }
    print(f"\nCase: {case['id']}")
    print(f"Passages: {', '.join(passage_ids)}")
    print(f"Direct answer: {baseline}")
    print(f"Direct paraphrase answer: {paraphrase_baseline}")
    print(f"Proxy answer: {generated['text']} (cache={generated['cache_status']})")
    print(
        f"Compression input tokens: {raw_tokens} -> {compressed_tokens}; "
        f"saved {compression_metrics['compression_input_tokens_saved']} "
        f"({compression_metrics['compression_input_savings_percent']:.1f}%)"
        if compression_metrics["compression_input_savings_percent"] is not None
        else f"Compression input tokens: {raw_tokens} -> {compressed_tokens}; savings percentage unavailable"
    )
    print(f"Compression stages: {case.get('compression_stages', [])}")
    print(
        "Potential cache tokens saved: "
        f"exact={cache_savings['exact_cache_potential_tokens_saved']}, "
        f"semantic={cache_savings['semantic_cache_potential_tokens_saved']}, "
        f"total={cache_savings['cache_potential_tokens_saved_total']}"
    )
    print(f"Total tokens saved (compression + potential cache): {total_tokens_saved}")
    print(f"Exact replay: {exact['cache_hit']} ({exact['cache_status']})")
    print(f"Paraphrase replay: {semantic['cache_hit']} ({semantic['cache_status']})")
    print(f"Evidence checks: {answer_checks}")
    return result


def run():
    """Prepare the KB and clients, then run every case in CASES."""
    from dotenv import load_dotenv

    load_dotenv()
    environment_settings = (
        "CACHE_ENABLED",
        "EXACT_CACHE_ENABLED",
        "SEMANTIC_CACHE_ENABLED",
        "SEMANTIC_CACHE_DISTANCE_THRESHOLD",
    )
    previous_environment = {name: os.environ.get(name) for name in environment_settings}
    for name in environment_settings[:3]:
        os.environ[name] = "true"
    os.environ["SEMANTIC_CACHE_DISTANCE_THRESHOLD"] = "0.1"

    direct_client, model = setup_gemini()
    if model != FREE_GEMINI_MODEL:
        raise RuntimeError(f"Unexpected benchmark model: {model}")
    metadata = setup_kb()
    strategy = metadata["strategy"]

    try:
        results = _run_cases(direct_client, model, strategy, metadata)
        _save_report(results, model, strategy, metadata)
        return results
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _save_report(results, model, strategy, metadata):
    """Write chart-friendly JSON to a unique, persistent folder for this run."""
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%fZ")
    output_dir = RESULTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "report_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "kb": {
            "version": metadata["kb_version"],
            "strategy": strategy,
        },
        "cases": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"JSON report: {report_path}")
    return report_path


def _run_cases(direct_client, model, strategy, metadata):
    """Initialize an isolated proxy runtime and execute all configured cases."""
    with tempfile.TemporaryDirectory(prefix="rag-comparison-") as temp_dir:
        import log_setup

        previous_log_dir = log_setup.LOG_DIR
        previous_db_path = os.environ.get("TELEMETRY_DB_PATH")
        log_setup.LOG_DIR = Path(temp_dir) / "logs"
        os.environ["TELEMETRY_DB_PATH"] = str(Path(temp_dir) / "telemetry.db")
        try:
            import main
            from fastapi.testclient import TestClient
            from response_cache import ResponseCache

            if main.MODEL != model:
                raise RuntimeError(f"Proxy model is {main.MODEL}, expected {model}")
            if not main.SEMANTIC_CACHE_ENABLED:
                raise RuntimeError("Semantic cache is disabled; enable it before importing the proxy")

            # Fail before paying for model calls if Redis is unavailable.
            from cache import semantic as semantic_cache
            from redis.exceptions import ConnectionError as RedisConnectionError

            try:
                semantic_cache.warm_up()
            except RedisConnectionError as exc:
                raise RuntimeError(
                    "Semantic-cache check needs Redis Stack; start it from project with `docker compose up -d`"
                ) from exc

            previous_response_cache = main.response_cache
            main.response_cache = ResponseCache(
                max_entries=previous_response_cache.max_entries,
                ttl_seconds=previous_response_cache.ttl_seconds,
                max_entry_bytes=previous_response_cache.max_entry_bytes,
                enabled=True,
            )
            try:
                with TestClient(main.app, raise_server_exceptions=False) as proxy_client:
                    run_id = uuid.uuid4().hex
                    results = []
                    for case in CASES:
                        scoped_case = {
                            **case,
                            "system": (
                                f"{case['system']}\n"
                                f"Ignore this benchmark-run identifier: {run_id}."
                            ),
                        }
                        results.append(
                            run_case(
                                scoped_case,
                                direct_client,
                                model,
                                strategy,
                                metadata,
                                proxy_client,
                                main,
                            )
                        )
                    return results
            finally:
                main.response_cache.clear()
                main.response_cache = previous_response_cache
        finally:
            root = logging.getLogger()
            temp_log = (log_setup.LOG_DIR / log_setup.LOG_FILE_NAME).resolve()
            for handler in list(root.handlers):
                base_filename = getattr(handler, "baseFilename", None)
                if base_filename and Path(base_filename).resolve() == temp_log:
                    root.removeHandler(handler)
                    handler.close()
            log_setup.LOG_DIR = previous_log_dir
            if previous_db_path is None:
                os.environ.pop("TELEMETRY_DB_PATH", None)
            else:
                os.environ["TELEMETRY_DB_PATH"] = previous_db_path


if __name__ == "__main__":
    run()
