import pytest

from compressor import pipeline
from compressor.pipeline import Chunk, Prompt


def test_stage_files_are_registered():
    assert "passthrough" in pipeline.STAGES


def test_a_stage_name_cant_be_used_twice():
    with pytest.raises(ValueError):
        pipeline.stage("passthrough")(lambda p: p)


@pytest.mark.parametrize("name", ["none", "a+b"])
def test_a_stage_name_cant_look_like_a_label(name):
    with pytest.raises(ValueError):
        pipeline.stage(name)(lambda p: p)


def test_compress_runs_the_stages_in_order(monkeypatch):
    monkeypatch.setitem(pipeline.STAGES, "a", lambda p: Prompt(p.text + "a"))
    monkeypatch.setitem(pipeline.STAGES, "b", lambda p: Prompt(p.text + "b"))

    assert pipeline.compress(Prompt("x"), ["a", "b"]).text == "xab"


def test_no_stages_returns_the_prompt():
    prompt = Prompt("x")

    assert pipeline.compress(prompt) is prompt


def test_label_and_parse_label_round_trip():
    assert pipeline.label([]) == "none"
    assert pipeline.label(["a", "b"]) == "a+b"
    assert pipeline.parse_label("none") == []
    assert pipeline.parse_label("a+b") == ["a", "b"]


def test_render_without_context_is_just_the_prompt():
    assert Prompt("hi").render() == "hi"


def test_render_puts_the_chunks_before_the_question():
    prompt = Prompt("what?", context=(Chunk("one", "a.md"), Chunk("two")))

    assert prompt.render() == "Context:\n[a.md]\none\n\ntwo\n\nQuestion:\nwhat?"


def test_passthrough_returns_the_prompt_unchanged():
    prompt = Prompt("  what   is\n X? ", "sys", (Chunk("c \n d", "f.py"),))

    assert pipeline.compress(prompt, ["passthrough"]) == prompt


# Preset, conciseness and counting contracts.
from dataclasses import replace
from types import SimpleNamespace

from compressor.compressor_service import CompressorService


def test_none_is_an_exact_noop():
    prompt = pipeline.Prompt("  Keep  whitespace\n", "Return detailed JSON.")
    assert pipeline.compress(prompt, "none") is prompt


def test_concise_preserves_requirements_and_is_idempotent():
    prompt = pipeline.Prompt("Explain every step in detail.", "Return only JSON with a steps array.")
    compressed = pipeline.compress(prompt, "light")
    assert compressed.text == prompt.text
    assert compressed.system.endswith("\n\n" + prompt.system)
    assert "Follow any request for detail" in compressed.system
    assert pipeline.compress(compressed, "light") == compressed
    assert prompt.system == "Return only JSON with a steps array."


def test_full_runs_code_then_concise(monkeypatch):
    seen = []

    def stage(name):
        def run(prompt):
            seen.append((name, prompt.text))
            return replace(prompt, text=prompt.text + name)
        return run

    monkeypatch.setitem(pipeline.STAGES, "code", stage("code"))
    monkeypatch.setitem(pipeline.STAGES, "concise", stage("concise"))
    assert pipeline.compress(pipeline.Prompt(""), "full").text == "codeconcise"
    assert seen == [("code", ""), ("concise", "code")]


def test_fingerprint_tracks_policy_and_version(monkeypatch):
    initial = pipeline.fingerprint("full")
    monkeypatch.setattr(pipeline, "CONCISE_INSTRUCTION", "Changed policy")
    assert pipeline.fingerprint("full") != initial
    code = pipeline.fingerprint("code")
    monkeypatch.setattr(pipeline, "PIPELINE_VERSION", "next")
    assert pipeline.fingerprint("code") != code


def test_counts_system_cost_and_reuses_unchanged_text():
    calls = []

    def count(text):
        calls.append(text)
        return len(text)

    model = SimpleNamespace(tokenizer=SimpleNamespace(count=count))
    original = pipeline.Prompt("Question", "Custom instruction")
    service = CompressorService(original, model, pipeline="light")
    counts = service.get_token_information()
    assert counts["uncompressed_input_tokens"] == len(original.text) + len(original.system)
    assert counts["compressed_input_tokens"] == len(original.text) + len(service.compressed_prompt.system)
    assert counts["tokens_saved"] < 0  # input overhead must not be hidden
    assert calls.count(original.text) == 1


def test_code_compression_reaches_service_and_leaves_system_untouched():
    original = pipeline.Prompt("Explain:\n```python\nx    =    1\n```\n", "Keep all facts.")
    service = CompressorService(original, SimpleNamespace(tokenizer=SimpleNamespace(count=len)), pipeline="code")
    assert len(service.compressed_text) < len(original.text)
    assert service.compressed_prompt.system == original.system
    assert service.get_token_information()["tokens_saved"] > 0


@pytest.mark.parametrize("level", ["code", "full"])
def test_code_compaction_is_reverted_if_model_token_count_grows(level):
    original = pipeline.Prompt("```python\nx    =    1\n```\n")
    compacted = pipeline.compress(original, level)
    assert compacted.text != original.text
    tokenizer = SimpleNamespace(count=lambda text: 20 if text == compacted.text else 10)
    service = CompressorService(original, SimpleNamespace(tokenizer=tokenizer), pipeline=level)
    counts = service.get_token_information()
    assert service.compressed_prompt.text == original.text
    assert service.compressed_text == original.text
    assert service.compressed_prompt.system == compacted.system
    assert counts["compressed_input_tokens"] == (20 if level == "full" else 10)


@pytest.mark.parametrize("name", ["", "unknown", "FULL"])
def test_unknown_levels_rejected(name):
    with pytest.raises(ValueError, match="unknown compression pipeline"):
        pipeline.compress(pipeline.Prompt("hi"), name)


@pytest.mark.parametrize(("selection", "expected"), [
    ([], ()), ((), ()), ("none", ()),
    ("light", ("concise",)), ("full", ("code", "concise")),
    ("code+concise", ("code", "concise")),
    (["concise", "passthrough", "code"], ("concise", "passthrough", "code")),
])
def test_resolve_stages_preserves_native_lists_and_preset_aliases(selection, expected):
    assert pipeline.resolve_stages(selection) == expected


@pytest.mark.parametrize("selection", [None, {}, [None], ["unknown"], "code+unknown", "code+"])
def test_resolve_stages_rejects_invalid_selection(selection):
    with pytest.raises(ValueError):
        pipeline.resolve_stages(selection)


def test_preset_validation_does_not_accept_arbitrary_stage_labels():
    with pytest.raises(ValueError, match="unknown compression pipeline"):
        pipeline.validate_pipeline("code+concise")
    assert pipeline.resolve_stages("code+concise") == ("code", "concise")


def test_aliases_share_effective_fingerprint_but_stage_order_matters():
    assert pipeline.fingerprint("full") == pipeline.fingerprint(["code", "concise"])
    assert pipeline.fingerprint("full") == pipeline.fingerprint("code+concise")
    assert pipeline.fingerprint("full") != pipeline.fingerprint(["concise", "code"])


def test_stage_counting_includes_rendered_context_sources_and_system():
    original = Prompt("Question", "System", (Chunk("Evidence", "notes.md"),))
    calls = []

    def count(text):
        calls.append(text)
        return len(text)

    service = CompressorService(original, SimpleNamespace(tokenizer=SimpleNamespace(count=count)),
                                stages=["passthrough"])
    counts = service.get_token_information()
    assert service.stages == ["passthrough"]
    assert service.input_text == service.compressed_text == original.render()
    assert counts["uncompressed_input_tokens"] == len(original.render()) + len(original.system)
    assert counts["tokens_saved"] == 0
    assert calls == [original.render(), original.system]


def test_code_stage_compresses_context_without_losing_source_labels():
    original = Prompt("Question", "System", (Chunk("```python\nx    =    1\n```\n", "code.md"),))
    result = pipeline.compress(original, ["code"])
    assert len(result.context[0].text) < len(original.context[0].text)
    assert result.context[0].source == "code.md"
    assert result.text == original.text
    assert result.system == original.system
    assert "[code.md]" in result.render()
    assert original.context[0].text == "```python\nx    =    1\n```\n"


def test_growth_fallback_restores_original_context_and_retains_system_policy():
    original = Prompt("Question", "System", (Chunk("```python\nx    =    1\n```\n", "code.md"),))
    compacted = pipeline.compress(original, ["code", "concise"])
    assert compacted.render() != original.render()
    tokenizer = SimpleNamespace(count=lambda text: 20 if text == compacted.render() else 10)
    service = CompressorService(original, SimpleNamespace(tokenizer=tokenizer), stages=["code", "concise"])
    counts = service.get_token_information()
    assert service.compressed_prompt.context == original.context
    assert service.compressed_prompt.text == original.text
    assert service.compressed_prompt.system == compacted.system
    assert service.compressed_text == original.render()
    assert counts["uncompressed_input_tokens"] == counts["compressed_input_tokens"] == 20


def test_service_rejects_conflicting_stage_and_preset_selections():
    with pytest.raises(ValueError, match="choose either"):
        CompressorService("Question", SimpleNamespace(), stages=["code"], pipeline="light")
