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
