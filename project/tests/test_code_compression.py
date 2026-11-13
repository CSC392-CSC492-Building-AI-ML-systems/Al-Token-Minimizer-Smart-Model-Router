"""Syntax and prompt-context regressions for CSC-17."""

import ast
import io
import tokenize

import pytest
from tree_sitter import Language, Parser
import tree_sitter_javascript

from compressor.code import compress_code_blocks


def fenced(source, language="python"):
    return f"```{language}\n{source}```\n"


def contents(prompt):
    return prompt.split("\n", 1)[1].rsplit("```", 1)[0]


def payload(source):
    return [
        (item.type, item.string)
        for item in tokenize.generate_tokens(io.StringIO(source).readline)
        if item.type in {tokenize.STRING, tokenize.COMMENT}
    ]


def test_python_compacts_layout_and_preserves_runtime_and_prompt_context():
    source = '''\n# prices are in cents
def price(value):
    """Return cents, not dollars.  Keep this contract."""

    if value < 0:
        raise ValueError("not valid  // #")
    return (
        value + 100
    )
'''
    compressed = contents(compress_code_blocks(fenced(source)))
    assert len(compressed) < len(source)
    assert ast.dump(ast.parse(compressed)) == ast.dump(ast.parse(source))
    assert payload(compressed) == payload(source)
    namespace = {}
    exec(compressed, namespace)
    assert namespace["price"](125) == 225
    assert namespace["price"].__doc__ == "Return cents, not dollars.  Keep this contract."
    with pytest.raises(ValueError, match="not valid"):
        namespace["price"](-1)


def test_python_preserves_multiline_string_bytes_and_type_directives():
    source = '''def example():
    # preserve units, noqa, and tool directives
    text = """first
        indented   text
    last"""
    value = 1  # type: int
    return text, value
'''
    compressed = contents(compress_code_blocks(fenced(source, "py")))
    assert payload(compressed) == payload(source)
    assert ast.dump(ast.parse(compressed, type_comments=True)) == ast.dump(ast.parse(source, type_comments=True))


@pytest.mark.parametrize("directive", [
    "# pylint: disable=unused-variable", "# noqa: F841", "# fmt: off",
    "# type: ignore[assignment]", "# isort: skip", "# pyright: strict",
    "# ruff: noqa", "# coding: utf-8", "#!/usr/bin/env python3",
])
def test_python_tool_directives_keep_original_indentation_and_lines(directive):
    source = f"def example():\n    {directive}\n    value = 42\n    return value\n"
    prompt = fenced(source)
    assert compress_code_blocks(prompt) == prompt


@pytest.mark.parametrize(("language", "source"), [
    ("python", "import inspect\n\ndef example():\n    return 42\nprint(inspect.getsource(example))\n"),
    ("python", "def example():\n    return 42\nprint(example.__code__.co_firstlineno)\n"),
    ("js", "function example( ) {\n    return 42;\n}\nconsole.log(example.toString());\n"),
    ("js", "function example( ) {\n    return 42;\n}\nFunction.prototype.toString.call(example);\n"),
])
def test_known_source_reflection_passes_through(language, source):
    prompt = fenced(source, language)
    assert compress_code_blocks(prompt) == prompt


@pytest.mark.parametrize("source", [
    "def broken(:\n    pass\n",
    "return 42\n",
    "text = '''unfinished\n",
    "if True:\n  a = 1\n b = 2\n",
])
def test_invalid_python_passes_through(source):
    prompt = fenced(source)
    assert compress_code_blocks(prompt) == prompt


def test_python_does_not_execute_prompt_code(tmp_path):
    destination = tmp_path / "must-not-exist"
    source = f"from pathlib import Path\n\nPath({str(destination)!r}).write_text('oops')\n"
    compress_code_blocks(fenced(source))
    assert not destination.exists()


def javascript_tree(source):
    parser = Parser(Language(tree_sitter_javascript.language()))
    tree = parser.parse(source.encode())
    assert not tree.root_node.has_error
    return str(tree.root_node)


def test_javascript_compacts_modern_syntax_without_losing_comments_or_literals():
    source = '''"use strict";

// timeout is in milliseconds
function config( value ) {
    const pattern = /https?:\\/\\/[^ ]+/g;
    const text = `line one
        literal indentation ${value}`;
    return { timeout: value?.timeout ?? 1000, pattern, text };
}
'''
    compressed = contents(compress_code_blocks(fenced(source, "javascript")))
    assert len(compressed) < len(source)
    assert javascript_tree(compressed) == javascript_tree(source)
    assert '"use strict";' in compressed
    assert "// timeout is in milliseconds\n" in compressed
    assert "/https?:\\/\\/[^ ]+/g" in compressed
    assert "`line one\n        literal indentation ${value}`" in compressed


@pytest.mark.parametrize("source", [
    "function f() {\n    return\n        { answer: 42 };\n}\n",
    "let x = 1;\n\nlet y = 2;\nx\n    ++y;\n",
    "const result = 1 + // preserve the next line\n    2;\n",
    "const f = async\n    () => 1;\n",  # invalid restricted newline is left alone
    "function broken( {\n",
    "const url = 'http://example.com/a  b';\n\n",
    "const text = String.raw`two  spaces\\n`;\n\n",
    "#!/usr/bin/env node\n\nconst value = 42;\n",
    "export const answer = 1_000n;\n\n",
])
def test_javascript_retains_asi_boundaries_and_fails_open(source):
    prompt = fenced(source, "js")
    compressed = contents(compress_code_blocks(prompt))
    parser = Parser(Language(tree_sitter_javascript.language()))
    if parser.parse(source.encode()).root_node.has_error:
        assert compressed == source
    else:
        assert javascript_tree(compressed) == javascript_tree(source)
    assert len(compressed) <= len(source)


def test_multiple_fences_only_change_supported_bodies():
    python = "def f():\n    return 42\n"
    javascript = "function f( ) {\n    return 42;\n}\n"
    prefix = "Please compare:\n\n"
    middle = "\nThis prose  has   meaningful spacing.\n"
    unknown = "```text\n```python\nnot actually code\n```\n"
    prompt = prefix + fenced(python) + middle + fenced(javascript, "js") + unknown
    result = compress_code_blocks(prompt)
    assert result.startswith(prefix)
    assert middle in result
    assert result.endswith(unknown)
    assert len(result) < len(prompt)


@pytest.mark.parametrize("prompt", [
    "plain prose  with   spacing\n",
    "```python\ndef f():\n    return 42\n",
    "```sql\nSELECT   *   FROM table;\n```\n",
    "```\ndef f():\n    return 42\n```\n",
    "````python\ndef f():\n    return '```'\n```\n",  # closing fence is too short
])
def test_unknown_or_unclosed_fences_pass_through(prompt):
    assert compress_code_blocks(prompt) == prompt


def test_long_fence_and_tilde_fences():
    prompt = "````python\ndef f():\n    return '```'\n````\n~~~js\nfunction f( ) {\n    return 1;\n}\n~~~\n"
    result = compress_code_blocks(prompt)
    assert result.startswith("````python\n")
    assert "\n````\n~~~js\n" in result
    assert result.endswith("\n~~~\n")
    assert len(result) < len(prompt)


def test_oversized_code_passes_through():
    prompt = fenced("# " + "x" * 256_000 + "\n")
    assert compress_code_blocks(prompt) == prompt


def test_vertical_tab_inside_literal_does_not_close_markdown_fence():
    source = 'def example():\n    return """first\x0b```\nlast"""\n'
    prompt = fenced(source)
    compressed = contents(compress_code_blocks(prompt))
    assert payload(compressed) == payload(source)
    assert ast.dump(ast.parse(compressed)) == ast.dump(ast.parse(source))
