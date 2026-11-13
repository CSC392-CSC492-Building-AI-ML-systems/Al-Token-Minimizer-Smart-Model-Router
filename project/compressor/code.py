"""Conservative, parser-checked compression of fenced Python and JavaScript.

Only structural whitespace is shortened. Comments, docstrings, strings, regexes,
and template literals can carry information needed to answer the prompt, so they
are retained. A candidate is accepted only if its syntax and literal tokens are
unchanged and it is shorter. This preserves syntax, not source coordinates or
source-reflection results (for example JavaScript Function.toString()).
"""

import ast
import io
import re
import token
import tokenize

from tree_sitter import Language, Parser
import tree_sitter_javascript


_JAVASCRIPT = Language(tree_sitter_javascript.language())
_MAX_CODE_BYTES = 256_000
_OPEN_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)$")
_LINES = re.compile(r"[^\r\n]*(?:\r\n|\r|\n|$)")
_PYTHON_DIRECTIVE = re.compile(
    r"^#!|\b(?:noqa|type\s*:|pylint\s*:|fmt\s*:|isort\s*:|pyright\s*:|"
    r"mypy\s*:|ruff\s*:|pyre-|coding\s*[:=])", re.IGNORECASE,
)
_SOURCE_REFLECTION = {"getsource", "getsourcelines", "getframeinfo", "__code__", "__traceback__"}
_PYTHON_LAYOUT = {
    token.ENDMARKER, token.INDENT, token.DEDENT, token.NEWLINE, tokenize.NL,
}
# Treat these entire nodes as opaque, including whitespace within literals.
_JS_OPAQUE = {"string", "template_string", "regex", "comment", "hash_bang_line", "jsx_text"}
_JS_DELIMITERS = {b"(", b")", b"[", b"]", b"{", b"}", b",", b";", b":"}


def _python_tokens(source):
    return list(tokenize.generate_tokens(io.StringIO(source).readline))


def _python_payload(tokens):
    return [(item.type, item.string) for item in tokens if item.type not in _PYTHON_LAYOUT]


def _compress_python(source):
    try:
        original = ast.parse(source, type_comments=True)
        # ast.parse alone also accepts incomplete executable contexts such as
        # a top-level return. Compiling checks them without executing anything.
        compile(original, "<compression>", "exec")
        original_tokens = _python_tokens(source)
        if any(
            (item.type == tokenize.COMMENT and _PYTHON_DIRECTIVE.search(item.string))
            or (item.type == token.NAME and item.string in _SOURCE_REFLECTION)
            for item in original_tokens
        ):
            # Tool directives can depend on line numbers and indentation, which
            # AST equality alone does not capture. Source reflection does too.
            return source
        compact_tokens = []
        depth = 0
        for item in original_tokens:
            value = item.string
            if item.type == token.INDENT:
                depth += 1
                value = " " * depth
            elif item.type == token.DEDENT:
                depth -= 1
            elif item.type == tokenize.NL:
                # NEWLINE ends a Python statement. NL is optional layout,
                # except when it terminates a comment.
                if not compact_tokens or compact_tokens[-1][0] != tokenize.COMMENT:
                    continue
            compact_tokens.append((item.type, value))
        candidate = tokenize.untokenize(compact_tokens)
        if len(candidate) >= len(source):
            return source
        parsed = ast.parse(candidate, type_comments=True)
        if ast.dump(original) != ast.dump(parsed):
            return source
        if _python_payload(original_tokens) != _python_payload(_python_tokens(candidate)):
            return source
        return candidate
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError):
        return source


def _js_parts(root, source):
    """Return an offset-independent tree signature and exact atomic tokens."""
    signature, atoms = [], []
    pending = [(root, False)]
    while pending:
        node, closing = pending.pop()
        if closing:
            signature.append(None)
        elif node.is_missing or node.is_error:
            return None, []
        elif node.child_count == 0 or node.type in _JS_OPAQUE:
            raw = source[node.start_byte:node.end_byte]
            signature.append((node.type, raw))
            if raw:
                atoms.append((node.start_byte, node.end_byte, raw))
        else:
            signature.append(node.type)
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(node.children))
    return signature, atoms


def _js_gap(gap, previous, following):
    # Newlines affect automatic semicolon insertion, return/throw, async,
    # postfix operators, and line comments. Retain a line boundary everywhere
    # one existed; only remove extra blank lines and indentation.
    if any(mark in gap for mark in (b"\n", b"\r", b"\xe2\x80\xa8", b"\xe2\x80\xa9")):
        return b"\n"
    if not gap or previous in _JS_DELIMITERS or following in _JS_DELIMITERS:
        return b""
    return b" "


def _compress_javascript(source):
    data = source.encode("utf-8")
    # Parser instances are local because requests may run concurrently.
    parser = Parser(_JAVASCRIPT)
    original = parser.parse(data)
    if original.root_node.has_error:
        return source
    signature, atoms = _js_parts(original.root_node, data)
    if signature is None or not atoms:
        return source
    if any(raw == b"toString" for _, _, raw in atoms):
        # Function.toString exposes exact source text. Conservatively preserve
        # any snippet using that property, including aliased prototype calls.
        return source
    parts = []
    end, previous = 0, b""
    for start, next_end, raw in atoms:
        gap = data[end:start]
        # Every byte not covered by a syntax node must be whitespace. Avoid
        # silently dropping unsupported parser content.
        if gap and not gap.decode("utf-8").isspace():
            return source
        parts.extend((_js_gap(gap, previous, raw), raw))
        end, previous = next_end, raw
    tail = data[end:]
    if tail and not tail.decode("utf-8").isspace():
        return source
    parts.append(_js_gap(tail, previous, b""))
    candidate = b"".join(parts)
    if len(candidate) >= len(data):
        return source
    reparsed = parser.parse(candidate)
    if reparsed.root_node.has_error:
        return source
    new_signature, _ = _js_parts(reparsed.root_node, candidate)
    return candidate.decode("utf-8") if signature == new_signature else source


def _compress_block(source, language):
    try:
        if len(source.encode("utf-8")) > _MAX_CODE_BYTES:
            return source
        if language in {"python", "py"}:
            return _compress_python(source)
        if language in {"javascript", "js"}:
            return _compress_javascript(source)
    except (UnicodeError, RecursionError, ValueError):
        # Prompt snippets may be incomplete, malformed, or too deeply nested.
        # Compression must not prevent the model from receiving those snippets.
        pass
    return source


def compress_code_blocks(text: str) -> str:
    """Shorten supported fenced blocks, keeping prose and all fences intact.

    Unknown languages, unfinished fences, invalid code, and blocks over 256 KB
    pass through unchanged. No user-provided code is executed.
    """
    # Markdown line boundaries are CR/LF. str.splitlines also splits characters
    # such as vertical tabs that may occur inside a source string literal.
    lines = [line for line in _LINES.findall(text) if line]
    output = []
    index = 0
    while index < len(lines):
        opening = _OPEN_FENCE.fullmatch(lines[index].rstrip("\r\n"))
        if opening is None or (opening[1][0] == "`" and "`" in opening[2]):
            output.append(lines[index])
            index += 1
            continue
        marker, info = opening.groups()
        closing = re.compile(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*$")
        end = index + 1
        while end < len(lines) and closing.fullmatch(lines[end].rstrip("\r\n")) is None:
            end += 1
        if end == len(lines):
            output.extend(lines[index:])
            break
        language = info.strip().split(maxsplit=1)[0].lower() if info.strip() else ""
        source = "".join(lines[index + 1:end])
        output.extend((lines[index], _compress_block(source, language), lines[end]))
        index = end + 1
    return "".join(output)
