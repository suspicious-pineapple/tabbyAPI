import ast
import json
import keyword
import re

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool

"""
LFM2 / LFM2.5 (Liquid AI) - Pythonic tool-call list

LFM2-family models natively emit a Pythonic call list rather than
OpenAI-style JSON, wrapped in <|tool_call_start|> / <|tool_call_end|>
sentinels. The argument values are real Python literals.

Raw format:
    <|tool_call_start|>[get_weather(location='Paris', days=3)]<|tool_call_end|>

Parallel calls appear as multiple calls in the list:
    <|tool_call_start|>[f(a=1), g(b='x')]<|tool_call_end|>

Each call becomes a ToolCall whose name is the called function and whose
arguments are the (JSON-encoded) keyword arguments.
"""

TOOLCALL_START = "<|tool_call_start|>"
TOOLCALL_END = "<|tool_call_end|>"

# Python reserved words that are illegal as keyword-argument names in a
# literal call list (e.g. from=...). They are renamed during parsing and
# restored afterwards.
_RESERVED = set(keyword.kwlist)

# Double- or single-quoted string literal (with backslash escapes).
_STR_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

# A reserved word used as a keyword-argument name: identifier followed by '='.
_RESERVED_KW_RE = re.compile(rf"\b({'|'.join(sorted(_RESERVED))})\s*(?==)")

# Leading-zero integer literal (e.g. 07 -> 7), invalid in Python 3.
_LEAD_ZERO_RE = re.compile(r"(?<![\w.])0(\d+)")


def _extract_tool_texts(text: str) -> list[str]:
    """Extract the Pythonic call-list text between sentinel tokens.

    Every <|tool_call_start|>...<|tool_call_end|> span is collected. An
    unterminated start token yields the text from it to the end (stream may
    have been cut off). If no start token appears at all, the whole input is
    treated as a call list so callers that strip sentinels themselves still
    work.
    """
    texts = []
    idx = 0
    while True:
        start = text.find(TOOLCALL_START, idx)
        if start == -1:
            break
        end = text.find(TOOLCALL_END, start)
        if end == -1:
            texts.append(text[start + len(TOOLCALL_START) :].strip())
            break
        texts.append(text[start + len(TOOLCALL_START) : end].strip())
        idx = end + len(TOOLCALL_END)

    if not texts:
        texts = [text.strip()]
    return texts


def _protect_strings(text: str) -> tuple[str, list[str]]:
    """Mask string literals so keyword/leading-zero rewrites never touch
    their contents, returning (masked_text, literals)."""
    literals = []

    def repl(match):
        literals.append(match.group(0))
        return f"@@S{len(literals) - 1}@@"

    return _STR_RE.sub(repl, text), literals


def _restore_strings(text: str, literals: list[str]) -> str:
    for i, lit in enumerate(literals):
        text = text.replace(f"@@S{i}@@", lit)
    return text


def _safe_parse_list(text: str) -> tuple:
    """Parse the Pythonic call list, applying progressive rewrites for text
    that is valid Pythonic-but-not-quite-Python. On success returns
    (module, reserved_kw_map); on failure (None, {}).

    The rewrites (renaming reserved-word keyword args, normalizing
    leading-zero ints) are deterministic, so repeated streaming chunks stay
    consistent — though in this server tool blocks are parsed once at the
    end of the stream, so that isn't relied upon here.
    """
    try:
        return ast.parse(text), {}
    except (SyntaxError, ValueError):
        pass

    # Fallback path: protect strings, then rewrite reserved keywords and
    # leading-zero ints independently, then restore strings and re-parse.
    protected, literals = _protect_strings(text)
    reserved_map = {}

    def _reserved_repl(match):
        original = match.group(1)
        placeholder = f"__lfmkw{len(reserved_map)}__"
        reserved_map[placeholder] = original
        return placeholder

    renamed = _RESERVED_KW_RE.sub(_reserved_repl, protected)
    normalized = _LEAD_ZERO_RE.sub(r"\1", renamed)
    restored = _restore_strings(normalized, literals)

    try:
        return ast.parse(restored), reserved_map
    except (SyntaxError, ValueError):
        return None, {}


def _call_to_toolcall(call: ast.Call, reserved_map: dict) -> ToolCall:
    """Convert a parsed ast.Call into a ToolCall with JSON arguments."""
    # Function name — plain names and dotted paths (os.path.join) both work.
    name = ast.unparse(call.func)

    args: dict = {}
    for kw in call.keywords:
        key = kw.arg
        if key in reserved_map:
            key = reserved_map[key]
        try:
            args[key] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            # Unrenderable literal — fall back to its textual form
            args[key] = ast.unparse(kw.value)

    return ToolCall(function=Tool(name=name, arguments=json.dumps(args, ensure_ascii=False)))


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []

    for block in _extract_tool_texts(text):
        module, reserved_map = _safe_parse_list(block)
        if module is None:
            continue

        node = getattr(module.body[0], "value", None)
        # An empty list ([]) is a deliberate no-op, not a tool call.
        if not (
            isinstance(node, ast.List)
            and node.elts
            and all(isinstance(e, ast.Call) for e in node.elts)
        ):
            continue

        results.extend(_call_to_toolcall(e, reserved_map) for e in node.elts)

    xlogger.debug(
        f"lfm2: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
