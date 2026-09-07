"""The chat template tojson filter must render exactly as transformers' does."""

import json

from common.templating import _create_environment, _tojson_compat

TOOL = {
    "type": "function",
    "function": {
        "name": "查询天气",
        "description": '获取城市的天气 <b>&</b> it\'s "quoted" ünïcödé 🚀',
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def transformers_tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """Verbatim copy of the filter transformers registers (chat_template_utils.py)."""

    return json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def render(source, **kwargs):
    return _create_environment().from_string(source).render(**kwargs)


def test_bare_tojson_matches_transformers_byte_for_byte():
    assert render("{{ t | tojson }}", t=TOOL) == transformers_tojson(TOOL)
    assert render("{{ t | tojson }}", t=TOOL) == (
        '{"type": "function", "function": {"name": "查询天气", "description": '
        '"获取城市的天气 <b>&</b> it\'s \\"quoted\\" ünïcödé 🚀", "parameters": '
        '{"type": "object", "properties": {"city": {"type": "string"}}}}}'
    )


def test_keyword_arguments_match_transformers():
    for kwargs in (
        {"ensure_ascii": True},
        {"indent": 2},
        {"separators": (",", ":")},
        {"sort_keys": True},
        {"ensure_ascii": False, "indent": 4, "sort_keys": True},
    ):
        assert _tojson_compat(TOOL, **kwargs) == transformers_tojson(TOOL, **kwargs), kwargs


def test_template_keyword_forms():
    assert render("{{ t | tojson(ensure_ascii=False) }}", t=TOOL) == transformers_tojson(TOOL)
    assert render("{{ t | tojson(ensure_ascii=True) }}", t=TOOL) == transformers_tojson(
        TOOL, ensure_ascii=True
    )
    assert render("{{ t | tojson(separators=(',', ':')) }}", t=TOOL) == transformers_tojson(
        TOOL, separators=(",", ":")
    )
    assert render("{{ t | tojson(indent=2) }}", t=TOOL) == transformers_tojson(TOOL, indent=2)


def test_returns_plain_string_not_markup():
    # A Markup return would HTML-escape the plain string it is concatenated with
    assert render("{{ '<b>' ~ (x | tojson) }}", x={"k": "<v>"}) == '<b>{"k": "<v>"}'
