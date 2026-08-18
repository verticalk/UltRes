"""Tests for tool-call JSON parsing and tool schemas."""

from __future__ import annotations

import json

from ultres.agent.research_loop import _extract_tool_call
from ultres.agent.tools import TOOL_SCHEMAS, system_prompt, tool_names


def test_tool_schemas_well_formed():
    names = tool_names()
    assert "search" in names
    assert "visit" in names
    assert "recall" in names
    assert "finish" in names
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_system_prompt_mentions_all_tools():
    sp = system_prompt()
    for name in tool_names():
        assert name in sp


def test_extract_tool_call_native():
    resp = {
        "choices": [
            {
                "message": {
                    "content": "Let me search.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"query": "cpp calculator", "n": 3}),
                            }
                        }
                    ],
                }
            }
        ]
    }
    name, args, content = _extract_tool_call(resp)
    assert name == "search"
    assert args["query"] == "cpp calculator"
    assert args["n"] == 3
    assert "Let me search" in content


def test_extract_tool_call_json_in_content():
    resp = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"name": "visit", "arguments": {"url": "https://example.com"}}\n```',
                }
            }
        ]
    }
    name, args, content = _extract_tool_call(resp)
    assert name == "visit"
    assert args["url"] == "https://example.com"


def test_extract_tool_call_no_tool():
    resp = {
        "choices": [
            {"message": {"content": "I'm thinking about it."}},
        ]
    }
    name, args, content = _extract_tool_call(resp)
    assert name is None
    assert "thinking" in content


def test_extract_tool_call_finish():
    resp = {
        "choices": [
            {
                "message": {
                    "content": "Done.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps({"answer": "Here is the answer."}),
                            }
                        }
                    ],
                }
            }
        ]
    }
    name, args, content = _extract_tool_call(resp)
    assert name == "finish"
    assert args["answer"] == "Here is the answer."
