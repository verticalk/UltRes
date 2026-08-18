"""Tool schemas for the UltRes agent loop (Qwen3 tool-call format).

Tools exposed to the model during the research loop:
    search(query, n)             -> new results into store
    visit(url)                   -> fetch + extract + store + index
    recall(query, k)             -> vector search across current store
    load_summary(topic)          -> inject topic-level summary into hot context
    load_slice(doc_id, section)  -> inject a raw doc section
    load_code(code_id)           -> inject verbatim code block
    finish(answer)               -> terminate loop

The tool-call format follows the Qwen3 / OpenAI-style function-calling schema.
"""

from __future__ import annotations

from typing import Any


# Qwen3 tool-call JSON schema (OpenAI-compatible).
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the web for a query. Returns a compact list of result "
                "titles + URLs (not full content). Use `visit` to fetch a URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "n": {
                        "type": "integer",
                        "description": "Max results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": (
                "Fetch a URL, extract its main content + code blocks, store them "
                "in the knowledge store, and return a compact confirmation with "
                "the doc_id, title, headings, and code_ids. Full content is NOT "
                "returned inline — use load_slice or load_code to pull it in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Semantic search across everything stored so far (notes, "
                "summaries, code snippets). Returns ranked chunks to pull into "
                "context. Use this before re-searching the web."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to recall."},
                    "k": {
                        "type": "integer",
                        "description": "Max hits (default 8).",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_summary",
            "description": (
                "Load a topic-level summary into context. Pass a topic name; "
                "the closest matching topic summary is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic name."},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_slice",
            "description": (
                "Load a section of a raw document into context. Pass doc_id and "
                "an optional section heading to filter to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "section": {
                        "type": "string",
                        "description": "Optional heading to filter to.",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_code",
            "description": (
                "Load a verbatim code block into context by its code_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code_id": {"type": "string"},
                },
                "required": ["code_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Finish the research loop and produce the final answer. Pass "
                "the complete answer as the `answer` argument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the user's query.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


def tool_names() -> list[str]:
    return [t["function"]["name"] for t in TOOL_SCHEMAS]


def system_prompt() -> str:
    """System prompt describing the UltRes research agent's behavior + tool format.

    The Qwen2.5 model doesn't reliably use native tool-calling via the `tools`
    API parameter, so we inject tool descriptions + the calling format directly
    into the system prompt and parse tool calls from the model's text output.
    """
    return (
        "You are UltRes, a self-researching AI. You answer the user's query by "
        "researching the web and reasoning over a disk-backed knowledge store.\n\n"
        "## Available tools\n\n"
        "Call a tool by responding with ONLY a JSON object on a single line:\n"
        '{"name": "<tool_name>", "arguments": {<args>}}\n\n'
        "Tools:\n"
        "- search(query: str, n: int=5): Search the web. Returns titles + URLs.\n"
        "- visit(url: str): Fetch a URL, extract content + code, store it. Returns doc_id + headings + code_ids.\n"
        "- recall(query: str, k: int=8): Semantic search across stored content.\n"
        "- load_summary(topic: str): Load a topic-level summary into context.\n"
        "- load_slice(doc_id: str, section: str=None): Load a raw doc section.\n"
        "- load_code(code_id: str): Load a verbatim code block.\n"
        "- finish(answer: str): Finish and produce the final answer.\n\n"
        "## Mandatory research workflow\n\n"
        "1. search: Call `search` with a relevant query.\n"
        "2. visit: Call `visit` on the MOST relevant URL from the search results. "
        "You MUST visit at least 2 pages before finishing.\n"
        "3. recall: After visiting pages, call `recall` to find relevant stored content.\n"
        "4. load_slice/load_code: Pull specific details into your context.\n"
        "5. finish: Only call `finish` AFTER you have visited at least 2 pages and "
        "have actual research findings. Do NOT answer from your own knowledge.\n\n"
        "## Critical rules\n\n"
        "- ALWAYS call a tool each turn. Do NOT answer directly.\n"
        "- Respond with ONLY the JSON tool call. No extra text.\n"
        "- NEVER call `finish` before visiting at least 2 URLs.\n"
        "- NEVER call `recall` before `visit` — recall searches stored content, "
        "which is empty until you visit pages.\n"
        "- After `search` returns URLs, your NEXT call MUST be `visit`.\n\n"
        "Example tool call:\n"
        '{"name": "search", "arguments": {"query": "C++ std::vector vs std::list"}}\n'
    )
