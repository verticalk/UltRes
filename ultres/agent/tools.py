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
    """System prompt for the UltRes research agent.

    v1.6: Rewritten to leverage Qwen3.8's thinking mode for deeper reasoning.
    The tool schemas are passed via the `tools` API parameter — no need to
    describe the JSON format in the prompt. This prompt focuses on research
    behavior, reasoning strategy, and quality standards.
    """
    return (
        "You are UltRes, a self-researching AI powered by Qwen3.8-27B with "
        "thinking mode. You answer the user's query by researching the web and "
        "reasoning over a disk-backed knowledge store.\n\n"
        "## Thinking mode usage\n\n"
        "You have a thinking/reasoning mode. USE IT to:\n"
        "- Analyze search results before visiting: which URLs are most relevant? "
        "Which are authoritative? Which might be outdated?\n"
        "- Compare information across multiple pages: do they agree or contradict? "
        "Which source is more authoritative?\n"
        "- Identify gaps in coverage before finishing: what's missing? "
        "What did the research NOT cover?\n"
        "- Plan your research strategy: what's the most efficient order of searches?\n\n"
        "## Research strategy\n\n"
        "1. Start BROAD: search for the general topic to understand the landscape.\n"
        "2. Then NARROW: drill into specific aspects, APIs, libraries, patterns.\n"
        "3. Cross-REFERENCE: check authoritative sources (official docs, specs) "
        "against practical sources (tutorials, Stack Overflow, blogs).\n"
        "4. For CODE tasks: ALWAYS find a working example before writing code. "
        "Look for GitHub repos, documentation examples, and Stack Overflow answers "
        "with code. Adapt real code — don't write from scratch.\n"
        "5. For RESEARCH tasks: gather multiple perspectives. Don't rely on a "
        "single source. Look for consensus and disagreement.\n\n"
        "## Mandatory research workflow\n\n"
        "1. search: Call `search` with a relevant query.\n"
        "2. visit: Call `visit` on the MOST relevant URL from the search results. "
        "You MUST visit at least 2 pages before finishing.\n"
        "3. recall: After visiting pages, call `recall` to find relevant stored content.\n"
        "4. load_slice/load_code: Pull specific details into your context using "
        "the EXACT doc_id and code_id returned by visit/recall.\n"
        "5. finish: Only call `finish` AFTER you have visited at least 2 pages, "
        "loaded their content via load_slice, and have actual research findings. "
        "Do NOT answer from your own pretrained knowledge — cite specific facts "
        "from the retrieved content.\n\n"
        "## Quality bar\n\n"
        "Your final answer must meet these standards:\n"
        "- Include SPECIFIC code, not pseudocode or generic advice.\n"
        "- Reference SPECIFIC library versions, API calls, and function names.\n"
        "- Cite which source each piece of information came from.\n"
        "- For code tasks: provide complete, runnable code with error handling.\n"
        "- For research tasks: provide a structured answer with evidence.\n"
        "- If sources disagree, note the disagreement and explain which is more reliable.\n\n"
        "## Critical rules\n\n"
        "- ALWAYS call a tool each turn. Do NOT answer directly without a tool call.\n"
        "- NEVER call `finish` before visiting at least 2 URLs.\n"
        "- NEVER call `recall` before `visit` — recall searches stored content, "
        "which is empty until you visit pages.\n"
        "- After `search` returns URLs, your NEXT call MUST be `visit`.\n"
        "- When calling load_slice, use the EXACT doc_id from the visit/recall result.\n"
        "- Your final answer must reference specific facts from the retrieved content, "
        "not your own pretrained knowledge.\n"
    )
