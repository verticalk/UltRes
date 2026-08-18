"""Planner: decompose a user query into research subtasks.

One model call with the user query -> a JSON list of research subtasks
(search queries / topics to investigate). This seeds the research loop.

v1.2 adds plan_deep() which generates 30-50 subtasks with query expansion
(3-5 variations per subtask) for the deep research pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ultres.research.source_prioritizer import detect_query_type


_PLANNER_PROMPT = """\
You are the planning stage of UltRes, a self-researching AI. Given a user's \
request, produce a JSON plan of research subtasks.

Each subtask is a focused web search query that will help answer the request. \
Produce 3-8 subtasks. Order them from most fundamental to most specific.

Respond with ONLY a JSON object of this exact shape:
{{
  "subtasks": [
    {{"query": "<search query>", "why": "<one sentence why this matters>"}}
  ]
}}

User request:
{query}
"""


_DEEP_PLANNER_PROMPT = """\
You are the planning stage of UltRes deep research, a self-researching AI. \
Given a user's request, produce a comprehensive JSON plan of research subtasks.

Generate 30-50 subtasks covering ALL aspects of the request:
- Core concepts and fundamentals
- Implementation approaches and examples
- Best practices and design patterns
- Common pitfalls and error handling
- Testing and debugging
- Performance and optimization
- Libraries, tools, and frameworks
- Real-world examples and case studies

For code-related requests, include searches for:
- GitHub repositories with working examples
- Documentation and API references
- Tutorial and step-by-step guides
- Stack Overflow solutions

Respond with ONLY a JSON object of this exact shape:
{{
  "subtasks": [
    {{"query": "<search query>", "why": "<one sentence why this matters>"}}
  ]
}}

User request:
{query}
"""


_QUERY_EXPANSION_PROMPT = """\
You are expanding search queries for deep research. For each query below, \
generate 3-5 variations that cover different angles:

- Tutorial angle: "how to <topic> tutorial step by step"
- Example angle: "<topic> example code implementation"
- Reference angle: "<topic> documentation reference API"
- Best practices angle: "<topic> best practices design patterns"
- Pitfall angle: "<topic> common mistakes pitfalls errors"

Respond with ONLY a JSON object:
{{
  "expanded": [
    "<variation 1>", "<variation 2>", ...
  ]
}}

Queries to expand:
{queries}
"""


def plan_query(llm: Any, user_query: str, temperature: float = 0.5) -> list[dict[str, str]]:
    """Decompose `user_query` into a list of subtask dicts via the model.

    Each subtask: {"query": str, "why": str}.
    Falls back to a single subtask (the raw query) if parsing fails.
    """
    prompt = _PLANNER_PROMPT.format(query=user_query)
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a precise JSON-only planner."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=512,
    )
    text = resp["choices"][0]["message"]["content"].strip()

    # Extract the first {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return [{"query": user_query, "why": "fallback: raw user query"}]
    try:
        data = json.loads(m.group(0))
        subtasks = data.get("subtasks", [])
        if not subtasks:
            return [{"query": user_query, "why": "fallback: empty plan"}]
        return subtasks
    except json.JSONDecodeError:
        return [{"query": user_query, "why": "fallback: parse error"}]


def plan_deep(
    llm: Any,
    user_query: str,
    temperature: float = 0.5,
) -> dict[str, Any]:
    """Deep planning: generate 30-50 subtasks + expand into 90-250 queries.

    Returns:
        {
            "subtasks": [{"query": str, "why": str}, ...],
            "expanded_queries": [str, ...],
            "query_type": "code" | "general" | "mixed",
        }
    """
    query_type = detect_query_type(user_query)

    # Call 1: Generate 30-50 subtasks.
    prompt = _DEEP_PLANNER_PROMPT.format(query=user_query)
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a comprehensive research planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            subtasks = data.get("subtasks", [])
        else:
            subtasks = [{"query": user_query, "why": "fallback"}]
    except Exception:
        subtasks = [{"query": user_query, "why": "fallback"}]

    # Call 2: Expand queries with variations.
    base_queries = [s["query"] for s in subtasks]
    # Build expansion prompt with all queries.
    queries_text = "\n".join(f"- {q}" for q in base_queries)
    expansion_prompt = _QUERY_EXPANSION_PROMPT.format(queries=queries_text)

    expanded_queries: list[str] = list(base_queries)  # Start with originals.
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a query expansion engine."},
                {"role": "user", "content": expansion_prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            expanded = data.get("expanded", [])
            expanded_queries.extend(expanded)
    except Exception:
        pass

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_queries: list[str] = []
    for q in expanded_queries:
        q_lower = q.lower().strip()
        if q_lower and q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)

    return {
        "subtasks": subtasks,
        "expanded_queries": unique_queries,
        "query_type": query_type,
    }
