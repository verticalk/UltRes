"""Planner: decompose a user query into research subtasks.

One model call with the user query -> a JSON list of research subtasks
(search queries / topics to investigate). This seeds the research loop.
"""

from __future__ import annotations

import json
import re
from typing import Any


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
