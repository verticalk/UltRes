"""Gap detector: identify missing topics after initial clustering.

Reviews cluster labels and coverage, then generates search queries
to fill identified gaps in the research.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ultres.research.clusterer import Cluster


def coverage_map(clusters: list[Cluster], user_query: str, query_type: str) -> str:
    """Build a human-readable coverage map for the model to review."""
    lines = [f"Research coverage for: {user_query}", f"Query type: {query_type}", ""]
    lines.append(f"Total clusters: {len(clusters)}")
    lines.append(f"Total pages: {sum(len(c.pages) for c in clusters)}")
    lines.append(f"Total code blocks: {sum(len(c.code_blocks) for c in clusters)}")
    lines.append("")
    lines.append("Clusters found:")
    for i, c in enumerate(clusters):
        lines.append(
            f"  {i+1}. {c.name} "
            f"({len(c.pages)} pages, {len(c.code_blocks)} code blocks, "
            f"quality={c.quality_score:.2f})"
        )
    return "\n".join(lines)


def detect_gaps(
    llm: Any,
    clusters: list[Cluster],
    user_query: str,
    query_type: str,
    temperature: float = 0.5,
) -> list[str]:
    """Use the model to detect missing topics and generate gap-fill queries.

    Returns a list of search queries that would fill identified gaps.
    If the model says coverage is complete, returns an empty list.
    """
    coverage = coverage_map(clusters, user_query, query_type)

    prompt = (
        f"You are reviewing the research coverage for a task.\n\n"
        f"{coverage}\n\n"
        f"Based on the clusters found, identify any MISSING topics that should be "
        f"researched to fully cover the task. Think about:\n"
        f"- Missing technical aspects (error handling, testing, build system, etc.)\n"
        f"- Missing best practices or common pitfalls\n"
        f"- Missing alternative approaches\n"
        f"- Missing implementation details\n\n"
        f"If the coverage is comprehensive, respond with ONLY 'COMPLETE'.\n"
        f"Otherwise, respond with a JSON object:\n"
        f'{{"gaps": ["search query 1", "search query 2", ...]}}\n\n'
        f"Generate 5-20 gap-fill queries. Be specific."
    )

    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a research gap analyzer."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=512,
        )
        text = resp["choices"][0]["message"]["content"].strip()

        if text.upper().startswith("COMPLETE"):
            return []

        # Parse JSON.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
        gaps = data.get("gaps", [])
        return [g.strip() for g in gaps if g.strip()]
    except Exception:
        return []
