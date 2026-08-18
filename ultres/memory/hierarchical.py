"""Hierarchical summary tree builder.

For each retrieved doc:
    raw page  ->  extractive notes  ->  subtopic summary  ->  topic summary

Code blocks bypass summarization entirely (they're stored verbatim and indexed
separately). Summaries are produced by the same Qwen model in a cheap
short-context call.

The three-level tree lets the model `load_summary(topic)` for a compressed view
and then drill down via `load_slice(doc_id, section)` or `load_code(code_id)`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from ultres.memory.store import KnowledgeStore
from ultres.search.base import Page


# Type alias for the summarizer function the agent loop provides.
# It takes (text, max_words) and returns a summary string.
Summarizer = Callable[[str, int], str]


# ---------------------------------------------------------------------------
# Extractive notes (no model needed)
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def _extractive_notes(page: Page, max_paras: int = 8) -> dict[str, Any]:
    """Build structured notes from a page without a model call.

    Picks the first/longest paragraphs, lists headings, and records code block
    IDs. This is the "warm tier" representation that gets vector-indexed.
    """
    paras = _split_paragraphs(page.text)
    # Prefer paragraphs with substance.
    paras.sort(key=lambda p: len(p), reverse=True)
    selected = paras[:max_paras]
    # Restore original order for readability.
    selected.sort(key=lambda p: page.text.find(p))

    headings = re.findall(r"^#{1,6}\s+(.+)$", page.text, re.MULTILINE)

    return {
        "url": page.url,
        "title": page.title,
        "headings": headings[:30],
        "key_paragraphs": selected,
        "code_ids": [cb.code_id for cb in page.code_blocks],
        "fetch_error": page.fetch_error,
    }


# ---------------------------------------------------------------------------
# Topic clustering (simple keyword-based for v1)
# ---------------------------------------------------------------------------

@dataclass
class TopicAssignment:
    topic_id: str
    topic_name: str
    subtopic_id: str
    subtopic_name: str


def _guess_topic(title: str, headings: list[str]) -> str:
    """Very lightweight topic guess from title + first heading.

    v1 uses this heuristic; v2 can replace it with a model-driven clustering
    pass over all notes.
    """
    text = (title + " " + " ".join(headings[:3])).lower()
    # Pick the first heading-ish noun phrase; fall back to the title.
    for h in headings:
        return h.strip()
    return title.strip() or "general"


def assign_topic(page: Page, notes: dict[str, Any]) -> TopicAssignment:
    topic_name = _guess_topic(page.title, notes.get("headings", []))
    topic_id = KnowledgeStore.make_topic_id(topic_name)
    subtopic_name = page.title.strip() or topic_name
    subtopic_id = KnowledgeStore.make_subtopic_id(topic_id, subtopic_name)
    return TopicAssignment(
        topic_id=topic_id,
        topic_name=topic_name,
        subtopic_id=subtopic_id,
        subtopic_name=subtopic_name,
    )


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

def build_tree(
    store: KnowledgeStore,
    page: Page,
    summarizer: Summarizer | None = None,
) -> dict[str, Any]:
    """Ingest a page into the store: notes, topic/subtopic assignment, summaries.

    If `summarizer` is provided, it's used to produce subtopic and topic
    summaries. Otherwise only extractive notes are written (no model summaries).
    Returns metadata about what was stored.
    """
    doc_rec = store.add_page(page)

    # 1. Extractive notes (no model).
    notes = _extractive_notes(page)
    store.save_notes(doc_rec.doc_id, notes)

    # 2. Topic assignment.
    assignment = assign_topic(page, notes)
    store.link_doc_to_topic(doc_rec.doc_id, assignment.topic_id)

    # 3. Subtopic summary (model, short context).
    sub_summary = ""
    if summarizer and page.text:
        sub_summary = summarizer(page.text, max_words=150)
    store.save_subtopic_summary(
        assignment.subtopic_id,
        assignment.subtopic_name,
        assignment.topic_id,
        sub_summary or "\n".join(notes["key_paragraphs"][:2]),
    )

    # 4. Topic summary (model, short context) — aggregate existing subtopics.
    topic_summary = ""
    if summarizer:
        # Gather all subtopic summaries for this topic.
        parts: list[str] = []
        for sub_id in store.meta.topics.get(assignment.topic_id, {}).get("subtopic_ids", []):
            s = store.get_subtopic_summary(sub_id)
            if s:
                parts.append(s)
        if parts:
            topic_summary = summarizer("\n\n".join(parts), max_words=200)
    store.save_topic_summary(
        assignment.topic_id,
        assignment.topic_name,
        topic_summary or sub_summary,
    )

    return {
        "doc_id": doc_rec.doc_id,
        "topic_id": assignment.topic_id,
        "topic_name": assignment.topic_name,
        "subtopic_id": assignment.subtopic_id,
        "subtopic_name": assignment.subtopic_name,
        "code_ids": doc_rec.code_ids,
    }
