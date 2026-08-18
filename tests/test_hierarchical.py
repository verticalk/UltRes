"""Tests for the hierarchical summary tree builder."""

from __future__ import annotations

from pathlib import Path

from ultres.memory.hierarchical import _extractive_notes, assign_topic, build_tree
from ultres.memory.store import KnowledgeStore
from ultres.search.base import CodeBlock, Page


def test_extractive_notes_picks_key_paragraphs():
    page = Page(
        url="https://example.com/a",
        title="Page A",
        text="Short.\n\nThis is a longer paragraph with more substance.\n\nAnother short.",
        code_blocks=[],
    )
    notes = _extractive_notes(page, max_paras=2)
    assert "key_paragraphs" in notes
    assert len(notes["key_paragraphs"]) == 2
    # The longest paragraph should be selected.
    assert any("longer paragraph" in p for p in notes["key_paragraphs"])


def test_extractive_notes_records_code_ids():
    cb = CodeBlock(code_id="code_1", language="python", content="print('hi')", source_url="u")
    page = Page(url="u", title="t", text="text", code_blocks=[cb])
    notes = _extractive_notes(page)
    assert notes["code_ids"] == ["code_1"]


def test_assign_topic_uses_first_heading():
    page = Page(
        url="u",
        title="C++ Calculator",
        text="## Building a parser\n\ncontent",
        code_blocks=[],
    )
    notes = {"headings": ["Building a parser", "Evaluation"]}
    assignment = assign_topic(page, notes)
    assert assignment.topic_name == "Building a parser"
    assert assignment.topic_id.startswith("topic_")
    assert assignment.subtopic_id.startswith("sub_")


def test_build_tree_without_summarizer(tmp_path: Path):
    store = KnowledgeStore(tmp_path, query_id="q1")
    page = Page(
        url="https://example.com/a",
        title="A",
        text="## Heading\n\ncontent here",
        code_blocks=[CodeBlock(code_id="code_x", language="cpp", content="int x;", source_url="u")],
    )
    info = build_tree(store, page, summarizer=None)
    assert info["doc_id"].startswith("doc_")
    assert info["code_ids"] == ["code_x"]
    # Notes should be persisted.
    notes = store.get_notes(info["doc_id"])
    assert notes is not None
    assert "Heading" in notes["headings"]


def test_build_tree_with_summarizer(tmp_path: Path):
    store = KnowledgeStore(tmp_path, query_id="q1")
    page = Page(url="u", title="A", text="Some content to summarize.", code_blocks=[])

    def fake_summarizer(text: str, max_words: int) -> str:
        return f"SUMMARY({max_words}): {text[:20]}"

    info = build_tree(store, page, summarizer=fake_summarizer)
    sub_summary = store.get_subtopic_summary(info["subtopic_id"])
    # Subtopic summaries are stored with a "## {name}\n\n" heading prefix.
    assert "SUMMARY(150):" in sub_summary
