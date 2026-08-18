"""Tests for the disk-backed KnowledgeStore."""

from __future__ import annotations

from pathlib import Path

from ultres.memory.store import KnowledgeStore
from ultres.search.base import CodeBlock, Page


def _make_page(url: str, title: str, text: str, code: list[CodeBlock] | None = None) -> Page:
    return Page(url=url, title=title, text=text, code_blocks=code or [])


def test_add_page_creates_raw_notes_and_meta(tmp_path: Path):
    store = KnowledgeStore(tmp_path, query_id="q1")
    store.set_user_query("test query")
    page = _make_page("https://example.com/a", "Page A", "Some content\n\nMore content.")
    rec = store.add_page(page)

    assert rec.url == "https://example.com/a"
    assert rec.doc_id.startswith("doc_")
    assert (store.raw_dir / f"{rec.doc_id}.md").exists()
    assert rec.doc_id in store.meta.docs
    assert store.meta.user_query == "test query"


def test_add_page_dedupes_by_url(tmp_path: Path):
    store = KnowledgeStore(tmp_path, query_id="q1")
    page = _make_page("https://example.com/a", "Page A", "content")
    r1 = store.add_page(page)
    r2 = store.add_page(page)
    assert r1.doc_id == r2.doc_id
    assert len(store.meta.docs) == 1


def test_code_blocks_stored_verbatim(tmp_path: Path):
    store = KnowledgeStore(tmp_path, query_id="q1")
    cb = CodeBlock(
        code_id="code_abc123",
        language="cpp",
        content="int main() { return 0; }",
        source_url="https://example.com/a",
    )
    page = _make_page("https://example.com/a", "A", "text", code=[cb])
    rec = store.add_page(page)

    assert rec.code_ids == ["code_abc123"]
    stored = store.get_code("code_abc123")
    assert stored is not None
    assert "int main() { return 0; }" in stored
    assert "language: cpp" in stored


def test_doc_slice_filters_by_section(tmp_path: Path):
    store = KnowledgeStore(tmp_path, query_id="q1")
    text = "# Title\n\nintro\n\n## Installation\n\nstep 1\n\n## Usage\n\nstep 2"
    page = _make_page("https://example.com/a", "A", text)
    rec = store.add_page(page)

    install = store.get_doc_slice(rec.doc_id, section="Installation")
    assert "step 1" in install
    assert "step 2" not in install


def test_topic_and_subtopic_summaries(tmp_path: Path):
    store = KnowledgeStore(tmp_path, query_id="q1")
    store.save_topic_summary("topic_x", "C++ Calculators", "Summary of C++ calculators.")
    store.save_subtopic_summary("sub_y", "Parser", "topic_x", "Parser subtopic summary.")

    assert "C++ Calculators" in store.get_topic_summary("topic_x")
    assert "Parser subtopic summary." in store.get_subtopic_summary("sub_y")
    topics = store.list_topics()
    assert any(t["topic_id"] == "topic_x" for t in topics)


def test_link_doc_to_topic(tmp_path: Path):
    store = KnowledgeStore(tmp_path, query_id="q1")
    page = _make_page("https://example.com/a", "A", "content")
    rec = store.add_page(page)
    store.save_topic_summary("topic_z", "Topic Z", "z")
    store.link_doc_to_topic(rec.doc_id, "topic_z")
    assert rec.doc_id in store.meta.topics["topic_z"]["doc_ids"]
