"""Disk-backed Knowledge Store.

Layout under `.ultres/store/<query-id>/`:
    raw/<doc-id>.md          # raw extracted text of a fetched page
    notes/<doc-id>.json      # structured notes (extractive summary)
    summaries/topic/<topic>.md
    summaries/subtopic/<subtopic>.md
    code/<code-id>.txt       # verbatim code blocks (NEVER summarized)
    meta.json                # query metadata, list of docs/topics

The store is the "cold tier" of UltRes's effectively-unlimited context. The
agent navigates it via tool calls (load_slice, load_summary, load_code, recall).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ultres.search.base import CodeBlock, Page


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocRecord:
    """Metadata for a fetched document."""

    doc_id: str
    url: str
    title: str
    fetched_at: float
    tokens: int = 0
    code_ids: list[str] = field(default_factory=list)
    fetch_error: str | None = None


@dataclass
class TopicRecord:
    topic_id: str
    name: str
    subtopic_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)


@dataclass
class SubtopicRecord:
    subtopic_id: str
    name: str
    topic_id: str
    doc_ids: list[str] = field(default_factory=list)


@dataclass
class QueryMeta:
    query_id: str
    user_query: str
    created_at: float
    docs: dict[str, dict[str, Any]] = field(default_factory=dict)
    topics: dict[str, dict[str, Any]] = field(default_factory=dict)
    subtopics: dict[str, dict[str, Any]] = field(default_factory=dict)
    code: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_id(url: str) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:12]
    return f"doc_{h}"


def _topic_id(name: str) -> str:
    h = hashlib.sha1(name.lower().encode()).hexdigest()[:10]
    return f"topic_{h}"


def _subtopic_id(topic_id: str, name: str) -> str:
    h = hashlib.sha1(f"{topic_id}/{name.lower()}".encode()).hexdigest()[:10]
    return f"sub_{h}"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class KnowledgeStore:
    """Filesystem-backed knowledge store for a single query."""

    def __init__(self, root: Path, query_id: str | None = None):
        self.root = Path(root)
        if query_id is None:
            query_id = uuid.uuid4().hex[:12]
        self.query_id = query_id
        self.qdir = self.root / query_id
        self.raw_dir = self.qdir / "raw"
        self.notes_dir = self.qdir / "notes"
        self.sum_topic_dir = self.qdir / "summaries" / "topic"
        self.sum_sub_dir = self.qdir / "summaries" / "subtopic"
        self.code_dir = self.qdir / "code"
        self.meta_path = self.qdir / "meta.json"
        for d in (
            self.raw_dir,
            self.notes_dir,
            self.sum_topic_dir,
            self.sum_sub_dir,
            self.code_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self.meta = self._load_meta()

    # -- meta ----------------------------------------------------------

    def _load_meta(self) -> QueryMeta:
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text("utf-8"))
            return QueryMeta(**data)
        return QueryMeta(
            query_id=self.query_id,
            user_query="",
            created_at=time.time(),
        )

    def _save_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps(asdict(self.meta), indent=2, ensure_ascii=False),
            "utf-8",
        )

    def set_user_query(self, q: str) -> None:
        self.meta.user_query = q
        self._save_meta()

    # -- docs ----------------------------------------------------------

    def add_page(self, page: Page) -> DocRecord:
        """Persist a fetched Page: raw text + code blocks + metadata."""
        doc_id = _doc_id(page.url)
        if doc_id in self.meta.docs:
            return DocRecord(**self.meta.docs[doc_id])

        # Raw text
        raw_path = self.raw_dir / f"{doc_id}.md"
        raw_content = f"# {page.title}\n\nSource: {page.url}\n\n{page.text}"
        raw_path.write_text(raw_content, "utf-8")

        # Code blocks
        code_ids: list[str] = []
        for cb in page.code_blocks:
            self._save_code_block(cb)
            code_ids.append(cb.code_id)
            self.meta.code[cb.code_id] = {
                "language": cb.language,
                "source_url": cb.source_url,
                "doc_id": doc_id,
                "chars": len(cb.content),
            }

        rec = DocRecord(
            doc_id=doc_id,
            url=page.url,
            title=page.title,
            fetched_at=time.time(),
            tokens=_estimate_tokens(raw_content),
            code_ids=code_ids,
            fetch_error=page.fetch_error,
        )
        self.meta.docs[doc_id] = asdict(rec)
        self._save_meta()
        return rec

    def get_doc_raw(self, doc_id: str) -> str:
        path = self.raw_dir / f"{doc_id}.md"
        return path.read_text("utf-8") if path.exists() else ""

    def get_doc_slice(self, doc_id: str, section: str | None = None) -> str:
        """Return a doc's raw text, optionally filtered to a section heading.

        `section` matches a markdown heading (case-insensitive substring).
        Returns the heading + body until the next same-or-higher level heading.
        """
        text = self.get_doc_raw(doc_id)
        if not section:
            return text
        lines = text.splitlines()
        out: list[str] = []
        capturing = False
        target = section.lower()
        for line in lines:
            if line.startswith("#"):
                if capturing:
                    # Stop at next same-or-higher heading.
                    if line.lstrip("#").strip().lower().startswith(target):
                        # same heading repeated — keep going
                        out.append(line)
                        continue
                    level = len(line) - len(line.lstrip("#"))
                    start_level = len(out[0]) - len(out[0].lstrip("#")) if out else 1
                    if level <= start_level:
                        break
                if target in line.lower():
                    capturing = True
                    out.append(line)
                    continue
            if capturing:
                out.append(line)
        return "\n".join(out) if out else text

    # -- notes ---------------------------------------------------------

    def save_notes(self, doc_id: str, notes: dict[str, Any]) -> None:
        path = self.notes_dir / f"{doc_id}.json"
        path.write_text(json.dumps(notes, indent=2, ensure_ascii=False), "utf-8")

    def get_notes(self, doc_id: str) -> dict[str, Any] | None:
        path = self.notes_dir / f"{doc_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text("utf-8"))

    # -- code ----------------------------------------------------------

    def _save_code_block(self, cb: CodeBlock) -> None:
        path = self.code_dir / f"{cb.code_id}.txt"
        header = f"// language: {cb.language or 'unknown'}\n// source: {cb.source_url}\n"
        path.write_text(header + cb.content, "utf-8")

    def get_code(self, code_id: str) -> str | None:
        path = self.code_dir / f"{code_id}.txt"
        return path.read_text("utf-8") if path.exists() else None

    # -- summaries -----------------------------------------------------

    def save_topic_summary(self, topic_id: str, name: str, summary: str) -> None:
        path = self.sum_topic_dir / f"{topic_id}.md"
        path.write_text(f"# {name}\n\n{summary}", "utf-8")
        if topic_id not in self.meta.topics:
            self.meta.topics[topic_id] = {
                "topic_id": topic_id,
                "name": name,
                "subtopic_ids": [],
                "doc_ids": [],
            }
        else:
            self.meta.topics[topic_id]["name"] = name
        self._save_meta()

    def save_subtopic_summary(
        self, subtopic_id: str, name: str, topic_id: str, summary: str
    ) -> None:
        path = self.sum_sub_dir / f"{subtopic_id}.md"
        path.write_text(f"## {name}\n\n{summary}", "utf-8")
        self.meta.subtopics[subtopic_id] = {
            "subtopic_id": subtopic_id,
            "name": name,
            "topic_id": topic_id,
            "doc_ids": [],
        }
        topic = self.meta.topics.setdefault(
            topic_id,
            {"topic_id": topic_id, "name": "", "subtopic_ids": [], "doc_ids": []},
        )
        if subtopic_id not in topic["subtopic_ids"]:
            topic["subtopic_ids"].append(subtopic_id)
        self._save_meta()

    def get_topic_summary(self, topic_id: str) -> str:
        path = self.sum_topic_dir / f"{topic_id}.md"
        return path.read_text("utf-8") if path.exists() else ""

    def get_subtopic_summary(self, subtopic_id: str) -> str:
        path = self.sum_sub_dir / f"{subtopic_id}.md"
        return path.read_text("utf-8") if path.exists() else ""

    def list_topics(self) -> list[dict[str, Any]]:
        return list(self.meta.topics.values())

    # -- linking -------------------------------------------------------

    def link_doc_to_topic(self, doc_id: str, topic_id: str) -> None:
        topic = self.meta.topics.setdefault(
            topic_id,
            {"topic_id": topic_id, "name": "", "subtopic_ids": [], "doc_ids": []},
        )
        if doc_id not in topic["doc_ids"]:
            topic["doc_ids"].append(doc_id)
        self._save_meta()

    # -- static helpers ------------------------------------------------

    @staticmethod
    def make_doc_id(url: str) -> str:
        return _doc_id(url)

    @staticmethod
    def make_topic_id(name: str) -> str:
        return _topic_id(name)

    @staticmethod
    def make_subtopic_id(topic_id: str, name: str) -> str:
        return _subtopic_id(topic_id, name)
