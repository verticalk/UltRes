"""Tests for v1.4 pipeline improvements: empty crawl, critique, unified trajectory."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from ultres.research.pipeline import _critique_and_refine, _save_trajectory_unified
from ultres.research.crawler import CrawlResult, DomainRateLimiter, _content_hash
from ultres.search.base import Page


def test_content_hash_dedup():
    """Content hash should be stable and differentiate pages."""
    h1 = _content_hash("This is page content about C++.")
    h2 = _content_hash("This is page content about C++.")
    h3 = _content_hash("Different content about Rust.")
    assert h1 == h2
    assert h1 != h3


def test_domain_rate_limiter():
    """Domain rate limiter should create per-domain semaphores."""
    import asyncio
    limiter = DomainRateLimiter(max_per_domain=3)
    sem_a = limiter.get_sem("github.com")
    sem_b = limiter.get_sem("github.com")
    sem_c = limiter.get_sem("stackoverflow.com")
    assert sem_a is sem_b  # Same domain, same semaphore
    assert sem_a is not sem_c  # Different domain, different semaphore


def test_critique_verified_keeps_answer():
    """Critique that says VERIFIED should keep the original answer."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "VERIFIED"}}]
    }
    result = _critique_and_refine(
        llm, "test query", "original answer", "brief text", rounds=1,
    )
    assert result == "original answer"


def test_critique_with_code_block_replaces():
    """Critique with a code block should replace the answer with the code."""
    llm = MagicMock()
    critique_text = (
        "The implementation has issues. Here is the corrected version:\n"
        "```cpp\n#include <iostream>\nint main() { return 0; }\n```"
    )
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": critique_text}}]
    }
    result = _critique_and_refine(
        llm, "test query", "short", "brief text", rounds=1,
    )
    assert "int main()" in result
    assert "issues" not in result  # Should be just the code, not the prose


def test_critique_no_code_keeps_answer():
    """Critique without code blocks should keep the original answer."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "The code needs better error handling."}}]
    }
    result = _critique_and_refine(
        llm, "test query", "original answer", "brief text", rounds=1,
    )
    assert result == "original answer"


def test_critique_exception_keeps_answer():
    """If critique fails, keep the original answer."""
    llm = MagicMock()
    llm.create_chat_completion.side_effect = Exception("model error")
    result = _critique_and_refine(
        llm, "test query", "original answer", "brief text", rounds=1,
    )
    assert result == "original answer"


def test_save_trajectory_unified_deep(tmp_path: Path):
    """Unified trajectory format for deep mode should have all fields."""
    cfg = MagicMock()
    cfg.ultres_dir = tmp_path / ".ultres"
    _save_trajectory_unified(
        cfg, "test123", "build a C++ calculator",
        answer="Here is the code.",
        plan="Plan: use classes.",
        brief="Research brief.",
        mode="deep", query_type="code",
        pages_crawled=100, clusters=10,
        gaps=["error handling"],
        visited_urls=["https://example.com"],
        doc_ids=["doc_1"], code_ids=["code_1"],
        timing={"total": 60.0},
    )
    traj_path = tmp_path / ".ultres" / "trajectories" / "test123.jsonl"
    assert traj_path.exists()
    record = json.loads(traj_path.read_text("utf-8").strip())
    assert record["mode"] == "deep"
    assert record["query_type"] == "code"
    assert record["plan"] == "Plan: use classes."
    assert record["brief"] == "Research brief."
    assert record["pages_crawled"] == 100
    assert record["clusters"] == 10
    assert record["steps"] is None  # deep mode has no steps


def test_save_trajectory_unified_fast(tmp_path: Path):
    """Unified trajectory format for fast mode should have steps, no plan."""
    cfg = MagicMock()
    cfg.ultres_dir = tmp_path / ".ultres"
    _save_trajectory_unified(
        cfg, "fast123", "What is C++?",
        answer="A programming language.",
        plan="", brief="",
        mode="fast", query_type="general",
        pages_crawled=5, clusters=0,
        gaps=[],
        visited_urls=["https://example.com"],
        doc_ids=["doc_1"], code_ids=[],
        timing={},
    )
    traj_path = tmp_path / ".ultres" / "trajectories" / "fast123.jsonl"
    assert traj_path.exists()
    record = json.loads(traj_path.read_text("utf-8").strip())
    assert record["mode"] == "fast"
    assert record["plan"] is None  # fast mode has no plan
    assert record["brief"] is None
    assert record["clusters"] is None


def test_crawl_result_success_count():
    """CrawlResult.success_count should return page count."""
    result = CrawlResult()
    assert result.success_count == 0
    result.pages.append(Page(url="https://example.com", title="Test", text="content"))
    assert result.success_count == 1
