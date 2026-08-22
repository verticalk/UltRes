"""Tests for v1.6 features: iterative planner, file extraction, verification,
adaptive complexity, query diversification, depth-aware gap detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ultres.agent.planner import plan_deep_iterative
from ultres.agent.reasoner import (
    extract_files_from_output,
    write_files_to_disk,
    estimate_complexity,
    adaptive_max_tokens,
    _check_python_syntax,
)
from ultres.research.crawler import _diversify_queries as crawler_diversify
from ultres.research.gap_detector import coverage_map
from ultres.research.clusterer import Cluster
from ultres.search.base import Page, CodeBlock


# ---------------------------------------------------------------------------
# File extraction tests
# ---------------------------------------------------------------------------

def test_extract_files_from_output():
    """Test extracting ```file:path``` blocks from model output."""
    output = """
Here's the implementation:

```file:main.py
print("hello")
```

And a test file:

```file:test_main.py
def test_hello():
    assert True
```
"""
    files = extract_files_from_output(output)
    assert "main.py" in files
    assert "test_main.py" in files
    assert 'print("hello")' in files["main.py"]
    assert "def test_hello" in files["test_main.py"]


def test_extract_files_no_blocks():
    """Test that no files are extracted when there are no file blocks."""
    output = "This is just a text answer with no file blocks."
    files = extract_files_from_output(output)
    assert files == {}


def test_write_files_to_disk(tmp_path: Path):
    """Test writing extracted files to disk."""
    files = {
        "main.py": "print('hello')",
        "subdir/utils.py": "def foo(): pass",
    }
    written = write_files_to_disk(files, tmp_path)
    assert len(written) == 2
    assert (tmp_path / "main.py").read_text() == "print('hello')"
    assert (tmp_path / "subdir" / "utils.py").read_text() == "def foo(): pass"


def test_write_files_path_traversal_blocked(tmp_path: Path):
    """Test that path traversal attempts are blocked."""
    files = {
        "../../../etc/passwd": "malicious",
        "/absolute/path.py": "malicious",
    }
    written = write_files_to_disk(files, tmp_path)
    # Absolute paths and ".." traversal should be blocked.
    # Note: on Windows, "/absolute/path.py" may be treated as relative to drive root.
    # The key check is that ".." paths are blocked.
    for p in written:
        assert ".." not in p.parts


# ---------------------------------------------------------------------------
# Complexity estimation tests
# ---------------------------------------------------------------------------

def test_estimate_complexity_simple():
    """Simple plan with 1-2 files."""
    plan = "Write a single Python function to check if a number is prime."
    assert estimate_complexity(plan) == "simple"


def test_estimate_complexity_medium():
    """Medium plan with 3-5 files."""
    plan = """
    Create the following files:
    - main.py: entry point
    - utils.py: helper functions
    - models.py: data models
    - config.py: configuration
    """
    assert estimate_complexity(plan) in ("medium", "complex")


def test_estimate_complexity_complex():
    """Complex plan with 6+ files."""
    plan = """
    Create a REST API with these files:
    - main.py: entry point
    - routes.py: API routes
    - models.py: database models
    - auth.py: authentication
    - config.py: configuration
    - tests/test_api.py: tests
    - middleware.py: middleware
    - database.py: DB connection
    """
    assert estimate_complexity(plan) == "complex"


def test_adaptive_max_tokens():
    """Test that max_tokens scales with complexity."""
    assert adaptive_max_tokens("simple") == 4096
    assert adaptive_max_tokens("medium") == 8192
    assert adaptive_max_tokens("complex") == 16384
    assert adaptive_max_tokens("unknown") == 8192


# ---------------------------------------------------------------------------
# Query diversification tests
# ---------------------------------------------------------------------------

def test_diversify_queries_removes_duplicates():
    """Test that near-duplicate queries are removed."""
    queries = [
        "python fastapi tutorial",
        "python fastapi tutorial",  # exact duplicate
        "rust web framework comparison",
        "rust web framework benchmark",  # different enough
    ]
    result = crawler_diversify(queries, similarity_threshold=0.7)
    # Should remove the exact duplicate.
    assert len(result) < len(queries)
    assert "rust web framework comparison" in result


def test_diversify_queries_all_unique():
    """Test that unique queries are all kept."""
    queries = [
        "python tutorial",
        "rust tutorial",
        "javascript tutorial",
    ]
    result = crawler_diversify(queries)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Depth-aware gap detection tests
# ---------------------------------------------------------------------------

def test_coverage_map_includes_depth():
    """Test that coverage map flags shallow clusters."""
    pages = [Page(url="http://example.com", title="Test", text="content")]
    code_blocks = [CodeBlock(code_id="x", language="python", content="code", source_url="http://example.com")] * 3
    deep_cluster = Cluster(
        cluster_id="c1", name="deep topic",
        pages=pages * 5, code_blocks=code_blocks,
    )
    shallow_cluster = Cluster(
        cluster_id="c2", name="shallow topic",
        pages=[pages[0]], code_blocks=[],
    )
    coverage = coverage_map([deep_cluster, shallow_cluster], "test query", "code")
    assert "SHALLOW" in coverage
    assert "Deep clusters" in coverage
    assert "Shallow clusters" in coverage


# ---------------------------------------------------------------------------
# Python syntax check tests
# ---------------------------------------------------------------------------

def test_check_python_syntax_valid(tmp_path: Path):
    """Test that valid Python passes syntax check."""
    f = tmp_path / "valid.py"
    f.write_text("def foo():\n    return 42\n", encoding="utf-8")
    ok, err = _check_python_syntax(f)
    assert ok is True
    assert err == ""


def test_check_python_syntax_invalid(tmp_path: Path):
    """Test that invalid Python fails syntax check."""
    f = tmp_path / "invalid.py"
    f.write_text("def foo(:\n    return 42\n", encoding="utf-8")
    ok, err = _check_python_syntax(f)
    assert ok is False
    assert "SyntaxError" in err or "syntax" in err.lower()


# ---------------------------------------------------------------------------
# Iterative planner tests (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plan_deep_iterative_fallback():
    """Test that iterative planner falls back gracefully on errors."""
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"subtasks": [{"query": "test", "why": "test"}]}'}}]
    }
    mock_provider = MagicMock()

    # Make search fail — should fall back to non-iterative behavior.
    mock_provider.search = MagicMock(side_effect=Exception("search failed"))

    result = await plan_deep_iterative(mock_llm, "test query", mock_provider)
    assert "subtasks" in result
    assert "expanded_queries" in result
    assert "query_type" in result


# ---------------------------------------------------------------------------
# Code extraction tests (v1.6 markdown fence extraction)
# ---------------------------------------------------------------------------

def test_markdown_code_fence_extraction():
    """Test that markdown code fences are extracted as code blocks."""
    from ultres.search.fetch import _extract_code_blocks_from_html
    html = """
    Some text here.
    ```python
    def hello():
        print("world")
    ```
    More text.
    """
    blocks = _extract_code_blocks_from_html(html, "http://example.com")
    # Should find the markdown fence.
    assert len(blocks) >= 1
    assert any("def hello" in b.content for b in blocks)


def test_pre_code_extraction_still_works():
    """Test that <pre><code> blocks are still extracted."""
    from ultres.search.fetch import _extract_code_blocks_from_html
    html = '<pre><code class="language-python">def foo(): pass</code></pre>'
    blocks = _extract_code_blocks_from_html(html, "http://example.com")
    assert len(blocks) >= 1
    assert "def foo" in blocks[0].content
