"""Tests for the source prioritizer module."""

from ultres.research.source_prioritizer import (
    detect_query_type,
    extract_links_from_page,
    get_source_priority,
    github_to_raw,
    is_github_url,
    prioritize_urls,
    score_page,
)
from ultres.search.base import CodeBlock, Page


def test_detect_query_type_code():
    """Code-related queries are detected as 'code'."""
    assert detect_query_type("build me a complex C++ calculator app") == "code"
    assert detect_query_type("implement a Python web scraper") == "code"
    assert detect_query_type("write a Rust async runtime") == "code"


def test_detect_query_type_general():
    """Non-code queries are detected as 'general'."""
    assert detect_query_type("what is the history of Rome") == "general"
    assert detect_query_type("explain quantum mechanics") == "general"


def test_get_source_priority():
    """Code queries prioritize code domains."""
    code_priority = get_source_priority("code")
    assert "github.com" in code_priority
    assert "stackoverflow.com" in code_priority
    general_priority = get_source_priority("general")
    assert "wikipedia.org" in general_priority


def test_prioritize_urls():
    """Priority domains come first."""
    urls = [
        "https://random-blog.com/post",
        "https://github.com/user/repo",
        "https://example.com/page",
        "https://stackoverflow.com/questions/123",
    ]
    result = prioritize_urls(urls, "code")
    # GitHub and SO should come before random blog and example.com.
    github_idx = result.index("https://github.com/user/repo")
    so_idx = result.index("https://stackoverflow.com/questions/123")
    blog_idx = result.index("https://random-blog.com/post")
    assert github_idx < blog_idx
    assert so_idx < blog_idx


def test_github_to_raw():
    """GitHub blob URLs are converted to raw URLs."""
    raw = github_to_raw("https://github.com/user/repo/blob/main/src/main.cpp")
    assert raw == "https://raw.githubusercontent.com/user/repo/main/src/main.cpp"
    # Non-blob URLs return None.
    assert github_to_raw("https://github.com/user/repo") is None


def test_is_github_url():
    assert is_github_url("https://github.com/user/repo") is True
    assert is_github_url("https://example.com") is False


def test_score_page_quality():
    """Page scoring rewards content + code blocks."""
    page_good = Page(
        url="https://example.com",
        title="C++ Calculator Tutorial",
        text="A" * 6000 + " calculator parser implementation",
        code_blocks=[CodeBlock(code_id="c1", language="cpp", content="int main(){}", source_url="")],
    )
    score_good = score_page(page_good, "C++ calculator", "code")
    assert score_good > 0.3

    page_bad = Page(url="https://example.com", title="", text="short")
    score_bad = score_page(page_bad, "C++ calculator", "code")
    assert score_bad < 0.2


def test_extract_links_from_page():
    """Links are extracted from page HTML."""
    page = Page(
        url="https://example.com/article",
        title="Article",
        text="content",
        raw_html='<a href="https://github.com/user/repo">Code</a>'
                 '<a href="/about">About</a>'
                 '<a href="https://random.com">Random</a>',
    )
    links = extract_links_from_page(page, "code", max_links=3)
    assert len(links) > 0
    assert "https://github.com/user/repo" in links
    # Relative URL resolved.
    assert any("example.com/about" in l for l in links)
