"""Search provider interface and shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SearchResult:
    """A single hit from a search query."""

    url: str
    title: str
    snippet: str = ""
    score: float | None = None


@dataclass
class CodeBlock:
    """A verbatim code block extracted from a page.

    Code blocks are first-class objects with stable IDs and are NEVER summarized.
    """

    code_id: str
    language: str | None
    content: str
    source_url: str
    # Anchor text / surrounding heading for context.
    context: str = ""


@dataclass
class Page:
    """A fetched and extracted web page."""

    url: str
    title: str
    # Main extracted text (markdown-ish).
    text: str
    # Verbatim code blocks found in the page.
    code_blocks: list[CodeBlock] = field(default_factory=list)
    # Raw HTML kept for re-extraction if needed (optional).
    raw_html: str | None = None
    fetch_error: str | None = None


@runtime_checkable
class SearchProvider(Protocol):
    """Interface every search backend implements."""

    async def search(self, query: str, n: int = 5) -> list[SearchResult]:
        """Return up to `n` results for `query`."""
        ...

    async def fetch(self, url: str) -> Page:
        """Fetch and extract a single URL into a Page (with code blocks)."""
        ...


def get_provider(backend: str, **kwargs) -> SearchProvider:
    """Factory: instantiate a SearchProvider by name."""
    if backend == "searxng":
        from ultres.search.searxng import SearXNGProvider

        return SearXNGProvider(**kwargs)
    if backend == "tavily":
        from ultres.search.tavily import TavilyProvider

        return TavilyProvider(**kwargs)
    if backend == "brave":
        from ultres.search.brave import BraveProvider

        return BraveProvider(**kwargs)
    raise ValueError(f"Unknown search backend: {backend!r}")
