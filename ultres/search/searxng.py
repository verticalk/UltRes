"""SearXNG search provider (local, free, no API key).

Expects a SearXNG instance running at `base_url` (default http://localhost:8080).
Quick start:
    docker run -d --name searxng -p 8080:8080 \
        -e SEARXNG_BASE_URL=http://localhost:8080 \
        searxng/searxng

The instance must have the `json` format enabled in settings.yml
(`search.formats: [html, json]`).
"""

from __future__ import annotations

import httpx

from ultres.search.base import Page, SearchResult
from ultres.search.fetch import fetch_page_async


class SearXNGProvider:
    """Search provider backed by a local SearXNG meta-engine instance."""

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def search(self, query: str, n: int = 5) -> list[SearchResult]:
        params = {"q": query, "format": "json", "safesearch": 1}
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            resp = await c.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("results", [])[:n]:
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    score=item.get("score"),
                )
            )
        return results

    async def fetch(self, url: str) -> Page:
        return await fetch_page_async(url)
