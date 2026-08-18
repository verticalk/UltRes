"""Tavily search provider (paid API, optimized for LLM agents).

Set ULTRES_SEARCH_API_KEY or pass api_key. Returns clean markdown content
directly from the API, so `fetch()` still uses the generic page fetcher for
URLs not covered by a Tavily result.
"""

from __future__ import annotations

import httpx

from ultres.search.base import Page, SearchResult
from ultres.search.fetch import fetch_page_async


_API = "https://api.tavily.com"


class TavilyProvider:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "TavilyProvider requires an API key (set ULTRES_SEARCH_API_KEY)."
            )
        self.timeout = timeout

    async def search(self, query: str, n: int = 5) -> list[SearchResult]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": n,
            "search_depth": "advanced",
            "include_answer": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            resp = await c.post(f"{_API}/search", json=payload)
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
