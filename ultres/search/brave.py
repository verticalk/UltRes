"""Brave Search API provider (paid, optional).

Set ULTRES_SEARCH_API_KEY or pass api_key. Uses the Brave Web Search endpoint.
"""

from __future__ import annotations

import httpx

from ultres.search.base import Page, SearchResult
from ultres.search.fetch import fetch_page_async


_API = "https://api.search.brave.com/res/v1/web/search"


class BraveProvider:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "BraveProvider requires an API key (set ULTRES_SEARCH_API_KEY)."
            )
        self.timeout = timeout

    async def search(self, query: str, n: int = 5) -> list[SearchResult]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": n}
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            resp = await c.get(_API, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", [])[:n]:
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("description", ""),
                    score=None,
                )
            )
        return results

    async def fetch(self, url: str) -> Page:
        return await fetch_page_async(url)
