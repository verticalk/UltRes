"""Bulk crawler: parallel page fetching with link following + quality filtering.

Fetches thousands of pages from search results, follows links from each page,
and filters by quality. Designed for the v1.2 deep research pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from ultres.search.base import Page, SearchProvider
from ultres.search.fetch import fetch_page_async
from ultres.research.source_prioritizer import (
    detect_query_type,
    extract_links_from_page,
    github_to_raw,
    is_github_url,
    prioritize_urls,
    score_page,
)
from ultres.streaming import ResearchStreamer


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CrawlResult:
    """Result of a bulk crawl operation."""
    pages: list[Page] = field(default_factory=list)
    urls_fetched: list[str] = field(default_factory=list)
    urls_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    deduped: int = 0

    @property
    def success_count(self) -> int:
        return len(self.pages)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    """Fast content hash for exact dedup."""
    return hashlib.sha1(text[:2000].encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Bulk crawl
# ---------------------------------------------------------------------------

async def bulk_crawl(
    provider: SearchProvider,
    queries: list[str],
    max_pages: int = 2000,
    concurrency: int = 15,
    results_per_query: int = 8,
    crawl_depth: int = 2,
    query_type: str | None = None,
    enable_quality_filter: bool = True,
    min_quality: float = 0.15,
    streamer: ResearchStreamer | None = None,
) -> CrawlResult:
    """Bulk crawl: search all queries, fetch results, follow links.

    Args:
        provider: Search provider (SearXNG/Tavily/Brave).
        queries: List of search queries to run.
        max_pages: Maximum total pages to fetch.
        concurrency: Max concurrent page fetches.
        results_per_query: Search results to fetch per query.
        crawl_depth: Link-following hops (1 = only search results, 2 = follow links).
        query_type: "code", "general", or "mixed". Auto-detected if None.
        enable_quality_filter: Drop low-quality pages.
        min_quality: Minimum page quality score (0.0-1.0).
        streamer: Optional streamer for progress display.

    Returns:
        CrawlResult with all fetched pages.
    """
    start = time.time()
    if query_type is None and queries:
        query_type = detect_query_type(" ".join(queries[:3]))

    result = CrawlResult()
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    sem = asyncio.Semaphore(concurrency)

    # --- Phase 1: Search all queries (staggered) ---
    all_urls: list[str] = []
    batch_size = 5
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        search_tasks = [provider.search(q, n=results_per_query) for q in batch]
        try:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        except Exception:
            search_results = []
        for sr in search_results:
            if isinstance(sr, Exception):
                result.errors.append(f"search: {sr}")
                continue
            for hit in sr:
                if hit.url and hit.url not in seen_urls:
                    all_urls.append(hit.url)
                    seen_urls.add(hit.url)
        if streamer and streamer.enabled:
            streamer.stage_progress(
                "crawl", len(all_urls), max_pages,
                f"searched {min(i + batch_size, len(queries))}/{len(queries)} queries, {len(all_urls)} URLs",
            )
        # Small delay between batches to avoid rate limits.
        await asyncio.sleep(0.5)

    # Prioritize URLs.
    all_urls = prioritize_urls(all_urls, query_type)
    # Limit initial URLs.
    all_urls = all_urls[: max_pages]

    # --- Phase 2: Fetch pages in parallel ---
    async def fetch_one(url: str) -> Page | None:
        async with sem:
            # GitHub special handling: convert blob URLs to raw.
            fetch_url = url
            if is_github_url(url):
                raw = github_to_raw(url)
                if raw:
                    fetch_url = raw
            try:
                page = await fetch_page_async(fetch_url, timeout=30.0)
                page.url = url  # Keep original URL for reference.
                return page
            except Exception as e:
                result.urls_failed.append(url)
                result.errors.append(f"fetch {url}: {e}")
                return None

    # Fetch initial batch.
    fetch_tasks = [fetch_one(u) for u in all_urls if u]
    pages_to_process: list[Page] = []

    # Process in chunks to avoid creating too many tasks at once.
    chunk_size = concurrency * 2
    for i in range(0, len(fetch_tasks), chunk_size):
        chunk = fetch_tasks[i : i + chunk_size]
        results = await asyncio.gather(*chunk, return_exceptions=True)
        for r in results:
            if isinstance(r, Page):
                pages_to_process.append(r)
        if streamer and streamer.enabled:
            fetched = len(pages_to_process)
            streamer.stage_progress(
                "crawl", fetched, max_pages,
                f"fetched {fetched} pages",
            )
        if len(pages_to_process) >= max_pages:
            break

    # --- Phase 3: Quality filter + dedup ---
    for page in pages_to_process:
        if not page.text or len(page.text) < 50:
            result.urls_failed.append(page.url)
            continue
        # Content hash dedup.
        chash = _content_hash(page.text)
        if chash in seen_hashes:
            result.deduped += 1
            continue
        seen_hashes.add(chash)
        # Quality filter.
        if enable_quality_filter:
            q = score_page(page, " ".join(queries[:3]), query_type)
            if q < min_quality:
                result.urls_failed.append(page.url)
                continue
        result.pages.append(page)
        result.urls_fetched.append(page.url)

    # --- Phase 4: Follow links (if crawl_depth >= 2) ---
    if crawl_depth >= 2 and len(result.pages) < max_pages:
        link_urls: list[str] = []
        for page in result.pages:
            links = extract_links_from_page(page, query_type, max_links=3)
            for link in links:
                if link not in seen_urls:
                    link_urls.append(link)
                    seen_urls.add(link)
            if len(link_urls) + len(result.pages) >= max_pages:
                break

        # Fetch link pages.
        link_urls = link_urls[: max_pages - len(result.pages)]
        link_tasks = [fetch_one(u) for u in link_urls]
        for i in range(0, len(link_tasks), chunk_size):
            chunk = link_tasks[i : i + chunk_size]
            results = await asyncio.gather(*chunk, return_exceptions=True)
            for r in results:
                if isinstance(r, Page):
                    if r.text and len(r.text) >= 50:
                        chash = _content_hash(r.text)
                        if chash not in seen_hashes:
                            seen_hashes.add(chash)
                            if enable_quality_filter:
                                q = score_page(r, " ".join(queries[:3]), query_type)
                                if q < min_quality:
                                    continue
                            result.pages.append(r)
                            result.urls_fetched.append(r.url)
            if streamer and streamer.enabled:
                streamer.stage_progress(
                    "crawl", len(result.pages), max_pages,
                    f"link-following: {len(result.pages)} pages",
                )
            if len(result.pages) >= max_pages:
                break

    result.elapsed = time.time() - start
    return result
