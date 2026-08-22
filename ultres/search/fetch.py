"""Page fetcher: Playwright render + trafilatura extraction + code block capture.

Code blocks (`<pre><code>`) are extracted as first-class `CodeBlock` objects with
stable IDs and are NEVER summarized downstream.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import httpx
import trafilatura
from tenacity import retry, stop_after_attempt, wait_exponential

from ultres.search.base import CodeBlock, Page


# Lightweight regex to pull language hints from common class names.
_LANG_RE = re.compile(r"(?:language-|lang-|highlight-)([\w+-]+)", re.I)


def _code_id(url: str, idx: int, content: str) -> str:
    h = hashlib.sha1(f"{url}#{idx}\n{content[:64]}".encode()).hexdigest()[:12]
    return f"code_{h}"


def _extract_code_blocks_from_html(html: str, url: str) -> list[CodeBlock]:
    """Pull code blocks from raw HTML using tolerant regexes.

    v1.6: Now extracts from multiple sources:
    1. <pre><code> blocks (original)
    2. Bare <pre> blocks without <code> wrapper
    3. Markdown code fences (```language ... ```) — for pages served as
       markdown or rendered from it (GitHub READMEs, dev.to, etc.)
    4. Stack Overflow answer cells (<div class="answercell"> code blocks)
    """
    blocks: list[CodeBlock] = []
    seen_texts: set[str] = set()

    def _add_block(idx: int, text: str, lang: str | None, raw: str) -> None:
        text = text.strip()
        if not text or len(text) < 10:
            return
        # Dedup by first 100 chars (avoid extracting the same code twice
        # from <pre><code> and markdown fence).
        key = text[:100]
        if key in seen_texts:
            return
        seen_texts.add(key)
        blocks.append(
            CodeBlock(
                code_id=_code_id(url, idx, text),
                language=lang,
                content=text,
                source_url=url,
                context="",
            )
        )

    # 1. Match <pre ...><code ...> ... </code></pre> (and bare <pre>...</pre>).
    pattern = re.compile(
        r"<pre[^>]*>(?:\s*<code[^>]*>)?(.*?)(?:</code>\s*)?</pre>",
        re.DOTALL | re.IGNORECASE,
    )
    for idx, m in enumerate(pattern.finditer(html)):
        raw = m.group(0)
        inner = m.group(1)
        text = re.sub(r"<[^>]+>", "", inner)
        text = _html_unescape(text)
        lang_match = _LANG_RE.search(raw)
        lang = lang_match.group(1).lower() if lang_match else None
        _add_block(idx, text, lang, raw)

    # 2. v1.6: Extract markdown code fences from the extracted text.
    # trafilatura outputs markdown, so code fences may be in the text.
    # Also check raw HTML for markdown fences (some pages serve .md).
    md_fence_re = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for idx, m in enumerate(md_fence_re.finditer(html)):
        lang = m.group(1).lower() if m.group(1) else None
        text = _html_unescape(m.group(2).strip())
        _add_block(idx + 1000, text, lang, m.group(0))

    # 3. v1.6: Stack Overflow-specific extraction.
    # SO answer code is in <div class="answercell"> ... <pre><code> blocks.
    # The general <pre><code> extractor above already catches these,
    # but SO sometimes uses <code> without <pre> for inline code in answers.
    if "stackoverflow.com" in url or "stackexchange.com" in url:
        # Extract from <code> tags within answer cells (not just <pre>).
        so_pattern = re.compile(
            r'<div[^>]*class="[^"]*answercell[^"]*"[^>]*>.*?<code[^>]*>(.*?)</code>',
            re.DOTALL | re.IGNORECASE,
        )
        for idx, m in enumerate(so_pattern.finditer(html)):
            text = re.sub(r"<[^>]+>", "", m.group(1))
            text = _html_unescape(text).strip()
            if len(text) > 20:  # Skip inline code snippets.
                _add_block(idx + 2000, text, None, m.group(0))

    return blocks


def _html_unescape(s: str) -> str:
    import html as html_mod

    return html_mod.unescape(s)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch_html_playwright(url: str, timeout: float) -> str:
    """Render a URL with Playwright (headless Chromium) and return HTML."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, timeout=int(timeout * 1000), wait_until="domcontentloaded")
            # Give lazy content a moment.
            page.wait_for_timeout(800)
            html = page.content()
        finally:
            browser.close()
    return html


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch_html_httpx(url: str, timeout: float) -> str:
    """Fetch HTML with a plain HTTP client (faster, no JS)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text


def fetch_page(
    url: str,
    timeout: float = 30.0,
    use_playwright: bool = True,
) -> Page:
    """Fetch and extract a page. Tries Playwright first (for JS-heavy sites),
    falls back to httpx. Always extracts code blocks from the raw HTML.
    """
    html: str | None = None
    error: str | None = None

    if use_playwright:
        try:
            html = _fetch_html_playwright(url, timeout)
        except Exception as e:
            error = f"playwright: {e}"

    if html is None:
        try:
            html = _fetch_html_httpx(url, timeout)
            error = None
        except Exception as e:
            error = f"httpx: {e}"
            return Page(url=url, title="", text="", fetch_error=error)

    code_blocks = _extract_code_blocks_from_html(html, url)

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        output_format="markdown",
    )
    text = extracted or ""

    # trafilatura can also give us the title via bare_extraction.
    meta = trafilatura.bare_extraction(html, include_comments=False)
    # In newer trafilatura versions, bare_extraction returns a Document object
    # (namedtuple) rather than a dict. Handle both cases.
    if meta is not None:
        if isinstance(meta, dict):
            title = meta.get("title") or url
        else:
            title = getattr(meta, "title", None) or url
    else:
        title = url

    return Page(
        url=url,
        title=title or url,
        text=text,
        code_blocks=code_blocks,
        raw_html=html,
        fetch_error=error,
    )


async def fetch_page_async(url: str, timeout: float = 30.0) -> Page:
    """Async wrapper around `fetch_page` (runs sync work in a thread).

    Uses Playwright-first (original behavior) — for the fast agentic loop
    where single-page quality matters more than speed.
    """
    import asyncio

    return await asyncio.to_thread(fetch_page, url, timeout)


# Thresholds for deciding if httpx response is "good enough" or needs
# Playwright fallback. Pages shorter than this or with very few words
# likely need JS rendering.
_MIN_HTTPX_TEXT_CHARS = 500
_MIN_HTTPX_WORD_COUNT = 50

# Domains known to require JS rendering (httpx won't get usable content).
_JS_HEAVY_DOMAINS = {
    "twitter.com", "x.com", "reactjs.org", "vuejs.org",
    "angular.io", "svelte.dev",
}


def _needs_playwright_fallback(html: str, url: str) -> bool:
    """Check if the httpx response is too thin and needs Playwright."""
    if not html or len(html) < _MIN_HTTPX_TEXT_CHARS:
        return True
    # Check word count of extracted text.
    text = trafilatura.extract(
        html, include_comments=False, favor_recall=True,
        output_format="markdown",
    ) or ""
    if len(text.split()) < _MIN_HTTPX_WORD_COUNT:
        return True
    # Check known JS-heavy domains.
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower()
    if any(d in domain for d in _JS_HEAVY_DOMAINS):
        return True
    return False


def fetch_page_httpx_first(
    url: str,
    timeout: float = 30.0,
) -> Page:
    """Fetch a page trying httpx first, falling back to Playwright.

    For bulk crawling (2000+ pages), this avoids launching Chromium for
    every page. httpx is ~10x faster and uses no browser memory. Only
    falls back to Playwright if httpx returns empty/JS-heavy content.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Page with extracted content + code blocks.
    """
    html: str | None = None
    error: str | None = None

    # Try httpx first.
    try:
        html = _fetch_html_httpx(url, timeout)
        error = None
    except Exception as e:
        error = f"httpx: {e}"
        html = None

    # Check if httpx response is good enough.
    if html and not _needs_playwright_fallback(html, url):
        # httpx response is good — extract content.
        code_blocks = _extract_code_blocks_from_html(html, url)
        extracted = trafilatura.extract(
            html, include_comments=False, include_tables=True,
            favor_recall=True, output_format="markdown",
        ) or ""
        meta = trafilatura.bare_extraction(html, include_comments=False)
        if meta is not None:
            if isinstance(meta, dict):
                title = meta.get("title") or url
            else:
                title = getattr(meta, "title", None) or url
        else:
            title = url
        return Page(
            url=url, title=title or url, text=extracted,
            code_blocks=code_blocks, raw_html=html, fetch_error=None,
        )

    # Fallback to Playwright.
    try:
        html = _fetch_html_playwright(url, timeout)
        error = None
    except Exception as e:
        if error:
            error = f"{error}; playwright: {e}"
        else:
            error = f"playwright: {e}"
        if html is None:
            return Page(url=url, title="", text="", fetch_error=error)

    # Extract from Playwright HTML.
    code_blocks = _extract_code_blocks_from_html(html, url)
    extracted = trafilatura.extract(
        html, include_comments=False, include_tables=True,
        favor_recall=True, output_format="markdown",
    ) or ""
    meta = trafilatura.bare_extraction(html, include_comments=False)
    if meta is not None:
        if isinstance(meta, dict):
            title = meta.get("title") or url
        else:
            title = getattr(meta, "title", None) or url
    else:
        title = url
    return Page(
        url=url, title=title or url, text=extracted,
        code_blocks=code_blocks, raw_html=html, fetch_error=error,
    )


async def fetch_page_httpx_first_async(url: str, timeout: float = 30.0) -> Page:
    """Async wrapper around `fetch_page_httpx_first`."""
    import asyncio

    return await asyncio.to_thread(fetch_page_httpx_first, url, timeout)
