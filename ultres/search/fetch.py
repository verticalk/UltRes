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
    """Pull `<pre><code>` blocks from raw HTML using a tolerant regex.

    We don't depend on a full HTML parser here to stay robust against malformed
    pages; trafilatura handles the main-text extraction separately.
    """
    blocks: list[CodeBlock] = []
    # Match <pre ...><code ...> ... </code></pre> (and bare <pre>...</pre>).
    pattern = re.compile(
        r"<pre[^>]*>(?:\s*<code[^>]*>)?(.*?)(?:</code>\s*)?</pre>",
        re.DOTALL | re.IGNORECASE,
    )
    for idx, m in enumerate(pattern.finditer(html)):
        raw = m.group(0)
        inner = m.group(1)
        # Strip nested tags inside the code block.
        text = re.sub(r"<[^>]+>", "", inner)
        text = _html_unescape(text)
        if not text.strip():
            continue
        lang_match = _LANG_RE.search(raw)
        lang = lang_match.group(1).lower() if lang_match else None
        blocks.append(
            CodeBlock(
                code_id=_code_id(url, idx, text),
                language=lang,
                content=text,
                source_url=url,
                context="",
            )
        )
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
    """Async wrapper around `fetch_page` (runs sync work in a thread)."""
    import asyncio

    return await asyncio.to_thread(fetch_page, url, timeout)
