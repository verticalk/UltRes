"""Source prioritizer: query type detection + URL ranking + page scoring.

Detects whether a query is code-related, general research, or mixed,
then prioritizes URLs from authoritative sources for that query type.
Also scores individual pages for quality filtering.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from ultres.search.base import Page


# ---------------------------------------------------------------------------
# Query type detection
# ---------------------------------------------------------------------------

_CODE_KEYWORDS = {
    "c++", "cpp", "python", "java", "javascript", "rust", "go", "golang",
    "typescript", "c#", "csharp", "ruby", "php", "swift", "kotlin", "scala",
    "html", "css", "sql", "bash", "shell", "powershell", "assembly", "asm",
    "code", "function", "class", "method", "algorithm", "compiler", "parser",
    "debug", "compile", "build", "library", "framework", "api", "endpoint",
    "program", "programming", "developer", "implement", "implementation",
    "refactor", "syntax", "runtime", "binary", "executable", "script",
    "shader", "opengl", "vulkan", "directx", "webgl", "cuda", "kernel",
    "concurrency", "multithreading", "async", "await", "callback",
}

_CODE_DOMAINS = {
    "github.com", "stackoverflow.com", "cppreference.com", "geeksforgeeks.org",
    "docs.rs", "dev.to", "codeproject.com", "coderbyte.com", "hackerrank.com",
    "leetcode.com", "replit.com", "gist.github.com", "gitlab.com",
    "npmjs.com", "pypi.org", "crates.io", "mvnrepository.com",
    "developer.mozilla.org", "w3schools.com", "tutorialspoint.com",
    "programiz.com", "javatpoint.com", "baeldung.com", "digitalocean.com",
}

_GENERAL_DOMAINS = {
    "wikipedia.org", "arxiv.org", "scholar.google.com", "nature.com",
    "science.org", "ieee.org", "acm.org", "springer.com", "sciencedirect.com",
    "britannica.com", "history.com", "nationalgeographic.com",
}


def detect_query_type(query: str) -> str:
    """Detect if a query is code-related, general research, or mixed.

    Returns "code", "general", or "mixed".
    """
    query_lower = query.lower()
    # Check for code keywords.
    words = set(re.findall(r"\b\w+\b", query_lower))
    code_hits = words & _CODE_KEYWORDS
    if len(code_hits) >= 2:
        return "code"
    if len(code_hits) == 1:
        # Check if the query mentions building/creating something.
        if any(w in query_lower for w in ("build", "create", "make", "implement", "write", "develop")):
            return "code"
        return "mixed"
    return "general"


def get_source_priority(query_type: str) -> list[str]:
    """Return ordered list of priority domains for a query type."""
    if query_type == "code":
        return list(_CODE_DOMAINS)
    if query_type == "general":
        return list(_GENERAL_DOMAINS)
    # mixed
    return list(_CODE_DOMAINS) + list(_GENERAL_DOMAINS)


# ---------------------------------------------------------------------------
# URL prioritization
# ---------------------------------------------------------------------------

def prioritize_urls(urls: list[str], query_type: str) -> list[str]:
    """Reorder URLs so priority domains come first.

    Within each priority tier, original order is preserved.
    """
    priority = get_source_priority(query_type)
    priority_set = set(priority)

    # Tier 1: priority domains (in priority order).
    tiered: dict[str, list[str]] = {d: [] for d in priority}
    rest: list[str] = []

    for url in urls:
        domain = _domain_of(url)
        matched = False
        for pd in priority:
            if pd in domain:
                tiered[pd].append(url)
                matched = True
                break
        if not matched:
            rest.append(url)

    result: list[str] = []
    for pd in priority:
        result.extend(tiered[pd])
    result.extend(rest)
    return result


def _domain_of(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Link extraction from pages
# ---------------------------------------------------------------------------

_LINK_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def extract_links_from_page(page: Page, query_type: str, max_links: int = 3) -> list[str]:
    """Extract relevant links from a page's raw HTML.

    Filters by priority domains and same-domain. Limits to max_links.
    Skips non-http links, anchors, and javascript: URIs.
    """
    if not page.raw_html:
        return []
    priority = get_source_priority(query_type)
    page_domain = _domain_of(page.url)

    candidates: list[str] = []
    seen: set[str] = set()
    for m in _LINK_RE.finditer(page.raw_html):
        href = m.group(1)
        # Skip anchors, javascript, mailto, etc.
        if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        # Resolve relative URLs.
        if href.startswith("/"):
            href = f"https://{page_domain}{href}"
        elif not href.startswith("http"):
            continue
        # Skip if already seen or same as page URL.
        if href in seen or href == page.url:
            continue
        seen.add(href)
        candidates.append(href)

    # Prioritize: priority domains first, then same-domain, then rest.
    priority_set = set(priority)
    priority_links = [u for u in candidates if any(p in _domain_of(u) for p in priority)]
    same_domain = [u for u in candidates if page_domain in _domain_of(u) and u not in priority_links]
    other = [u for u in candidates if u not in priority_links and u not in same_domain]

    return (priority_links + same_domain + other)[:max_links]


# ---------------------------------------------------------------------------
# GitHub special handling
# ---------------------------------------------------------------------------

def is_github_url(url: str) -> bool:
    return "github.com" in _domain_of(url)


def github_to_raw(url: str) -> str | None:
    """Convert a GitHub file URL to a raw.githubusercontent.com URL.

    Returns None if the URL is not a GitHub file URL (e.g. repo root, issues).
    """
    # Pattern: github.com/{user}/{repo}/blob/{branch}/{path}
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)",
        url,
    )
    if m:
        user, repo, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{path}"
    return None


# ---------------------------------------------------------------------------
# Page quality scoring
# ---------------------------------------------------------------------------

def score_page(page: Page, query: str, query_type: str) -> float:
    """Score a page's quality (0.0 to 1.0).

    Factors:
    - Text length (longer = better, up to a cap)
    - Code block count (more = better for code queries)
    - v1.6: Code density (ratio of code chars to text chars)
    - Has title
    - No fetch error
    - Keyword overlap with query
    """
    score = 0.0

    # Text length (0-0.3).
    text_len = len(page.text)
    if text_len > 5000:
        score += 0.3
    elif text_len > 1000:
        score += 0.2
    elif text_len > 200:
        score += 0.1
    elif text_len < 100:
        return 0.0  # Too short, reject immediately.

    # Code blocks (0-0.3 for code queries, 0-0.1 for general).
    n_code = len(page.code_blocks)
    if query_type == "code":
        if n_code >= 3:
            score += 0.3
        elif n_code >= 1:
            score += 0.15
        # v1.6: Code density bonus — pages with high code-to-text ratio
        # are more valuable for code queries.
        total_code_chars = sum(len(cb.content) for cb in page.code_blocks)
        if text_len > 0:
            code_ratio = total_code_chars / text_len
            if code_ratio > 0.3:
                score += 0.1  # High code density bonus.
            elif code_ratio > 0.1:
                score += 0.05
    else:
        if n_code >= 1:
            score += 0.05

    # Has title (0-0.1).
    if page.title and page.title != page.url:
        score += 0.1

    # No fetch error (0-0.1).
    if not page.fetch_error:
        score += 0.1

    # Keyword overlap with query (0-0.2).
    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    text_words = set(re.findall(r"\b\w+\b", page.text.lower()[:5000]))
    if query_words:
        overlap = len(query_words & text_words) / len(query_words)
        score += overlap * 0.2

    return min(score, 1.0)
