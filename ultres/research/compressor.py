"""Compressor: batch summarize clusters + hierarchical compression.

Takes clustered pages and produces:
1. Per-cluster summaries (batched model calls)
2. Topic summaries (combining cluster summaries)
3. A master brief (structured document representing all research)
4. Best code examples selected across all clusters
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ultres.research.clusterer import Cluster, rank_code_examples
from ultres.search.base import CodeBlock
from ultres.streaming import ResearchStreamer


# ---------------------------------------------------------------------------
# Cluster summary parsing (v1.4: more robust than splitting on "---")
# ---------------------------------------------------------------------------

_CLUSTER_HEADER_RE = re.compile(
    r"#{1,4}\s*(?:Cluster\s*)?(\d+)\s*[:\-]?\s*(.*)",
    re.IGNORECASE,
)


def _parse_cluster_summaries(text: str, expected_count: int) -> list[str]:
    """Parse model output into per-cluster summaries.

    Tries multiple formats in order:
    1. Numbered headers: "### Cluster 1: ..." or "### 1. ..."
    2. "---" separator
    3. Double-newline paragraphs
    """
    # Try numbered headers first.
    headers = list(_CLUSTER_HEADER_RE.finditer(text))
    if len(headers) >= expected_count:
        parts: list[str] = []
        for i, m in enumerate(headers):
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            parts.append(text[start:end].strip())
        return parts

    # Fall back to "---" separator.
    if "---" in text:
        parts = text.split("---")
        # Strip and filter empty.
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= expected_count:
            return parts

    # Fall back to double-newline paragraphs.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= expected_count:
        return paragraphs

    # Last resort: return what we have, padded with empty strings.
    while len(paragraphs) < expected_count:
        paragraphs.append("")
    return paragraphs[:expected_count]


# ---------------------------------------------------------------------------
# Master brief
# ---------------------------------------------------------------------------

@dataclass
class MasterBrief:
    """Compressed representation of all research."""
    text: str = ""
    code_examples: list[CodeBlock] = field(default_factory=list)
    cluster_summaries: list[str] = field(default_factory=list)
    topic_summaries: list[str] = field(default_factory=list)
    total_pages: int = 0
    total_clusters: int = 0
    total_code_blocks: int = 0


# ---------------------------------------------------------------------------
# Batch summarize
# ---------------------------------------------------------------------------

def batch_summarize(
    clusters: list[Cluster],
    llm: Any,
    max_tokens: int = 800,
    streamer: ResearchStreamer | None = None,
) -> list[Cluster]:
    """Summarize each cluster using batched model calls.

    Groups 5 clusters per model call to reduce total calls.
    Returns clusters with populated `summary` and `best_examples`.
    """
    batch_size = 10  # v1.4: increased from 5 to halve model calls
    total_batches = (len(clusters) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(clusters))
        batch = clusters[start:end]

        # Build the prompt with all clusters in the batch.
        # v1.6: Increased page text sample from 1000→2000 chars and code
        # from 500→1000 chars. We have 64K context now (not 32K), so we
        # can afford to preserve more detail.
        cluster_descriptions = []
        for i, c in enumerate(batch):
            page_texts = [p.text[:2000] for p in c.pages[:5]]
            code_texts = [cb.content[:1000] for cb in c.code_blocks[:3]]
            cluster_descriptions.append(
                f"Cluster {start+i+1}: {c.name}\n"
                f"  Pages: {len(c.pages)}, Code blocks: {len(c.code_blocks)}\n"
                f"  Sample content:\n{' '.join(page_texts[:3])[:3000]}\n"
                f"  Sample code:\n{' '.join(code_texts)[:1500]}"
            )

        prompt = (
            f"Summarize the key patterns from these {len(batch)} research clusters. "
            f"For each cluster, identify:\n"
            f"- The main topic and key findings\n"
            f"- Common patterns or approaches\n"
            f"- Best practices mentioned\n"
            f"- For code clusters: note the best code examples and what makes them good\n\n"
            f"{'─' * 60}\n\n".join(cluster_descriptions) + f"\n\n{'─' * 60}\n\n"
            f"Respond with a summary for each cluster. Use this exact format:\n"
            f"### Cluster 1:\n<summary>\n### Cluster 2:\n<summary>\n"
            f"Keep each summary to 100-200 words."
        )

        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a research summarizer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            text = resp["choices"][0]["message"]["content"].strip()
            # v1.4: Parse numbered cluster summaries (### Cluster N:) first,
            # then fall back to "---" separator, then to line-based parsing.
            parts = _parse_cluster_summaries(text, len(batch))
            for i, c in enumerate(batch):
                c.summary = parts[i].strip() if i < len(parts) else ""
                if not c.summary:
                    # Fallback: use first paragraph of first page.
                    c.summary = c.pages[0].text[:200] if c.pages else "No summary."
                # Select best code examples.
                best = rank_code_examples(c.code_blocks, max_per_cluster=3)
                c.best_examples = [cb.content[:500] for cb in best]
        except Exception as e:
            for c in batch:
                if not c.summary:
                    # Fallback: use first paragraph of first page.
                    c.summary = c.pages[0].text[:200] if c.pages else "Summary failed."
                    c.best_examples = [cb.content[:500] for cb in c.code_blocks[:3]]

        if streamer and streamer.enabled:
            streamer.stage_progress(
                "summarize", batch_idx + 1, total_batches,
                f"batch {batch_idx+1}/{total_batches}",
            )

    return clusters


# ---------------------------------------------------------------------------
# Hierarchical compression
# ---------------------------------------------------------------------------

def hierarchical_compress(
    clusters: list[Cluster],
    llm: Any,
    user_query: str,
    max_words: int = 8000,
    streamer: ResearchStreamer | None = None,
) -> MasterBrief:
    """Compress cluster summaries into a structured master brief.

    1. Group cluster summaries into topics (5 clusters per topic).
    2. Summarize each topic.
    3. Combine topic summaries into a master brief.
    4. Select best code examples across all clusters.
    """
    # --- Step 1: Topic summaries ---
    topic_size = 5
    topic_summaries: list[str] = []
    total_topics = (len(clusters) + topic_size - 1) // topic_size

    for topic_idx in range(total_topics):
        start = topic_idx * topic_size
        end = min(start + topic_size, len(clusters))
        batch = clusters[start:end]

        cluster_sums = "\n\n".join(
            f"Cluster: {c.name}\n{c.summary}" for c in batch if c.summary
        )

        prompt = (
            f"Combine these cluster summaries into a single topic summary.\n"
            f"Focus on the key findings, patterns, and best practices.\n"
            f"Keep it to 200-400 words.\n\n"
            f"Topic context: {user_query}\n\n"
            f"Cluster summaries:\n{cluster_sums[:6000]}"
        )

        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a research synthesizer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            topic_summaries.append(resp["choices"][0]["message"]["content"].strip())
        except Exception:
            topic_summaries.append(cluster_sums[:500])

        if streamer and streamer.enabled:
            streamer.stage_progress(
                "compress", topic_idx + 1, total_topics,
                f"topic {topic_idx+1}/{total_topics}",
            )

    # --- Step 2: Master brief ---
    all_topic_sums = "\n\n---\n\n".join(topic_summaries)

    prompt = (
        f"You are writing the master research brief for: {user_query}\n\n"
        f"Below are topic summaries from research on {sum(len(c.pages) for c in clusters)} pages "
        f"across {len(clusters)} clusters.\n\n"
        f"Write a structured brief with these sections:\n"
        f"## Overview\n(What the task is, key challenges)\n"
        f"## Architecture Decisions\n(Recommended approach, with rationale from research. "
        f"List each decision as a bullet with the rationale.)\n"
        f"## Key Algorithms & Patterns\n(From research, with references)\n"
        f"## Code Patterns\n(The top 5 most useful code patterns/snippets found in research. "
        f"Include the actual code inline, not just references.)\n"
        f"## Common Pitfalls\n(From research)\n"
        f"## Best Practices\n(From research)\n"
        f"## Recommended Libraries & Tools\n(From research, with specific versions if mentioned)\n\n"
        f"Keep the brief to ~{max_words} words. Be specific and cite research findings.\n\n"
        f"Topic summaries:\n{all_topic_sums[:20000]}"
    )

    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a master research synthesizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,  # v1.6: increased from 2000 for 8000-word briefs
        )
        brief_text = resp["choices"][0]["message"]["content"].strip()
    except Exception:
        brief_text = all_topic_sums[:3000]

    # --- Step 3: Select best code examples ---
    all_code: list[CodeBlock] = []
    for c in clusters:
        all_code.extend(c.code_blocks)
    best_code = rank_code_examples(all_code, max_per_cluster=15)

    return MasterBrief(
        text=brief_text,
        code_examples=best_code,
        cluster_summaries=[c.summary for c in clusters if c.summary],
        topic_summaries=topic_summaries,
        total_pages=sum(len(c.pages) for c in clusters),
        total_clusters=len(clusters),
        total_code_blocks=len(all_code),
    )
