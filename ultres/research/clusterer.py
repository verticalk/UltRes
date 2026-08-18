"""Clusterer: vector clustering of crawled content.

Embeds all page notes + code snippets via the vector index, then clusters
them by cosine similarity. Labels clusters by top TF-IDF terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ultres.memory.vector import VectorIndex, RecallHit
from ultres.search.base import CodeBlock, Page


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """A cluster of related pages/code blocks."""
    cluster_id: str
    name: str = ""
    doc_ids: list[str] = field(default_factory=list)
    code_ids: list[str] = field(default_factory=list)
    pages: list[Page] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    summary: str = ""
    best_examples: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    centroid_text: str = ""


# ---------------------------------------------------------------------------
# TF-IDF labeling
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b", text.lower())


def _tfidf_labels(texts: list[str], top_n: int = 5) -> str:
    """Extract top TF-IDF terms from a list of texts."""
    # Simple TF-IDF: term frequency * inverse document frequency.
    from collections import Counter
    import math

    # Document frequency.
    df: Counter[str] = Counter()
    for text in texts:
        tokens = set(_tokenize(text))
        for t in tokens:
            df[t] += 1

    # Term frequency in combined corpus.
    tf: Counter[str] = Counter()
    for text in texts:
        for t in _tokenize(text):
            tf[t] += 1

    n_docs = len(texts) or 1
    scores = {}
    for term, freq in tf.most_common(100):
        idf = math.log(n_docs / (df[term] + 1))
        scores[term] = freq * idf

    top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return " ".join(t for t, _ in top)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_pages(
    pages: list[Page],
    index: VectorIndex,
    threshold: float = 0.7,
) -> list[Cluster]:
    """Cluster pages by content similarity.

    Uses the vector index to embed pages, then groups by cosine similarity.
    Pages with similarity > threshold are in the same cluster.

    Args:
        pages: List of fetched pages.
        index: Vector index for embedding.
        threshold: Cosine similarity threshold for clustering.

    Returns:
        List of clusters.
    """
    if not pages:
        return []

    # Build page texts for embedding.
    page_texts = []
    for page in pages:
        # Use first 2000 chars of text + title for embedding.
        text = (page.title + " " + page.text[:2000]).strip()
        page_texts.append(text)

    # Embed all pages via the vector index.
    # We use the index's collection to query each page against all others.
    # First, add all pages to the index.
    for i, (page, text) in enumerate(zip(pages, page_texts)):
        doc_id = f"crawl_{i:05d}"
        index.add_note(doc_id, text, url=page.url, title=page.title)

    # Now cluster: for each page, find similar pages.
    clusters: list[Cluster] = []
    assigned: set[int] = set()

    for i, text in enumerate(page_texts):
        if i in assigned:
            continue
        # Query the index for similar pages.
        hits = index.recall(text, k=min(50, len(pages)))
        # Build cluster from hits above threshold.
        cluster_members: list[int] = [i]
        assigned.add(i)
        for hit in hits:
            # Parse doc_id to get index.
            meta = hit.metadata
            hit_doc_id = meta.get("doc_id", "")
            if hit_doc_id.startswith("crawl_"):
                try:
                    idx = int(hit_doc_id.split("_")[1])
                except (ValueError, IndexError):
                    continue
                if idx not in assigned and hit.score >= threshold:
                    cluster_members.append(idx)
                    assigned.add(idx)

        # Create cluster.
        cluster_pages_list = [pages[idx] for idx in cluster_members]
        cluster_texts = [page_texts[idx] for idx in cluster_members]
        label = _tfidf_labels(cluster_texts)

        # Collect code blocks.
        all_code: list[CodeBlock] = []
        for page in cluster_pages_list:
            all_code.extend(page.code_blocks)

        cluster = Cluster(
            cluster_id=f"cluster_{i:05d}",
            name=label,
            pages=cluster_pages_list,
            code_blocks=all_code,
            quality_score=_cluster_quality(cluster_pages_list, all_code),
            centroid_text=" ".join(cluster_texts)[:500],
        )
        clusters.append(cluster)

    # Merge clusters with very similar centroids (>0.95).
    clusters = _merge_similar_clusters(clusters, threshold=0.95)

    return clusters


def _cluster_quality(pages: list[Page], code_blocks: list[CodeBlock]) -> float:
    """Score a cluster's overall quality (0.0-1.0)."""
    if not pages:
        return 0.0
    avg_text = sum(len(p.text) for p in pages) / len(pages)
    text_score = min(avg_text / 5000, 1.0) * 0.4
    code_score = min(len(code_blocks) / 10, 1.0) * 0.3
    page_score = min(len(pages) / 20, 1.0) * 0.3
    return text_score + code_score + page_score


def _merge_similar_clusters(clusters: list[Cluster], threshold: float = 0.95) -> list[Cluster]:
    """Merge clusters with very similar centroids."""
    if len(clusters) <= 1:
        return clusters

    merged: list[Cluster] = []
    used: set[int] = set()

    for i, c in enumerate(clusters):
        if i in used:
            continue
        for j in range(i + 1, len(clusters)):
            if j in used:
                continue
            other = clusters[j]
            # Simple word overlap similarity.
            words_a = set(c.centroid_text.lower().split())
            words_b = set(other.centroid_text.lower().split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            if overlap >= threshold:
                # Merge j into i.
                c.pages.extend(other.pages)
                c.code_blocks.extend(other.code_blocks)
                c.quality_score = max(c.quality_score, other.quality_score)
                used.add(j)
        merged.append(c)

    return merged


def rank_code_examples(code_blocks: list[CodeBlock], max_per_cluster: int = 5) -> list[CodeBlock]:
    """Rank code blocks by quality and return the top N.

    Quality factors: length, has comments, has error handling, completeness.
    """
    def code_quality(cb: CodeBlock) -> float:
        score = 0.0
        # Length (longer = better, up to cap).
        score += min(len(cb.content) / 2000, 1.0) * 0.3
        # Has comments.
        if "//" in cb.content or "#" in cb.content or "/*" in cb.content:
            score += 0.2
        # Has error handling.
        if any(kw in cb.content.lower() for kw in ("try", "catch", "error", "exception", "assert")):
            score += 0.2
        # Has function/class definition.
        if any(kw in cb.content for kw in ("def ", "function ", "class ", "void ", "int ", "public ", "private ")):
            score += 0.15
        # Has return statement.
        if "return" in cb.content:
            score += 0.15
        return min(score, 1.0)

    ranked = sorted(code_blocks, key=code_quality, reverse=True)
    return ranked[:max_per_cluster]
