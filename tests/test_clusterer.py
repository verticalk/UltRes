"""Tests for the clusterer module."""

from pathlib import Path
from typing import cast

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from ultres.research.clusterer import Cluster, cluster_pages, rank_code_examples
from ultres.search.base import CodeBlock, Page


class DummyEmbeddingFunction(EmbeddingFunction):
    """Deterministic hash-based bag-of-words embedding for offline tests."""

    DIM = 256

    def __call__(self, input: Documents) -> Embeddings:
        import hashlib

        results: list[list[float]] = []
        for doc in input:
            vec = [0.0] * self.DIM
            for word in doc.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.DIM] += 1.0
            results.append(vec)
        return cast(Embeddings, results)

    def name(self) -> str:
        return "dummy_cluster_test"


def test_cluster_pages_basic():
    """Pages are clustered by content similarity."""
    from ultres.memory.vector import VectorIndex
    from pathlib import Path

    # Use a persistent dir under .ultres/test to avoid Windows temp file lock issues.
    test_dir = Path(".ultres/test_cluster")
    test_dir.mkdir(parents=True, exist_ok=True)
    index = VectorIndex(test_dir, "testcluster", embedding_function=DummyEmbeddingFunction())
    pages = [
        Page(url="https://example.com/1", title="C++ Parser", text="C++ parser tutorial shunting yard algorithm"),
        Page(url="https://example.com/2", title="C++ Calculator", text="C++ calculator implementation parser"),
        Page(url="https://example.com/3", title="Cooking Recipe", text="How to make pasta carbonara recipe"),
    ]
    clusters = cluster_pages(pages, index, threshold=0.0)
    assert len(clusters) >= 1
    assert all(c.name for c in clusters)


def test_rank_code_examples():
    """Code examples are ranked by quality."""
    blocks = [
        CodeBlock(code_id="c1", language="cpp", content="int main() { return 0; }", source_url=""),
        CodeBlock(
            code_id="c2", language="cpp",
            content="// Full implementation\n// with comments\nclass Calculator {\npublic:\n    int add(int a, int b) {\n        try { return a + b; }\n        catch(...) { return 0; }\n    }\n};",
            source_url="",
        ),
        CodeBlock(code_id="c3", language="cpp", content="x = 1", source_url=""),
    ]
    ranked = rank_code_examples(blocks, max_per_cluster=2)
    assert len(ranked) == 2
    # The longer, more complete block should rank first.
    assert ranked[0].code_id == "c2"


def test_cluster_quality():
    """Cluster quality score is computed."""
    from ultres.research.clusterer import _cluster_quality
    pages = [Page(url=f"https://e.com/{i}", title=f"T{i}", text="A" * 3000) for i in range(10)]
    code = [CodeBlock(code_id=f"c{i}", language="cpp", content="code", source_url="") for i in range(5)]
    score = _cluster_quality(pages, code)
    assert 0.0 < score <= 1.0
