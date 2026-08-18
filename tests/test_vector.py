"""Tests for the VectorIndex (Chroma in-process)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import chromadb
import pytest
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from ultres.memory.vector import VectorIndex


class DummyEmbeddingFunction(EmbeddingFunction):
    """Deterministic hash-based bag-of-words embedding for offline tests.

    No network access needed. Uses a fixed dimension (256) and hashes each
    word to a dimension, so documents with overlapping words have high
    cosine similarity.
    """

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
        return "dummy_hash_bow"


@pytest.fixture
def index(tmp_path: Path) -> VectorIndex:
    return VectorIndex(
        tmp_path / "idx",
        query_id="qtest",
        embedding_function=DummyEmbeddingFunction(),
    )


def test_add_and_recall(index: VectorIndex):
    index.add_note("doc_a", "C++ calculator with parser and evaluator", url="u", title="t")
    index.add_note("doc_b", "Rust web server with tokio", url="u2", title="t2")

    hits = index.recall("calculator parser", k=2)
    assert len(hits) >= 1
    # The C++ note should rank higher for a calculator query.
    assert hits[0].metadata.get("doc_id") == "doc_a"


def test_recall_empty_index(index: VectorIndex):
    assert index.recall("anything") == []


def test_add_summary(index: VectorIndex):
    index.add_summary("topic_x", "Topic about C++ calculators", kind="topic", name="cpp")
    hits = index.recall("C++ calculator", k=1)
    assert len(hits) == 1
    assert "topic" in hits[0].metadata.get("kind", "")


def test_add_code(index: VectorIndex):
    index.add_code("code_1", "int main() { return 0; }", language="cpp")
    hits = index.recall("main function", k=1)
    assert len(hits) == 1
    assert hits[0].metadata.get("code_id") == "code_1"


def test_count(index: VectorIndex):
    assert index.count() == 0
    index.add_note("doc_a", "text")
    assert index.count() == 1
