"""Tests for the gap detector module."""

from ultres.research.clusterer import Cluster
from ultres.research.gap_detector import coverage_map, detect_gaps
from ultres.search.base import CodeBlock, Page


class MockLLM:
    """Mock LLM for testing."""
    def __init__(self, response: str):
        self._response = response

    def create_chat_completion(self, **kwargs):
        return {"choices": [{"message": {"content": self._response}}]}


def test_coverage_map():
    """Coverage map lists all clusters."""
    clusters = [
        Cluster(cluster_id="c1", name="parsers", pages=[Page(url="u1", title="t1", text="x")], code_blocks=[]),
        Cluster(cluster_id="c2", name="UI patterns", pages=[Page(url="u2", title="t2", text="y")], code_blocks=[]),
    ]
    cm = coverage_map(clusters, "build calculator", "code")
    assert "parsers" in cm
    assert "UI patterns" in cm
    assert "Total clusters: 2" in cm


def test_detect_gaps_complete():
    """When model says COMPLETE, no gaps are returned."""
    llm = MockLLM("COMPLETE")
    clusters = [Cluster(cluster_id="c1", name="test")]
    gaps = detect_gaps(llm, clusters, "test query", "code")
    assert gaps == []


def test_detect_gaps_found():
    """When model finds gaps, they're returned as a list."""
    llm = MockLLM('{"gaps": ["error handling C++", "testing calculator", "build system cmake"]}')
    clusters = [Cluster(cluster_id="c1", name="parsers")]
    gaps = detect_gaps(llm, clusters, "build calculator", "code")
    assert len(gaps) == 3
    assert "error handling C++" in gaps


def test_detect_gaps_error():
    """When model call fails, returns empty list."""
    class ErrorLLM:
        def create_chat_completion(self, **kwargs):
            raise RuntimeError("model error")

    gaps = detect_gaps(ErrorLLM(), [], "test", "code")
    assert gaps == []
