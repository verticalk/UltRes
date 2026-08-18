"""Tests for the compressor module."""

from ultres.research.clusterer import Cluster
from ultres.research.compressor import batch_summarize, hierarchical_compress, MasterBrief
from ultres.search.base import CodeBlock, Page


class MockLLM:
    """Mock LLM for testing."""
    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    def create_chat_completion(self, **kwargs):
        self.call_count += 1
        return {"choices": [{"message": {"content": self._response}}]}


def test_batch_summarize():
    """Batch summarize populates cluster summaries."""
    clusters = [
        Cluster(
            cluster_id="c1", name="parsers",
            pages=[Page(url="u1", title="Parser", text="Shunting yard algorithm for parsing")],
            code_blocks=[CodeBlock(code_id="cb1", language="cpp", content="int parse(){}", source_url="")],
        ),
        Cluster(
            cluster_id="c2", name="UI",
            pages=[Page(url="u2", title="UI", text="Qt calculator UI tutorial")],
            code_blocks=[],
        ),
    ]
    llm = MockLLM("Summary: This cluster covers parser algorithms.\n---\nSummary: This cluster covers UI design.")
    result = batch_summarize(clusters, llm, streamer=None)
    assert all(c.summary for c in result)
    assert llm.call_count >= 1


def test_hierarchical_compress():
    """Hierarchical compression produces a master brief."""
    clusters = [
        Cluster(cluster_id="c1", name="parsers", summary="Parser algorithms: shunting yard, recursive descent.",
                pages=[Page(url="u1", title="P", text="x")],
                code_blocks=[CodeBlock(code_id="cb1", language="cpp", content="int main(){return 0;}", source_url="")]),
        Cluster(cluster_id="c2", name="UI", summary="Qt UI framework for calculator.",
                pages=[Page(url="u2", title="U", text="y")],
                code_blocks=[]),
    ]
    llm = MockLLM("## Overview\nCalculator needs parser + UI.\n## Architecture\nUse shunting yard + Qt.")
    brief = hierarchical_compress(clusters, llm, "build calculator", streamer=None)
    assert isinstance(brief, MasterBrief)
    assert "Overview" in brief.text or len(brief.text) > 0
    assert len(brief.code_examples) > 0
    assert brief.total_clusters == 2


def test_batch_summarize_error_fallback():
    """When model fails, fallback summary is used."""
    class ErrorLLM:
        def create_chat_completion(self, **kwargs):
            raise RuntimeError("model error")

    clusters = [
        Cluster(
            cluster_id="c1", name="test",
            pages=[Page(url="u1", title="Test", text="Some content here for fallback")],
            code_blocks=[],
        ),
    ]
    result = batch_summarize(clusters, ErrorLLM(), streamer=None)
    assert result[0].summary  # Should have fallback summary.
