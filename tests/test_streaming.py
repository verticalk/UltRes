"""Tests for the streaming module."""

from ultres.streaming import ResearchStreamer, SSEStreamer, StageInfo


def test_stage_info_elapsed():
    """StageInfo tracks elapsed time."""
    info = StageInfo(name="test", label="Test Stage")
    info.start_time = 100.0
    info.end_time = 105.0
    assert info.elapsed == 5.0
    assert info.elapsed_str == "00:05"


def test_research_streamer_stages():
    """ResearchStreamer tracks stages."""
    streamer = ResearchStreamer(enabled=False)
    streamer.stage_start("plan", "Planning")
    assert len(streamer.stages) == 1
    assert streamer.stages[0].name == "plan"
    streamer.stage_done("plan", {"queries": 50})
    assert streamer.stages[0].done
    assert streamer.stages[0].stats["queries"] == 50


def test_research_streamer_summary():
    """Streamer summary includes all stages."""
    streamer = ResearchStreamer(enabled=False)
    streamer.stage_start("plan", "Planning")
    streamer.stage_done("plan", {"queries": 50})
    streamer.stage_start("crawl", "Crawling")
    streamer.stage_done("crawl", {"pages": 100})
    summary = streamer.summary()
    assert summary["total_elapsed"] >= 0
    assert len(summary["stages"]) == 2
    assert summary["stages"][0]["name"] == "plan"
    assert summary["stages"][1]["stats"]["pages"] == 100


def test_sse_streamer_events():
    """SSEStreamer formats events correctly."""
    sse = SSEStreamer()
    event = sse.stage_start("plan", "Planning")
    assert "event: stage_start" in event
    assert "Planning" in event
    token_event = sse.token("hello")
    assert "event: token" in token_event
    assert "hello" in token_event
