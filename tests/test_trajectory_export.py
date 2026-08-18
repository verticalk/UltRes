"""Tests for v1.4 trajectory export, validation, and statistics."""

from __future__ import annotations

import json
from pathlib import Path

from ultres.lora.cache import (
    export_trajectories,
    validate_trajectory,
    validate_all_trajectories,
    trajectory_stats,
)


def _make_traj(
    qid: str,
    mode: str = "deep",
    query_type: str = "code",
    answer: str = "Here is the implementation.",
    pages: int = 50,
    brief: str = "Research brief about C++ calculators.",
    plan: str = "Plan: build a calculator with classes.",
) -> dict:
    return {
        "query_id": qid,
        "query": "build me a C++ calculator",
        "mode": mode,
        "query_type": query_type,
        "answer": answer,
        "plan": plan if mode == "deep" else None,
        "brief": brief if mode == "deep" else None,
        "steps": 10 if mode == "fast" else None,
        "pages_crawled": pages,
        "clusters": 5 if mode == "deep" else None,
        "gaps": [] if mode == "deep" else None,
        "visited_urls": ["https://example.com", "https://github.com/repo"],
        "doc_ids": ["doc_1", "doc_2"],
        "code_ids": ["code_1"],
        "timing": {"total_elapsed": 120.0},
        "timestamp": 1234.5,
    }


def _save_trajs(traj_dir: Path, trajs: list[dict]) -> None:
    traj_dir.mkdir(parents=True, exist_ok=True)
    for t in trajs:
        (traj_dir / f"{t['query_id']}.jsonl").write_text(json.dumps(t) + "\n")


def test_export_trajectories_jsonl(tmp_path: Path):
    traj_dir = tmp_path / "trajectories"
    _save_trajs(traj_dir, [_make_traj("t1"), _make_traj("t2", mode="fast")])
    output = tmp_path / "train.jsonl"
    count = export_trajectories(traj_dir, output, format="jsonl")
    assert count == 2
    lines = output.read_text("utf-8").strip().split("\n")
    assert len(lines) == 2
    ex = json.loads(lines[0])
    assert "instruction" in ex
    assert "output" in ex
    assert "context" in ex
    assert ex["instruction"] == "build me a C++ calculator"
    assert ex["output"] == "Here is the implementation."
    assert "Research brief" in ex["context"]


def test_export_trajectories_json(tmp_path: Path):
    traj_dir = tmp_path / "trajectories"
    _save_trajs(traj_dir, [_make_traj("t1")])
    output = tmp_path / "train.json"
    count = export_trajectories(traj_dir, output, format="json")
    assert count == 1
    data = json.loads(output.read_text("utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["instruction"] == "build me a C++ calculator"


def test_export_skips_empty(tmp_path: Path):
    traj_dir = tmp_path / "trajectories"
    _save_trajs(traj_dir, [
        _make_traj("t1"),
        _make_traj("t2", answer=""),
        _make_traj("t3", answer="ok"),  # valid
    ])
    output = tmp_path / "train.jsonl"
    count = export_trajectories(traj_dir, output)
    assert count == 2  # t2 skipped (empty answer)


def test_validate_trajectory_valid():
    traj = _make_traj("t1")
    is_valid, issues = validate_trajectory(traj)
    assert is_valid
    assert issues == []


def test_validate_trajectory_missing_field():
    traj = _make_traj("t1")
    del traj["query_id"]
    is_valid, issues = validate_trajectory(traj)
    assert not is_valid
    assert any("query_id" in i for i in issues)


def test_validate_trajectory_short_answer():
    traj = _make_traj("t1", answer="hi")
    is_valid, issues = validate_trajectory(traj)
    assert not is_valid
    assert any("short" in i for i in issues)


def test_validate_trajectory_deep_low_pages():
    traj = _make_traj("t1", pages=5)
    is_valid, issues = validate_trajectory(traj)
    assert not is_valid
    assert any("pages" in i for i in issues)


def test_validate_trajectory_fast_no_urls():
    traj = _make_traj("t1", mode="fast")
    traj["visited_urls"] = []
    is_valid, issues = validate_trajectory(traj)
    assert not is_valid
    assert any("URLs" in i for i in issues)


def test_validate_all_trajectories(tmp_path: Path):
    traj_dir = tmp_path / "trajectories"
    _save_trajs(traj_dir, [
        _make_traj("t1"),
        _make_traj("t2", answer="hi"),  # invalid
    ])
    results = validate_all_trajectories(traj_dir)
    assert len(results) == 2
    t1_valid = [r for r in results if r[0] == "t1"][0]
    t2_valid = [r for r in results if r[0] == "t2"][0]
    assert t1_valid[1] is True
    assert t2_valid[1] is False


def test_trajectory_stats(tmp_path: Path):
    traj_dir = tmp_path / "trajectories"
    _save_trajs(traj_dir, [
        _make_traj("t1", mode="deep", query_type="code", pages=100),
        _make_traj("t2", mode="deep", query_type="general", pages=50),
        _make_traj("t3", mode="fast", query_type="code", pages=5),
    ])
    stats = trajectory_stats(traj_dir)
    assert stats["total"] == 3
    assert stats["by_mode"]["deep"] == 2
    assert stats["by_mode"]["fast"] == 1
    assert stats["by_query_type"]["code"] == 2
    assert stats["by_query_type"]["general"] == 1
    assert stats["total_pages"] == 155
    assert stats["avg_pages_deep"] == 75.0  # (100 + 50) / 2
    assert stats["total_estimated_tokens"] > 0


def test_trajectory_stats_empty(tmp_path: Path):
    stats = trajectory_stats(tmp_path / "trajectories")
    assert stats["total"] == 0
    assert stats["by_mode"]["deep"] == 0
