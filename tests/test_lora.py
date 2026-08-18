"""Tests for the LoRA cache stub."""

from __future__ import annotations

import json
from pathlib import Path

from ultres.lora.cache import (
    find_matching_adapter,
    list_adapters,
    list_trajectories,
    lora_path_for_query,
    topic_similarity,
    trajectory_count,
)


def test_topic_similarity():
    assert topic_similarity("c++ calculator", "c++ calculator") == 1.0
    assert topic_similarity("c++ calculator", "rust web server") < 0.5
    assert topic_similarity("", "anything") == 0.0


def test_list_adapters_empty(tmp_path: Path):
    assert list_adapters(tmp_path) == []


def test_list_adapters_with_files(tmp_path: Path):
    # Create a fake adapter + metadata.
    adapter_path = tmp_path / "cpp-calculator.safetensors"
    adapter_path.write_bytes(b"fake")
    meta = {"name": "cpp-calc", "topic": "c++ calculator", "base_model": "stock", "created_at": 1.0}
    (tmp_path / "cpp-calculator.json").write_text(json.dumps(meta))

    adapters = list_adapters(tmp_path)
    assert len(adapters) == 1
    assert adapters[0].topic == "c++ calculator"


def test_find_matching_adapter(tmp_path: Path):
    adapter_path = tmp_path / "cpp.safetensors"
    adapter_path.write_bytes(b"fake")
    meta = {"topic": "c++ calculator", "base_model": "stock", "created_at": 1.0}
    (tmp_path / "cpp.json").write_text(json.dumps(meta))

    match = find_matching_adapter(tmp_path, "c++ calculator")
    assert match is not None
    assert match.topic == "c++ calculator"

    no_match = find_matching_adapter(tmp_path, "rust web server")
    assert no_match is None


def test_lora_path_for_query(tmp_path: Path):
    adapter_path = tmp_path / "cpp.safetensors"
    adapter_path.write_bytes(b"fake")
    meta = {"topic": "c++ calculator", "base_model": "stock", "created_at": 1.0}
    (tmp_path / "cpp.json").write_text(json.dumps(meta))

    path = lora_path_for_query(tmp_path, "c++ calculator")
    assert path is not None
    assert path.endswith("cpp.safetensors")

    assert lora_path_for_query(tmp_path, "unrelated topic") is None


def test_trajectory_save_and_list(tmp_path: Path):
    """Trajectories are saved as JSONL and can be listed for v1.5 QLoRA training."""
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    record = {
        "query_id": "abc123",
        "query": "What is C++?",
        "answer": "A programming language.",
        "steps": 5,
        "visited_urls": ["https://example.com"],
        "doc_ids": ["doc_1"],
        "code_ids": [],
        "timestamp": 1234.5,
    }
    (traj_dir / "abc123.jsonl").write_text(json.dumps(record) + "\n")

    trajs = list_trajectories(traj_dir)
    assert len(trajs) == 1
    assert trajs[0]["query"] == "What is C++?"
    assert trajs[0]["answer"] == "A programming language."
    assert trajectory_count(traj_dir) == 1
    assert trajectory_count(tmp_path) == 0
