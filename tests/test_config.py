"""Tests for config load/save and env overrides."""

from __future__ import annotations

import os
from pathlib import Path

from ultres.config import UltResConfig


def test_default_config(tmp_path: Path):
    cfg = UltResConfig(project_dir=tmp_path)
    assert cfg.model.selection == "stock"
    assert cfg.search.backend == "searxng"
    assert cfg.agent.max_steps == 30
    assert cfg.ultres_dir == tmp_path / ".ultres"


def test_save_and_load(tmp_path: Path):
    cfg = UltResConfig(project_dir=tmp_path)
    cfg.agent.max_steps = 99
    cfg.search.backend = "tavily"
    cfg.save()

    assert cfg.config_path.exists()
    loaded = UltResConfig.load(tmp_path)
    assert loaded.agent.max_steps == 99
    assert loaded.search.backend == "tavily"


def test_env_overrides(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ULTRES_SEARCH_BACKEND", "brave")
    monkeypatch.setenv("ULTRES_SEARCH_API_KEY", "BSA123")
    monkeypatch.setenv("ULTRES_MAX_STEPS", "50")
    cfg = UltResConfig.load(tmp_path)
    assert cfg.search.backend == "brave"
    assert cfg.search.brave_api_key == "BSA123"
    assert cfg.agent.max_steps == 50


def test_ensure_dirs(tmp_path: Path):
    cfg = UltResConfig(project_dir=tmp_path)
    cfg.ensure_dirs()
    assert cfg.ultres_dir.exists()
    assert cfg.store_dir.exists()
    assert cfg.index_dir.exists()
    assert cfg.answers_dir.exists()
    assert cfg.adapters_dir.exists()
