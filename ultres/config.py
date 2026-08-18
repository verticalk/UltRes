"""UltRes configuration: Pydantic settings persisted to TOML.

Config lives at `<project>/.ultres/config.toml` (per-project) and is merged with
sensible defaults. User-level model cache lives at `~/.ultres/models/`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


def _strip_none(obj):
    """Recursively remove None values from dicts/lists (tomli_w can't serialize None)."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_DIR = Path.cwd()
USER_HOME_ULTRES = Path.home() / ".ultres"
DEFAULT_MODEL_CACHE = USER_HOME_ULTRES / "models"


class ModelConfig(BaseModel):
    """Which base model UltRes uses."""

    # v1.1: stock = Qwen2.5-7B-Instruct (general + tool calling), coder = Qwen2.5-Coder-7B.
    # v1.5 adds "ultres-base", v2 adds "ultres-base-long".
    selection: Literal["stock", "coder", "ultres-base", "ultres-base-long"] = "stock"
    quant: str = "Q4_K_M"
    # Override the auto-detected context window (tokens). None = use model default.
    n_ctx: int | None = None
    # GPU layers to offload. -1 = all, 0 = CPU only.
    n_gpu_layers: int = -1


class SearchConfig(BaseModel):
    """Search backend selection and credentials."""

    backend: Literal["searxng", "tavily", "brave"] = "searxng"
    searxng_base_url: str = "http://localhost:8080"
    # API keys are read from env vars by default but can be set in config.
    tavily_api_key: str | None = None
    brave_api_key: str | None = None
    # Number of results per search call.
    results_per_query: int = 5
    # Page fetch timeout (seconds).
    fetch_timeout: float = 30.0


class AgentConfig(BaseModel):
    """Research loop parameters."""

    max_steps: int = 30
    # Soft token budget for the hot context window before eviction kicks in.
    # v1 default leaves headroom under the 32K native window.
    hot_window_token_budget: int = 28_000
    # Temperature for the planner/loop calls.
    temperature: float = 0.7
    # Temperature for the final reasoner call (slightly lower for coherence).
    final_temperature: float = 0.4
    # Self-critique: after producing an answer, critique it against evidence
    # and re-research gaps. 0 = disabled, 1 = one critique round (default).
    enable_self_critique: bool = True
    critique_rounds: int = 1
    # Minimum pages to visit before finish is allowed.
    min_visits_before_finish: int = 2


class MemoryConfig(BaseModel):
    """Knowledge store + vector index settings."""

    # Embedding model for Chroma (ONNX, runs on CPU).
    embedding_model: str = "all-MiniLM-L6-v2"
    # Number of chunks to recall per `recall()` tool call.
    recall_k: int = 8
    # Best-effort KV-cache reuse. Falls back to re-inference if unsupported.
    kv_cache_reuse: bool = True


class UltResConfig(BaseModel):
    """Top-level UltRes configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # Paths (resolved at load time).
    project_dir: Path = DEFAULT_PROJECT_DIR
    model_cache_dir: Path = DEFAULT_MODEL_CACHE

    @property
    def ultres_dir(self) -> Path:
        return self.project_dir / ".ultres"

    @property
    def store_dir(self) -> Path:
        return self.ultres_dir / "store"

    @property
    def index_dir(self) -> Path:
        return self.ultres_dir / "index"

    @property
    def answers_dir(self) -> Path:
        return self.ultres_dir / "answers"

    @property
    def adapters_dir(self) -> Path:
        return self.ultres_dir / "adapters"

    @property
    def config_path(self) -> Path:
        return self.ultres_dir / "config.toml"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create all runtime directories if missing."""
        for d in (
            self.ultres_dir,
            self.store_dir,
            self.index_dir,
            self.answers_dir,
            self.adapters_dir,
            self.model_cache_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Write config to `<project>/.ultres/config.toml`."""
        self.ensure_dirs()
        data = self.model_dump(mode="json")
        # Strip path fields — they're resolved at load time, not persisted.
        for k in ("project_dir", "model_cache_dir"):
            data.pop(k, None)
        # tomli_w can't serialize None — drop null values recursively.
        data = _strip_none(data)
        with self.config_path.open("wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def load(cls, project_dir: Path | None = None) -> "UltResConfig":
        """Load config from `<project>/.ultres/config.toml`, or defaults if absent.

        Environment variables override config values:
        - ULTRES_SEARCH_BACKEND
        - ULTRES_SEARCH_API_KEY  (maps to the selected backend's key)
        - ULTRES_MODEL_CACHE_DIR
        - ULTRES_MODEL_SELECTION
        - ULTRES_MAX_STEPS
        """
        project_dir = project_dir or Path.cwd()
        config_path = project_dir / ".ultres" / "config.toml"

        model_cache_dir = Path(
            os.environ.get("ULTRES_MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE))
        )

        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
            cfg = cls(
                project_dir=project_dir,
                model_cache_dir=model_cache_dir,
                **data,
            )
        else:
            cfg = cls(project_dir=project_dir, model_cache_dir=model_cache_dir)

        # --- env overrides ---
        backend = os.environ.get("ULTRES_SEARCH_BACKEND")
        if backend in ("searxng", "tavily", "brave"):
            cfg.search.backend = backend  # type: ignore[assignment]

        api_key = os.environ.get("ULTRES_SEARCH_API_KEY")
        if api_key:
            if cfg.search.backend == "tavily":
                cfg.search.tavily_api_key = api_key
            elif cfg.search.backend == "brave":
                cfg.search.brave_api_key = api_key

        selection = os.environ.get("ULTRES_MODEL_SELECTION")
        if selection in ("stock", "coder", "ultres-base", "ultres-base-long"):
            cfg.model.selection = selection  # type: ignore[assignment]

        max_steps = os.environ.get("ULTRES_MAX_STEPS")
        if max_steps and max_steps.isdigit():
            cfg.agent.max_steps = int(max_steps)

        return cfg
