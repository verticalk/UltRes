"""Model loader: download GGUF from HuggingFace and init llama.cpp.

Handles:
  - Resolving a ModelSpec + quant to a local file path.
  - Downloading with resume + size verification (sha256 if pinned).
  - Detecting available RAM/VRAM for sensible defaults.
  - Initializing a `llama.Llama` instance with the Qwen3 tool-call chat template.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import httpx
import psutil

from ultres.config import UltResConfig
from ultres.models.registry import ModelSpec, filename_for, get_spec


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_vram_gb() -> float:
    """Best-effort detection of GPU VRAM in GB. Returns 0.0 if no GPU."""
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0


def detect_ram_gb() -> float:
    """Available system RAM in GB."""
    return psutil.virtual_memory().total / 1e9


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_HF_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"


def _download_url(repo: str, filename: str) -> str:
    return _HF_BASE.format(repo=repo, filename=filename)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    spec: ModelSpec,
    quant: str | None = None,
    cache_dir: Path | None = None,
    verify_sha256: bool = True,
    progress: bool = True,
) -> Path:
    """Download a GGUF file to the cache dir if not already present.

    Returns the local Path. Verifies sha256 if the spec pins one.
    """
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    console = Console()
    q = quant or spec.default_quant
    filename = filename_for(spec, q)
    cache_dir = cache_dir or Path.home() / ".ultres" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        # Verify sha256 if pinned.
        if verify_sha256 and spec.sha256:
            actual = _sha256(dest)
            if actual != spec.sha256:
                console.print(
                    f"[yellow]SHA256 mismatch for {dest.name}; re-downloading.[/yellow]"
                )
                dest.unlink()
            else:
                console.print(f"[green]Model already present:[/green] {dest}")
                return dest
        else:
            console.print(f"[green]Model already present:[/green] {dest}")
            return dest

    url = _download_url(spec.hf_repo, filename)
    console.print(f"[cyan]Downloading[/cyan] {url}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        progress_ctx = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=not progress,
        )
        with progress_ctx as p, tmp.open("wb") as f:
            task = p.add_task(filename, total=total or None)
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                p.update(task, advance=len(chunk))

    tmp.replace(dest)

    if verify_sha256 and spec.sha256:
        actual = _sha256(dest)
        if actual != spec.sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {dest.name}: expected {spec.sha256}, got {actual}"
            )

    console.print(f"[green]Saved to[/green] {dest}")
    return dest


# ---------------------------------------------------------------------------
# llama.cpp init
# ---------------------------------------------------------------------------

def load_llama(
    cfg: UltResConfig,
    spec: ModelSpec | None = None,
    model_path: Path | None = None,
    **overrides: Any,
) -> "Any":
    """Initialize a llama-cpp-python `Llama` instance.

    Either pass `model_path` directly, or pass `spec` (defaults to cfg's selection).
    Context window defaults to the spec's native_ctx unless cfg.model.n_ctx is set.
    """
    from llama_cpp import Llama  # type: ignore[import-not-found]

    if model_path is None:
        spec = spec or get_spec(cfg.model.selection)
        model_path = download_model(spec, cfg.model.quant, cfg.model_cache_dir)

    n_ctx = cfg.model.n_ctx or spec.native_ctx if spec else cfg.model.n_ctx or 32_768
    n_gpu = overrides.pop("n_gpu_layers", cfg.model.n_gpu_layers)

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu,
        "n_threads": max(1, os.cpu_count() or 4),
        "verbose": False,
        # Use the chat template baked into the GGUF (Qwen3 tool-call format).
        "chat_format": "chatml",
    }
    kwargs.update(overrides)
    return Llama(**kwargs)


def resolve_spec(cfg: UltResConfig) -> ModelSpec:
    """Return the ModelSpec matching the config's model selection."""
    return get_spec(cfg.model.selection)


def ensure_model(cfg: UltResConfig) -> Path:
    """Download (if needed) and return the local path to the active model."""
    spec = resolve_spec(cfg)
    return download_model(spec, cfg.model.quant, cfg.model_cache_dir)
