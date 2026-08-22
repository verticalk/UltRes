"""Model loader: download GGUF from HuggingFace and init llama.cpp.

Handles:
  - Resolving a ModelSpec + quant to a local file path.
  - Downloading with resume + size verification (sha256 if pinned).
  - Multi-part GGUF downloads (some HuggingFace repos split large files).
  - Detecting available RAM/VRAM for sensible defaults.
  - Initializing a `llama.Llama` instance with auto-detected chat format.
  - Adding CUDA DLL paths for GPU acceleration on Windows.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import psutil

from ultres.config import UltResConfig
from ultres.models.registry import ModelSpec, filename_for, get_spec


# ---------------------------------------------------------------------------
# CUDA DLL path setup (Windows + CUDA wheel)
# ---------------------------------------------------------------------------

_cuda_paths_added = False


def _add_cuda_dll_paths() -> None:
    """Add CUDA runtime DLL paths so llama.dll can find cublas/cudnn/etc.

    Needed when using the prebuilt CUDA wheel on Windows — the CUDA runtime
    DLLs are installed via pip packages (nvidia-cublas-cu12, etc.) but Windows
    doesn't automatically search those paths. We prepend them to PATH.
    """
    global _cuda_paths_added
    if _cuda_paths_added or sys.platform != "win32":
        return
    # Find nvidia CUDA DLLs from pip packages.
    try:
        import site

        cuda_paths: list[str] = []
        for site_dir in site.getsitepackages():
            nvidia_base = Path(site_dir) / "nvidia"
            if not nvidia_base.exists():
                continue
            for root, dirs, files in os.walk(nvidia_base):
                if any(f.endswith(".dll") for f in files):
                    cuda_paths.append(root)
                    try:
                        os.add_dll_directory(root)
                    except Exception:
                        pass
        # Also prepend to PATH — this is the reliable approach on Windows.
        if cuda_paths:
            os.environ["PATH"] = ";".join(cuda_paths) + ";" + os.environ.get("PATH", "")
        _cuda_paths_added = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_vram_gb() -> float:
    """Best-effort detection of GPU VRAM in GB. Returns 0.0 if no GPU.

    Tries torch first (most accurate), then falls back to nvidia-smi parsing
    (no torch dependency needed).
    """
    # Try torch first.
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    # Fallback: parse nvidia-smi output.
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().splitlines()[0]) / 1024.0
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

    Handles multi-part GGUF files (some HuggingFace repos split large files).
    Returns the local Path to the first part (llama.cpp loads splits automatically).
    Verifies sha256 if the spec pins one.
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

    # Check if the single file exists.
    if dest.exists() and dest.stat().st_size > 0:
        if verify_sha256 and spec.sha256:
            actual = _sha256(dest)
            if actual != spec.sha256:
                console.print(f"[yellow]SHA256 mismatch for {dest.name}; re-downloading.[/yellow]")
                dest.unlink()
            else:
                console.print(f"[green]Model already present:[/green] {dest}")
                return dest
        else:
            console.print(f"[green]Model already present:[/green] {dest}")
            return dest

    # Try downloading as a single file first.
    url = _download_url(spec.hf_repo, filename)
    try:
        _download_single(url, dest, console, progress)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise
        # 404 — try multi-part pattern: base-00001-of-00002.gguf, etc.
        console.print(f"[yellow]Single file not found; trying multi-part GGUF...[/yellow]")
        dest = _download_multipart(spec, filename, cache_dir, console, progress)

    if verify_sha256 and spec.sha256:
        actual = _sha256(dest)
        if actual != spec.sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {dest.name}: expected {spec.sha256}, got {actual}"
            )

    console.print(f"[green]Saved to[/green] {dest}")
    return dest


def _download_single(url: str, dest: Path, console: Console, progress: bool) -> None:
    """Download a single file with progress bar."""
    from rich.progress import (
        BarColumn, DownloadColumn, Progress,
        TextColumn, TimeRemainingColumn, TransferSpeedColumn,
    )

    console.print(f"[cyan]Downloading[/cyan] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)

    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        progress_ctx = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(), DownloadColumn(), TransferSpeedColumn(),
            TimeRemainingColumn(), console=console, disable=not progress,
        )
        with progress_ctx as p, tmp.open("wb") as f:
            task = p.add_task(dest.name, total=total or None)
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                p.update(task, advance=len(chunk))
    tmp.replace(dest)


def _download_multipart(
    spec: ModelSpec, base_filename: str, cache_dir: Path,
    console: Console, progress: bool,
) -> Path:
    """Download multi-part GGUF files (e.g. base-00001-of-00002.gguf).

    Returns the path to the first part (llama.cpp loads splits from part 1).
    Checks local cache first before hitting the network.
    """
    base = base_filename.replace(".gguf", "")
    parts: list[Path] = []
    part_num = 1

    # Phase 1: Check if parts are already on disk (no network needed).
    for total in range(2, 6):
        local_parts: list[Path] = []
        for pn in range(1, total + 1):
            suffix = f"-{pn:05d}-of-{total:05d}.gguf"
            filename = f"{base}{suffix}"
            dest = cache_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                local_parts.append(dest)
            else:
                break
        if local_parts and len(local_parts) == total:
            console.print(f"[green]Model already present ({total} parts):[/green] {local_parts[0].parent.name}/{local_parts[0].name}")
            return local_parts[0]

    # Phase 2: Not fully cached — download from network.
    while True:
        # Try patterns: base-00001-of-00002.gguf, base-00001-of-00003.gguf, etc.
        # We don't know the total, so try incrementing part counts.
        found = False
        for total in range(2, 6):
            suffix = f"-{part_num:05d}-of-{total:05d}.gguf"
            filename = f"{base}{suffix}"
            url = _download_url(spec.hf_repo, filename)
            try:
                resp = httpx.head(url, follow_redirects=True, timeout=30.0)
                if resp.status_code == 200:
                    dest = cache_dir / filename
                    if not (dest.exists() and dest.stat().st_size > 0):
                        _download_single(url, dest, console, progress)
                    parts.append(dest)
                    found = True
                    # Check if there's a next part.
                    part_num += 1
                    break
            except Exception:
                continue
        if not found:
            break
    if not parts:
        raise FileNotFoundError(
            f"Could not find GGUF file(s) for {spec.hf_repo}/{base_filename} "
            f"(tried single + multi-part patterns)"
        )
    return parts[0]


# ---------------------------------------------------------------------------
# llama.cpp init
# ---------------------------------------------------------------------------

def load_llama(
    cfg: UltResConfig,
    spec: ModelSpec | None = None,
    model_path: Path | None = None,
    mode: str = "deep",
    **overrides: Any,
) -> "Any":
    """Initialize a llama-cpp-python `Llama` instance.

    Either pass `model_path` directly, or pass `spec` (defaults to cfg's selection).
    Context window defaults to the spec's native_ctx unless cfg.model.n_ctx is set.
    When cfg.model.use_extended_context is True and the spec supports YaRN,
    context is extended via rope scaling (32K → 64K).

    v1.6.1: `mode` controls context size + GPU layers for speed:
    - "fast": 32K context, 55 GPU layers (research loop — only needs 28K)
    - "deep": 48K context, 52 GPU layers (deep pipeline — needs up to 28K)
    - "full": 65K context, 50 GPU layers (maximum context, original behavior)
    """
    # Ensure CUDA DLL paths are available BEFORE importing llama_cpp,
    # because the import itself loads the shared library.
    _add_cuda_dll_paths()
    from llama_cpp import Llama  # type: ignore[import-not-found]

    if model_path is None:
        spec = spec or get_spec(cfg.model.selection)
        model_path = download_model(spec, cfg.model.quant, cfg.model_cache_dir)

    # v1.6.1: Determine context window + GPU layers based on mode.
    if cfg.model.n_ctx:
        n_ctx = cfg.model.n_ctx
    elif mode == "fast":
        n_ctx = cfg.model.fast_ctx
    elif mode == "deep":
        n_ctx = cfg.model.deep_ctx
    elif spec and cfg.model.use_extended_context and spec.extended_ctx > 0:
        n_ctx = cfg.model.extended_ctx or spec.extended_ctx
    elif spec:
        n_ctx = spec.native_ctx
    else:
        n_ctx = 32_768

    # v1.6.1: GPU layers based on mode (unless explicitly overridden).
    if "n_gpu_layers" in overrides:
        n_gpu = overrides.pop("n_gpu_layers")
    elif mode == "fast":
        n_gpu = cfg.model.fast_n_gpu_layers
    elif mode == "deep":
        n_gpu = cfg.model.deep_n_gpu_layers
    else:
        n_gpu = cfg.model.n_gpu_layers

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu,
        "n_threads": max(1, os.cpu_count() or 4),
        "verbose": False,
        # v1.6.1: Inference speed optimizations (lossless).
        "n_batch": 2048,          # Faster prompt processing (was default 512).
        "n_ubatch": 512,          # Uniform batch size.
        "n_threads_batch": max(1, os.cpu_count() or 4),  # All threads for batch.
        "no_perf": True,          # Skip performance counter collection.
        "offload_kqv": True,      # GPU-accelerated KV cache ops.
        "use_mmap": True,         # Memory-mapped model loading (faster).
    }

    # v1.5: Flash attention (enabled by default for Qwen3.8 hybrid attention).
    if cfg.model.flash_attn:
        kwargs["flash_attn"] = True

    # v1.5: Quantized KV cache (q4_0 = type 2, saves ~75% KV memory).
    # Essential for fitting 64K context in 8GB VRAM with a 27B model.
    if cfg.model.kv_cache_type and cfg.model.kv_cache_type > 0:
        kwargs["type_k"] = cfg.model.kv_cache_type
        kwargs["type_v"] = cfg.model.kv_cache_type

    # YaRN rope scaling for extended context.
    if spec and cfg.model.use_extended_context and spec.extended_ctx > 0 and n_ctx > spec.native_ctx:
        # LLAMA_ROPE_SCALING_TYPE_YARN = 2
        kwargs["rope_scaling_type"] = 2
        kwargs["yarn_orig_ctx"] = spec.native_ctx

    kwargs.update(overrides)
    llm = Llama(**kwargs)

    # Warn if GPU offload was requested but no GPU is available.
    if n_gpu == -1 and detect_vram_gb() == 0.0:
        import warnings

        warnings.warn(
            "n_gpu_layers=-1 but no GPU detected. Model will run on CPU. "
            "Install the CUDA wheel: pip install llama-cpp-python==0.3.35 "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125 "
            "--force-reinstall --no-deps",
            stacklevel=2,
        )
    return llm


def unload_model(llm: Any) -> None:
    """Properly free VRAM from a loaded Llama instance.

    Closes the llama.cpp model context, deletes the model object, and forces
    garbage collection + CUDA cache clearing. Needed for dual-model workflows
    where we swap between Instruct and Coder on 8GB VRAM.
    """
    # Explicitly close the model's internal resources (mmap, contexts).
    try:
        if hasattr(llm, "close"):
            llm.close()
    except Exception:
        pass
    # Also try the context manager exit for older llama-cpp-python versions.
    try:
        if hasattr(llm, "_stack"):
            llm._stack.close()
    except Exception:
        pass
    del llm
    import gc

    # Run GC twice to ensure all references are collected.
    gc.collect()
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_coder_model(cfg: UltResConfig) -> "Any":
    """Load the coder model for the implementation pass.

    v1.5: With Qwen3.8-27B as the default, the "coder" is the same model —
    no separate coder model is loaded. This function is kept for backward
    compatibility but just loads the stock model.
    """
    spec = resolve_spec(cfg)
    model_path = download_model(spec, cfg.model.quant, cfg.model_cache_dir)
    return load_llama(cfg, spec=spec, model_path=model_path)


class ModelManager:
    """Manages model loading for the pipeline.

    v1.5: With Qwen3.8-27B as the default, there is NO model swap — the same
    27B model handles all stages (plan, summarize, implement, critique).
    This is possible because the 27B is smart enough for both reasoning and
    coding, eliminating the v1.2-v1.4 Instruct→Coder→Instruct swap cycle.

    For legacy 7B models, the dual-model swap behavior is preserved via
    cfg.deep_research.coder_model.

    Usage (v1.5 single-model):
        mgr = ModelManager(cfg)
        llm = mgr.load_instruct()   # all stages use this one model
        mgr.unload_current()

    Usage (legacy dual-model):
        mgr = ModelManager(cfg)
        llm = mgr.load_instruct()   # stages 1-7
        mgr.unload_current()
        coder = mgr.load_coder()    # stage 8 pass 2
        mgr.unload_current()
        llm = mgr.load_instruct()   # stage 9
    """

    def __init__(self, cfg: UltResConfig, mode: str = "deep"):
        self.cfg = cfg
        self.mode = mode  # v1.6.1: "fast", "deep", or "full"
        self._current: Any = None
        self._current_key: str | None = None
        self._instruct_path: Path | None = None
        self._coder_path: Path | None = None
        self._instruct_spec = None
        self._coder_spec = None

    @property
    def is_single_model(self) -> bool:
        """True when using a single model for all stages (v1.5 Qwen3.8-27B)."""
        # Single-model mode when the stock model is NOT a legacy 7B.
        spec = resolve_spec(self.cfg)
        return spec.params_b > 10.0  # 27B = single model; 7B = dual model

    def load_instruct(self) -> "Any":
        """Load the Instruct (main) model, unloading any currently loaded model first."""
        if self._current_key == "instruct" and self._current is not None:
            return self._current
        self.unload_current()
        if self._instruct_path is None:
            self._instruct_spec = resolve_spec(self.cfg)
            self._instruct_path = download_model(
                self._instruct_spec, self.cfg.model.quant, self.cfg.model_cache_dir
            )
        self._current = load_llama(self.cfg, spec=self._instruct_spec, model_path=self._instruct_path, mode=self.mode)
        self._current_key = "instruct"
        return self._current

    def load_coder(self) -> "Any":
        """Load the Coder model for the implementation pass.

        v1.5: In single-model mode (Qwen3.8-27B), this returns the SAME model
        that's already loaded — no swap needed. The 27B handles both planning
        and coding.
        """
        # Single-model mode: just return the already-loaded instruct model.
        if self.is_single_model:
            return self.load_instruct()
        # Legacy dual-model mode: load the separate coder model.
        if self._current_key == "coder" and self._current is not None:
            return self._current
        self.unload_current()
        if self._coder_path is None:
            self._coder_spec = get_spec(self.cfg.deep_research.coder_model)
            self._coder_path = download_model(
                self._coder_spec, self.cfg.model.quant, self.cfg.model_cache_dir
            )
        self._current = load_llama(self.cfg, spec=self._coder_spec, model_path=self._coder_path, mode=self.mode)
        self._current_key = "coder"
        return self._current

    def unload_current(self) -> None:
        """Unload the currently loaded model and free VRAM.

        Drops our reference BEFORE calling gc.collect() so the model is
        actually eligible for collection. Adds a small delay on Windows
        to let the OS release file handles.
        """
        if self._current is not None:
            # Grab a local ref, then drop the manager's reference so
            # gc.collect() inside unload_model can actually collect it.
            llm = self._current
            self._current = None
            self._current_key = None
            unload_model(llm)
            # Give Windows time to release mmap file handles.
            if sys.platform == "win32":
                import time as _time
                _time.sleep(1.0)

    def current(self) -> "Any | None":
        """Return the currently loaded model, or None."""
        return self._current

    @property
    def current_key(self) -> str | None:
        """Return the key of the currently loaded model ('instruct' or 'coder')."""
        return self._current_key


def resolve_spec(cfg: UltResConfig) -> ModelSpec:
    """Return the ModelSpec matching the config's model selection."""
    return get_spec(cfg.model.selection)


def ensure_model(cfg: UltResConfig) -> Path:
    """Download (if needed) and return the local path to the active model."""
    spec = resolve_spec(cfg)
    return download_model(spec, cfg.model.quant, cfg.model_cache_dir)
