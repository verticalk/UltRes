# AGENTS.md

Guidance for AI agents (and humans) working on the UltRes codebase.

**Repo:** https://github.com/verticalk/UltRes (private)

> **Keep this file up to date.** If you make changes to the codebase — new
> features, fixed bugs, changed dependencies, updated configs, new verification
> results, roadmap progress — update the relevant sections of AGENTS.md in the
> same commit. Do not let this file go stale. It is the single source of truth
> for project status, build/test commands, and architecture decisions.

## Project status

**v1.2 — Deep research pipeline with bulk crawl, gap detection, two-pass implementation, and live streaming.**

| Milestone | Status |
|---|---|
| Package scaffold + CLI | Done |
| Config (Pydantic + TOML) | Done |
| Model registry + loader (GGUF + llama.cpp + YaRN 64K) | Done |
| Search layer (SearXNG / Tavily / Brave + fetch) | Done |
| Memory layer (store / vector / hierarchical / KV cache) | Done |
| Agent layer (tools / planner / research loop / reasoner) | Done |
| LoRA cache stub + trajectory accumulation (v1.5/v2) | Done |
| GPU acceleration (CUDA 12.1 wheel, RTX 4060 Ti) | Done — 43.6 tokens/sec |
| Native tool calling (Qwen2.5-7B-Instruct) | Done |
| Synthesis pass (evidence-cited answers) | Done |
| Self-critique loop (reasoning about reasoning) | Done |
| Trajectory saving for v1.5 QLoRA | Done |
| **v1.2: Deep research pipeline (bulk crawl 2000+ pages)** | **Done** |
| **v1.2: Query expansion (90-250 search queries)** | **Done** |
| **v1.2: Source prioritizer (GitHub/SO/cppreference for code)** | **Done** |
| **v1.2: Vector clustering + TF-IDF labeling** | **Done** |
| **v1.2: Gap detection + multi-round re-crawl** | **Done** |
| **v1.2: Batch summarize + hierarchical compression** | **Done** |
| **v1.2: Two-pass implementation (Instruct plan → Coder code)** | **Done** |
| **v1.2: Live streaming with per-stage timers** | **Done** |
| **v1.2: 64K context via YaRN (verified working)** | **Done** |
| **v1.2: Dual model loading (load/unload Instruct ↔ Coder)** | **Done** |
| Tests (54 unit tests) | Done — all passing |
| End-to-end pipeline verified | Done |
| Git repo + pushed to GitHub | Done |

**v1.2 changes from v1.1:**
- **Deep research pipeline**: Default mode now crawls 2000+ pages per query (was ~10).
- **Query expansion**: 30-50 subtasks × 3-5 variations = 90-250 search queries.
- **Source prioritization**: Code queries prioritize GitHub, Stack Overflow, cppreference.
- **Quality filtering**: Pages scored and filtered by relevance, content, code blocks.
- **Clustering**: Vector clustering into 50-200 topic clusters with TF-IDF labels.
- **Gap detection**: Model reviews coverage, identifies missing topics, re-crawls for gaps.
- **Batch summarize**: 5 clusters per model call, hierarchical compression into master brief.
- **Two-pass implementation**: Instruct model writes plan from research, Coder model implements from plan. Coder is FORCED to follow research, not pretrained knowledge.
- **Live streaming**: Per-stage timers, progress bars, token-by-token streaming for implementation.
- **64K context**: YaRN rope scaling (verified working on RTX 4060 Ti 8GB).
- **Dual model loading**: Sequential load/unload of Instruct and Coder models to fit 8GB VRAM.
- **Fast mode**: `--fast` flag runs the v1.1 agentic loop for quick queries.

**Base model:** Qwen2.5-7B-Instruct Q4_K_M (4.7 GB, Apache 2.0, 64K context via YaRN).
General-purpose instruct model with native function-calling support.
Coder model: Qwen2.5-Coder-7B-Instruct (used for implementation pass in two-pass mode).

**Roadmap:**
- v1.2 (done): Deep research pipeline + 64K context + two-pass + streaming + gap detection
- v1.5 (next): QLoRA-specialize on accumulated UltRes trajectories → UltRes-Base-7B (Kaggle free T4)
- v2: full fine-tune + YaRN long-context extension (64K→256K) → UltRes-Base-7B-Long (cloud GPU credits)

## Build & install

```bash
pip install -e .            # install package + ultres CLI
pip install -e ".[dev]"     # + pytest, pytest-asyncio

# GPU inference (NVIDIA CUDA 12.1, no compiler needed):
pip install llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --force-reinstall --no-deps
pip install nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-nvrtc-cu12==12.4.127

# CPU inference (fallback, no GPU needed):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

playwright install chromium # for the page fetcher
```

## Test

```bash
pytest                      # unit tests in tests/ (54 tests, no GPU/model needed)
```

Tests cover: store CRUD, hierarchical summary tree, vector recall, tool-call
JSON parsing (native + fallback), config TOML round-trip, LoRA cache stub,
trajectory save/load, streaming, source prioritizer, clusterer, gap detector,
compressor. They do NOT require a GPU, downloaded model, or network access.

## Run

```bash
# 1. Start SearXNG (one-time):
docker run -d --name searxng -p 8080:8080 -v ./searxng_settings.yml:/etc/searxng/settings.yml:ro searxng/searxng

# 2. Pull the models (~4.7 GB each, multi-part GGUF):
ultres models pull              # Instruct model (default)
ultres models pull --model coder # Coder model (for two-pass)

# 3. Run a deep research query (default, 2000+ pages, ~40-60 min):
ultres "build me a complex C++ calculator app"

# 4. Fast mode (v1.1 agentic loop, 30 steps, ~2 min):
ultres --fast "What is C++ std::vector?"

# 5. Options:
ultres --max-pages 500 "query"     # Limit crawl to 500 pages
ultres --no-critique "query"       # Skip self-critique
ultres --no-stream "query"         # Disable live streaming
```

Requires a running SearXNG instance (see README) or a Tavily/Brave API key.

## Verified working

- **54/54 tests pass** on Python 3.12.8, Windows 11.
- **GPU**: RTX 4060 Ti 8GB, CUDA 12.1 wheel, **43.6 tokens/sec**.
- **Model**: Qwen2.5-7B-Instruct Q4_K_M (4.7 GB, multi-part GGUF, 64K via YaRN).
- **Context**: 64K via YaRN verified working (128K fails — KV cache too large).
- **Native tool calling**: Qwen2.5-7B-Instruct uses `tools=` API parameter.
- **Search**: SearXNG in Docker with JSON format + limiter disabled.
- **Deep pipeline**: 10-stage pipeline with bulk crawl, clustering, gap detection, two-pass implementation.
- **Live streaming**: Per-stage timers, progress bars, token-by-token streaming.
- **Self-critique**: After implementation, answer is critiqued against research.
- **Trajectories**: Saved to `.ultres/trajectories/` with timing data for v1.5 QLoRA.

## Architecture (one-paragraph)

**Deep mode (default):** A user query goes to the **Planner** (generates 30-50
subtasks × 3-5 variations = 90-250 search queries). The **Bulk Crawler** fetches
2000+ pages in parallel with quality filtering and link following. Pages are
**clustered** into 50-200 topic clusters. **Gap detection** identifies missing
topics and re-crawls. Clusters are **batch summarized** and hierarchically
compressed into a **master brief** (~5000 words). **Two-pass implementation**:
Instruct model writes a plan from the brief, Coder model implements from the
plan (forced to follow research). **Self-critique** checks the implementation
against research. Trajectory saved for v1.5 QLoRA. All stages show **live
streaming** with per-stage timers.

**Fast mode (`--fast`):** The v1.1 agentic loop — model picks tool calls
(`search`/`visit`/`recall`/`load_*`/`finish`) via native function calling.
30 steps, ~2 min, touches ~10 pages.

## Key constraints

- **v1.2 ships no training.** The `lora/` module is a loader stub + trajectory accumulator. Training is v1.5.
- **v1.2 uses Qwen2.5-7B-Instruct** (64K via YaRN, native tool calling). Coder model: Qwen2.5-Coder-7B for implementation pass.
- **64K context via YaRN** (verified). 128K fails — KV cache too large for 8GB VRAM + 16GB RAM.
- **GPU requires CUDA 12.1 wheel** (v0.3.4). CPU wheel works as fallback.
- **Deep pipeline takes 40-60 min** per query. Use `--fast` for quick queries.
- **Dual model loading**: 8GB VRAM can't hold both models. Sequential load/unload.
- **KV-cache reuse is best-effort.** v2 wires in real persistence.
- **Self-critique adds latency**. Disable via `--no-critique`.

## v1.5 training (Kaggle free T4×2)

Goal: produce `UltRes-Base-7B` via QLoRA on UltRes agent trajectories.

1. **Trajectories accumulate automatically** in `.ultres/trajectories/` as
   JSONL files (query → plan → tool calls → answer → doc_ids → code_ids).
   Run v1.1 on a curated set of coding/research tasks to build the dataset.
   Target ~5-10K examples. Use `ultres.lora.cache.list_trajectories()` to
   inspect the accumulated data.
2. **Upload as a Kaggle dataset** (CPU session, internet on).
3. **Train on Kaggle T4×2** (GPU session, internet off):
   - Base: `unsloth/qwen2.5-7b-instruct-bnb-4bit` (general reasoning model)
   - LoRA rank 32, alpha 64, all linear layers
   - Max seq len 4096, batch 2, grad accum 4, 3 epochs, LR 2e-4 cosine
   - Use Unsloth's official Kaggle T4×2 notebooks as the template.
4. **Merge + convert to GGUF** (Q4_K_M, Q8_0) via llama.cpp.
5. **Publish** to HuggingFace as `ultres/UltRes-Base-7B-GGUF`.
6. **Pin** the repo URL + sha256 in `ultres/models/registry.py`.

## v2 training (Vultr $250 free A100 80GB, ~68 hrs)

Goal: `UltRes-Base-7B-Long` via full fine-tune + YaRN 32K→256K, plus
per-project LoRA cache training.

1. **Dry-run the pipeline on Kaggle T4 first** (smaller scale) to avoid
   wasting A100 hours on bugs.
2. **Full fine-tune** on expanded UltRes corpus (~8-16 hrs on A100 80GB).
3. **YaRN extension:** apply 8x rope-scaling, fine-tune on 32K-128K sequences
   using Unsloth's long-context gradient checkpointing (~20-40 hrs).
4. **Per-project LoRA:** QLoRA-train small adapters on user-revisited topics
   (feasible on the user's 4060 Ti 8GB at seq 2K, or on Kaggle).
5. **Publish** `ultres/UltRes-Base-7B-Long-GGUF` + pin in registry.

## Files to touch for common changes

| Change | Files |
|---|---|
| Add a search backend | `ultres/search/base.py` (Protocol), new provider file, `ultres/search/__init__`, `ultres/cli.py` (provider construction in `run`) |
| Add a tool | `ultres/agent/tools.py` (schema + system prompt), `ultres/agent/research_loop.py` (`ToolExecutor.execute` + `_extract_tool_call`) |
| Change hot-window eviction | `ultres/agent/research_loop.py` (`HotWindow._evict`) |
| Add a model | `ultres/models/registry.py` (`_REGISTRY`), optionally `ultres/models/loader.py` |
| Change store layout | `ultres/memory/store.py` (all paths derived from `qdir`) |
