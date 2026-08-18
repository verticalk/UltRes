# AGENTS.md

Guidance for AI agents (and humans) working on the UltRes codebase.

**Repo:** https://github.com/verticalk/UltRes (private)

> **Keep this file up to date.** If you make changes to the codebase — new
> features, fixed bugs, changed dependencies, updated configs, new verification
> results, roadmap progress — update the relevant sections of AGENTS.md in the
> same commit. Do not let this file go stale. It is the single source of truth
> for project status, build/test commands, and architecture decisions.

## Project status

**v1.4 — Bug fixes, optimization, and v1.5 readiness. Fixes critical bugs (model lifecycle crash, Playwright memory exhaustion, missing disk persistence), adds per-domain rate limiting, graceful cancellation, unified trajectory format, and training-ready export.**

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
| v1.2: Deep research pipeline (bulk crawl 2000+ pages) | Done |
| v1.2: Query expansion (90-250 search queries) | Done |
| v1.2: Source prioritizer (GitHub/SO/cppreference for code) | Done |
| v1.2: Vector clustering + TF-IDF labeling | Done |
| v1.2: Gap detection + multi-round re-crawl | Done |
| v1.2: Batch summarize + hierarchical compression | Done |
| v1.2: Two-pass implementation (Instruct plan → Coder code) | Done |
| v1.2: Live streaming with per-stage timers | Done |
| v1.2: 64K context via YaRN (verified working) | Done |
| **v1.4: ModelManager (fixes dual-model crash on 8GB VRAM)** | **Done** |
| **v1.4: httpx-first fetch (10x faster, no Chromium spam)** | **Done** |
| **v1.4: Pages persisted to disk KnowledgeStore in deep mode** | **Done** |
| **v1.4: Per-domain rate limiting (max 3 concurrent/domain)** | **Done** |
| **v1.4: Graceful Ctrl+C cancellation with partial save** | **Done** |
| **v1.4: Unified trajectory format (fast + deep)** | **Done** |
| **v1.4: Training-ready export (`ultres trajectories export`)** | **Done** |
| **v1.4: Trajectory validation + statistics** | **Done** |
| **v1.4: Pipeline timeout (configurable, default 90 min)** | **Done** |
| **v1.4: Fixed 10 critical bugs from v1.2** | **Done** |
| Tests (76 unit tests) | Done — all passing |
| Git repo + pushed to GitHub | Done |

**v1.4 changes from v1.2:**
- **Fixed: Model lifecycle crash** — CLI loaded both models into 8GB VRAM. New `ModelManager` class swaps models at pipeline stage boundaries.
- **Fixed: Playwright memory exhaustion** — Crawler now uses httpx-first with Playwright fallback (10x faster, no browser spam for 2000+ pages).
- **Fixed: Pages not persisted to disk** — Deep mode now ingests crawled pages into KnowledgeStore + VectorIndex as they're fetched.
- **Fixed: Stage name mismatch** — Crawler progress updates were silently failing. Now parameterized.
- **Fixed: Empty crawl not handled** — Pipeline returns graceful error instead of producing garbage.
- **Fixed: Clusterer index pollution** — Crawl docs are cleaned up after clustering.
- **Fixed: Weak critique logic** — Now properly extracts code blocks from critique, doesn't replace with raw prose.
- **Fixed: Pass 1 plan not saved** — Trajectory now includes both plan and implementation.
- **Fixed: Version strings outdated** — Bumped to 1.4.0, added `__version__`.
- **Fixed: CLI docstring outdated** — Updated to match actual flags.
- **Added: Per-domain rate limiting** — Max 3 concurrent fetches per domain.
- **Added: Graceful cancellation** — Ctrl+C saves partial results.
- **Added: Pipeline timeout** — Configurable, default 90 min.
- **Added: Unified trajectory format** — Both fast and deep modes use the same schema.
- **Added: Training-ready export** — `ultres trajectories export --output train.jsonl`.
- **Added: Trajectory validation + statistics** — `ultres trajectories validate` and `ultres trajectories stats`.
- **Improved: Query type detection** — Removed overly generic keywords ("app", "calculator", etc.).
- **Improved: Batch summarize parsing** — Uses numbered headers instead of "---" separator.
- **Improved: CLI** — `ultres list` shows both fast and deep queries. `ultres show` shows plan + brief. `ultres clean` also cleans trajectories.

**Base model:** Qwen2.5-7B-Instruct Q4_K_M (4.7 GB, Apache 2.0, 64K context via YaRN).
General-purpose instruct model with native function-calling support.
Coder model: Qwen2.5-Coder-7B-Instruct (used for implementation pass in two-pass mode).

**Roadmap:**
- v1.2 (done): Deep research pipeline + 64K context + two-pass + streaming + gap detection
- v1.4 (done): Bug fixes, optimization, v1.5 readiness (unified trajectories, export, validation)
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
pytest                      # unit tests in tests/ (76 tests, no GPU/model needed)
```

Tests cover: store CRUD, hierarchical summary tree, vector recall, tool-call
JSON parsing (native + fallback), config TOML round-trip, LoRA cache stub,
trajectory save/load/export/validate/stats, streaming, source prioritizer,
clusterer, gap detector, compressor, pipeline critique logic, domain rate
limiting, unified trajectory format. They do NOT require a GPU, downloaded
model, or network access.

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

- **76/76 tests pass** on Python 3.12.8, Windows 11.
- **GPU**: RTX 4060 Ti 8GB, CUDA 12.1 wheel, **43.6 tokens/sec**.
- **Model**: Qwen2.5-7B-Instruct Q4_K_M (4.7 GB, multi-part GGUF, 64K via YaRN).
- **Context**: 64K via YaRN verified working (128K fails — KV cache too large).
- **Native tool calling**: Qwen2.5-7B-Instruct uses `tools=` API parameter.
- **Search**: SearXNG in Docker with JSON format + limiter disabled.
- **Deep pipeline**: 10-stage pipeline with bulk crawl, clustering, gap detection, two-pass implementation.
- **ModelManager**: Sequential model loading/unloading (Instruct for stages 1-7+9, Coder for stage 8).
- **httpx-first fetch**: 10x faster than Playwright for bulk crawl, with Playwright fallback.
- **Disk persistence**: Deep mode pages ingested into KnowledgeStore + VectorIndex.
- **Live streaming**: Per-stage timers, progress bars, token-by-token streaming.
- **Self-critique**: After implementation, answer is critiqued against research (extracts code blocks only).
- **Trajectories**: Unified format (fast + deep), saved to `.ultres/trajectories/` for v1.5 QLoRA.
- **Trajectory export**: `ultres trajectories export --output train.jsonl` produces training-ready JSONL.

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

- **v1.4 ships no training.** The `lora/` module is a loader stub + trajectory accumulator + export. Training is v1.5.
- **v1.4 uses Qwen2.5-7B-Instruct** (64K via YaRN, native tool calling). Coder model: Qwen2.5-Coder-7B for implementation pass.
- **64K context via YaRN** (verified). 128K fails — KV cache too large for 8GB VRAM + 16GB RAM.
- **GPU requires CUDA 12.1 wheel** (v0.3.4). CPU wheel works as fallback.
- **Deep pipeline takes 40-60 min** per query. Use `--fast` for quick queries. Pipeline timeout default 90 min.
- **ModelManager**: 8GB VRAM can't hold both models. Sequential load/unload via ModelManager.
- **httpx-first fetch**: Bulk crawl uses httpx (10x faster), falls back to Playwright for JS-heavy sites.
- **Per-domain rate limiting**: Max 3 concurrent fetches per domain to avoid hammering sites.
- **KV-cache reuse is best-effort.** v2 wires in real persistence.
- **Self-critique adds latency**. Disable via `--no-critique`.

## v1.5 training (Kaggle free T4×2)

Goal: produce `UltRes-Base-7B` via QLoRA on UltRes agent trajectories.

1. **Trajectories accumulate automatically** in `.ultres/trajectories/` as
   JSONL files (unified v1.4 format: query → mode → plan → answer → brief →
   pages → clusters → gaps → visited_urls → doc_ids → code_ids → timing).
   Run v1.4 on a curated set of coding/research tasks to build the dataset.
   Target ~5-10K examples.
2. **Export training data**: `ultres trajectories export --output train.jsonl`
   produces instruction/response pairs with research context. Use
   `ultres trajectories stats` to check dataset size and
   `ultres trajectories validate` to check quality.
3. **Upload as a Kaggle dataset** (CPU session, internet on).
4. **Train on Kaggle T4×2** (GPU session, internet off):
   - Base: `unsloth/qwen2.5-7b-instruct-bnb-4bit` (general reasoning model)
   - LoRA rank 32, alpha 64, all linear layers
   - Max seq len 4096, batch 2, grad accum 4, 3 epochs, LR 2e-4 cosine
   - Use Unsloth's official Kaggle T4×2 notebooks as the template.
5. **Merge + convert to GGUF** (Q4_K_M, Q8_0) via llama.cpp.
6. **Publish** to HuggingFace as `ultres/UltRes-Base-7B-GGUF`.
7. **Pin** the repo URL + sha256 in `ultres/models/registry.py`.

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
