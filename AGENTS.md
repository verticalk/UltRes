# AGENTS.md

Guidance for AI agents (and humans) working on the UltRes codebase.

**Repo:** https://github.com/verticalk/UltRes (private)

> **Keep this file up to date.** If you make changes to the codebase — new
> features, fixed bugs, changed dependencies, updated configs, new verification
> results, roadmap progress — update the relevant sections of AGENTS.md in the
> same commit. Do not let this file go stale. It is the single source of truth
> for project status, build/test commands, and architecture decisions.

## Project status

**v1.6 — Smarter research, coding, and benchmark results. 10 pipeline improvements: iterative planner with coverage feedback, smarter code extraction (markdown fences + Stack Overflow), adaptive clustering with semantic labels, deeper compression (8K-word briefs), file output with compile+test verification, thinking-mode system prompt, HumanEval+MMLU benchmark harness, query diversification, depth-aware gap detection, adaptive max_tokens. 92 tests passing.**

**v1.5 — Single-model upgrade to Qwen3.8-27B. Eliminates the dual-model swap (Instruct→Coder→Instruct) by using one 27B model for all stages. Hybrid attention architecture (16/64 layers full attn, 48 Gated DeltaNet) enables 64K context in 8GB VRAM with q4_0 KV cache. Also fixes clustering bottleneck (74min → 8sec, 550x faster).**

| Milestone | Status |
|---|---|
| Package scaffold + CLI | Done |
| Config (Pydantic + TOML) | Done |
| Model registry + loader (GGUF + llama.cpp + YaRN 64K) | Done |
| Search layer (SearXNG / Tavily / Brave + fetch) | Done |
| Memory layer (store / vector / hierarchical / KV cache) | Done |
| Agent layer (tools / planner / research loop / reasoner) | Done |
| LoRA cache stub + trajectory accumulation (v1.5/v2) | Done |
| GPU acceleration (CUDA 12.5 wheel, RTX 4060 Ti) | Done — 7.9 tokens/sec (27B) |
| Native tool calling (Qwen3.8-27B) | Done |
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
| v1.4: ModelManager (fixes dual-model crash on 8GB VRAM) | Done |
| v1.4: httpx-first fetch (10x faster, no Chromium spam) | Done |
| v1.4: Pages persisted to disk KnowledgeStore in deep mode | Done |
| v1.4: Per-domain rate limiting (max 3 concurrent/domain) | Done |
| v1.4: Graceful Ctrl+C cancellation with partial save | Done |
| v1.4: Unified trajectory format (fast + deep) | Done |
| v1.4: Training-ready export (`ultres trajectories export`) | Done |
| v1.4: Trajectory validation + statistics | Done |
| v1.4: Pipeline timeout (configurable, default 90 min) | Done |
| v1.4: Fixed 10 critical bugs from v1.2 | Done |
| **v1.5: Qwen3.8-27B as default model (single-model, no swap)** | **Done** |
| **v1.5: Hybrid attention support (Gated DeltaNet, 16/64 layers)** | **Done** |
| **v1.5: q4_0 KV cache + flash attention (64K context in 8GB)** | **Done** |
| **v1.5: Thinking mode (Qwen3.8 reasoning before answers)** | **Done** |
| **v1.5: Clustering bottleneck fix (74min → 8sec, 550x faster)** | **Done** |
| **v1.5: llama-cpp-python 0.3.35 (hybrid attention + cu125 wheel)** | **Done** |
| **v1.6: Iterative planner with coverage feedback** | **Done** |
| **v1.6: Smarter code extraction (markdown fences + SO-specific)** | **Done** |
| **v1.6: Adaptive clustering + semantic labeling** | **Done** |
| **v1.6: Deeper compression (8K-word briefs, code patterns section)** | **Done** |
| **v1.6: File output + compile + test verification (Pass 3)** | **Done** |
| **v1.6: Thinking-mode system prompt (research strategy + quality bar)** | **Done** |
| **v1.6: Benchmark harness (HumanEval pass@1 + MMLU accuracy)** | **Done** |
| **v1.6: Query diversification + near-duplicate URL dedup** | **Done** |
| **v1.6: Depth-aware gap detection (shallow cluster flagging)** | **Done** |
| **v1.6: Adaptive max_tokens by task complexity** | **Done** |
| Tests (92 unit tests) | Done — all passing |
| Git repo + pushed to GitHub | Done |

**v1.5 changes from v1.4:**
- **Upgraded: Default model to Qwen3.8-27B-UD-IQ2_XXS** — 27B dense model with hybrid attention (16/64 layers full attention, 48 Gated DeltaNet linear attention). Beats Opus 4.6 Max on SWE-bench Pro (61.7 vs 53.4). Apache 2.0.
- **Eliminated: Dual-model swap** — The 27B handles all stages (plan, summarize, implement, critique) with no model switching. `ModelManager.is_single_model` returns True for 27B, `load_coder()` returns the same model.
- **Added: q4_0 KV cache** — Quantized KV cache (type_k=2, type_v=2) reduces KV memory by ~75%. Essential for fitting 64K context in 8GB VRAM with a 27B model.
- **Added: Flash attention** — Enabled by default for Qwen3.8's hybrid attention architecture.
- **Added: Thinking mode** — Qwen3.8 generates `thinking` reasoning before answers. Thinking tokens are stripped from final output via `_strip_thinking()` but used for better reasoning quality.
- **Fixed: Clustering bottleneck** — Clusterer was making ~14,000 wasted ChromaDB recall calls (result ignored with `pass`). Rewrote to embed all pages in a single batch call and do in-memory cosine similarity. 74min → 8sec (550x faster) for 600 pages.
- **Upgraded: llama-cpp-python 0.3.4 → 0.3.35** — Required for Qwen3.8 hybrid attention (Gated DeltaNet) support. Uses cu125 CUDA wheel (cu124 crashes with STATUS_ILLEGAL_INSTRUCTION).
- **Changed: GPU layers default** — `-1` (all) → `50` (50/65 layers on GPU for 27B on 8GB VRAM).
- **Changed: Config defaults** — `quant` = `UD-IQ2_XXS`, `n_gpu_layers` = `50`, `flash_attn` = `True`, `kv_cache_type` = `2` (q4_0), `enable_thinking` = `True`.
- **Kept: Legacy 7B models** — `qwen25-7b` and `coder` model keys preserved for backward compatibility. Use `--model qwen25-7b` for the fast 7B model.
- **Kept: Dual-model mode** — When using a 7B model (`params_b <= 10`), `ModelManager.is_single_model` returns False and the Instruct→Coder swap is preserved.

**v1.6 changes from v1.5:**
- **Added: Iterative planner with coverage feedback** — `plan_deep_iterative()` runs quick searches for the top 5 subtasks, feeds result titles back to the model, and asks it to refine the plan (remove dead ends, add promising angles). One extra model call (~30 sec) for much more targeted queries.
- **Added: Smarter code extraction** — Fetcher now extracts markdown code fences (```language ... ```) in addition to `<pre><code>` blocks. Stack Overflow-specific extraction pulls code from answer cells. Code density scoring in `score_page()` gives bonus to pages with high code-to-text ratio.
- **Added: Adaptive clustering + semantic labeling** — Over-merged clusters (10+ pages, avg pairwise similarity < 0.5) are split at a higher threshold. Semantic labels generated by the model (e.g., "Qt signal-slot event handling" instead of "qt signal slot event handler connect").
- **Added: Deeper compression** — Page text samples increased 1000→2000 chars, code samples 500→1000 chars. Master brief increased 5000→8000 words with new "Code Patterns" section (top 5 code snippets inline) and "Architecture Decisions" with rationale bullets.
- **Added: File output + compile + test verification (Pass 3)** — Model outputs files in ```file:path``` format. Files are extracted to a temp directory, syntax-checked (py_compile/g++), import-checked, and tests run in a sandboxed subprocess with 30s timeout. Failed verification triggers a fix pass (max 2 iterations).
- **Added: Thinking-mode system prompt** — Rewritten to teach the model to use thinking tokens for analyzing search results, comparing sources, identifying gaps. Adds research strategy (broad→narrow, cross-reference) and quality bar (specific code, specific versions, cite sources).
- **Added: Benchmark harness** — `python -m ultres.eval --benchmark humaneval --n 20` runs HumanEval (pass@1) and `--benchmark mmlu` runs MMLU (accuracy). Fully automatic scoring. Downloads datasets to `.ultres/eval_cache/`.
- **Added: Query diversification** — Near-duplicate queries (>90% word overlap) are removed before searching. Near-duplicate URLs (same domain, >85% title similarity) are deduped after search. Reduces wasted fetches by ~30-40%.
- **Added: Depth-aware gap detection** — Coverage map now flags shallow clusters (<3 pages or <2 code blocks) as [SHALLOW]. Includes sample page titles. Gap detector prompt explicitly asks to fill shallow clusters.
- **Added: Adaptive max_tokens** — `estimate_complexity()` counts file/component references in the plan. Simple=4096, medium=8192, complex=16384 tokens. Prevents truncation on complex tasks and saves time on simple ones.

**Base model:** Qwen3.8-27B-UD-IQ2_XXS (6.77 GB, Apache 2.0, 64K context with q4_0 KV cache).
27B dense model with hybrid attention (16/64 full attn + 48 Gated DeltaNet), native tool calling, thinking mode, vision-language support.
~7.9 tokens/sec on RTX 4060 Ti 8GB (50/65 GPU layers, flash attn, q4_0 KV).

**Roadmap:**
- v1.2 (done): Deep research pipeline + 64K context + two-pass + streaming + gap detection
- v1.4 (done): Bug fixes, optimization, v1.5 readiness (unified trajectories, export, validation)
- v1.5 (done): Qwen3.8-27B single-model upgrade + clustering fix + hybrid attention
- v1.6 (done): 10 pipeline improvements for smarter research/coding + benchmark harness
- v1.5+ (next): QLoRA-specialize on accumulated UltRes trajectories → UltRes-Base-27B (Kaggle free T4)
- v2: full fine-tune + YaRN long-context extension (64K→256K) → UltRes-Base-27B-Long (cloud GPU credits)

## Build & install

```bash
pip install -e .            # install package + ultres CLI
pip install -e ".[dev]"     # + pytest, pytest-asyncio

# GPU inference (NVIDIA CUDA 12.5, no compiler needed):
# v1.5: cu125 wheel required — cu124 crashes with STATUS_ILLEGAL_INSTRUCTION.
pip install llama-cpp-python==0.3.35 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125 --force-reinstall --no-deps
pip install nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-nvrtc-cu12==12.4.127

# CPU inference (fallback, no GPU needed):
pip install llama-cpp-python==0.3.35 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --force-reinstall --no-deps

playwright install chromium # for the page fetcher
```

## Test

```bash
pytest                      # unit tests in tests/ (92 tests, no GPU/model needed)
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

# 2. Pull the model (~6.77 GB, single GGUF):
ultres models pull              # Qwen3.8-27B-UD-IQ2_XXS (default, single model)

# 3. Run a deep research query (default, 2000+ pages, ~90-120 min with 27B):
ultres "build me a complex C++ calculator app"

# 4. Fast mode (v1.1 agentic loop, 30 steps, ~5-10 min with 27B):
ultres --fast "What is C++ std::vector?"

# 5. Options:
ultres --max-pages 500 "query"     # Limit crawl to 500 pages
ultres --no-critique "query"       # Skip self-critique
ultres --no-stream "query"         # Disable live streaming
ultres --model qwen25-7b "query"   # Use legacy fast 7B model instead
```

Requires a running SearXNG instance (see README) or a Tavily/Brave API key.

## Verified working

- **92/92 tests pass** on Python 3.12.8, Windows 11.
- **GPU**: RTX 4060 Ti 8GB, CUDA 12.5 wheel (cu125), **7.9 tokens/sec** (27B), 43.6 tokens/sec (legacy 7B).
- **Model**: Qwen3.8-27B-UD-IQ2_XXS (6.77 GB, single GGUF, 64K with q4_0 KV cache).
- **Architecture**: Hybrid attention (16/64 full attn + 48 Gated DeltaNet) — supported in llama-cpp-python 0.3.35.
- **Context**: 64K with q4_0 KV cache + flash attention (fits 8GB VRAM with 50/65 GPU layers).
- **Native tool calling**: Qwen3.8-27B uses `tools=` API parameter.
- **Thinking mode**: Qwen3.8 generates `thinking` reasoning before answers (stripped from final output).
- **Search**: SearXNG in Docker with JSON format + limiter disabled.
- **Deep pipeline**: 10-stage pipeline with bulk crawl, clustering, gap detection, two-pass implementation.
- **Single-model mode**: Qwen3.8-27B handles all stages (plan, summarize, implement, critique) — no model swap.
- **httpx-first fetch**: 10x faster than Playwright for bulk crawl, with Playwright fallback.
- **Disk persistence**: Deep mode pages ingested into KnowledgeStore + VectorIndex.
- **Live streaming**: Per-stage timers, progress bars, token-by-token streaming.
- **Self-critique**: After implementation, answer is critiqued against research (extracts code blocks only).
- **Trajectories**: Unified format (fast + deep), saved to `.ultres/trajectories/` for v1.5 QLoRA.
- **Trajectory export**: `ultres trajectories export --output train.jsonl` produces training-ready JSONL.

## Architecture (one-paragraph)

**Deep mode (default):** A user query goes to the **Planner** (Qwen3.8-27B
generates 30-50 subtasks × 3-5 variations = 90-250 search queries). The **Bulk
Crawler** fetches 2000+ pages in parallel with quality filtering and link
following. Pages are **clustered** into 50-200 topic clusters. **Gap detection**
identifies missing topics and re-crawls. Clusters are **batch summarized** and
hierarchically compressed into a **master brief** (~5000 words). **Two-pass
implementation**: Qwen3.8-27B writes a plan from the brief, then implements from
the plan (forced to follow research) — same model for both passes, no swap.
**Self-critique** checks the implementation against research. Trajectory saved
for v1.5 QLoRA. All stages show **live streaming** with per-stage timers.

**Fast mode (`--fast`):** The v1.1 agentic loop — model picks tool calls
(`search`/`visit`/`recall`/`load_*`/`finish`) via native function calling.
30 steps, ~2 min, touches ~10 pages.

## Key constraints

- **v1.5 ships no training.** The `lora/` module is a loader stub + trajectory accumulator + export. Training is v1.5+.
- **v1.5 uses Qwen3.8-27B-UD-IQ2_XXS** (64K with q4_0 KV cache, native tool calling, thinking mode). Single model for all stages — no swap.
- **64K context with q4_0 KV cache + flash attention** (verified). Native context is 262K but 64K is the practical limit for 8GB VRAM.
- **GPU requires CUDA 12.5 wheel** (v0.3.35, cu125). cu124 crashes with STATUS_ILLEGAL_INSTRUCTION. CPU wheel works as fallback.
- **llama-cpp-python 0.3.35 required** for Qwen3.8 hybrid attention (Gated DeltaNet). Older versions don't support the architecture.
- **Deep pipeline takes 90-120 min** per query with 27B (vs 40-60 min with legacy 7B). Use `--fast` for quick queries. Pipeline timeout default 90 min.
- **Single-model mode**: When `spec.params_b > 10`, `ModelManager.is_single_model` returns True and `load_coder()` returns the same model. Legacy 7B models preserve dual-model swap.
- **Thinking mode enabled by default**. Qwen3.8 generates `thinking` reasoning before answers. Stripped from final output via `_strip_thinking()`. Disable via `cfg.model.enable_thinking = False`.
- **httpx-first fetch**: Bulk crawl uses httpx (10x faster), falls back to Playwright for JS-heavy sites.
- **Per-domain rate limiting**: Max 3 concurrent fetches per domain to avoid hammering sites.
- **KV-cache reuse is best-effort.** v2 wires in real persistence.
- **Self-critique adds latency**. Disable via `--no-critique`.

## v1.5+ training (Kaggle free T4×2)

Goal: produce `UltRes-Base-27B` via QLoRA on UltRes agent trajectories.

1. **Trajectories accumulate automatically** in `.ultres/trajectories/` as
   JSONL files (unified v1.4 format: query → mode → plan → answer → brief →
   pages → clusters → gaps → visited_urls → doc_ids → code_ids → timing).
   Run v1.5 on a curated set of coding/research tasks to build the dataset.
   Target ~5-10K examples.
2. **Export training data**: `ultres trajectories export --output train.jsonl`
   produces instruction/response pairs with research context. Use
   `ultres trajectories stats` to check dataset size and
   `ultres trajectories validate` to check quality.
3. **Upload as a Kaggle dataset** (CPU session, internet on).
4. **Train on Kaggle T4×2** (GPU session, internet off):
   - Base: `unsloth/qwen3.8-27b-bnb-4bit` (general reasoning model)
   - LoRA rank 32, alpha 64, all linear layers
   - Max seq len 4096, batch 2, grad accum 4, 3 epochs, LR 2e-4 cosine
   - Use Unsloth's official Kaggle T4×2 notebooks as the template.
5. **Merge + convert to GGUF** (UD-IQ2_XXS, Q4_K_M, Q8_0) via llama.cpp.
6. **Publish** to HuggingFace as `ultres/UltRes-Base-27B-GGUF`.
7. **Pin** the repo URL + sha256 in `ultres/models/registry.py`.

## v2 training (Vultr $250 free A100 80GB, ~68 hrs)

Goal: `UltRes-Base-27B-Long` via full fine-tune + YaRN 64K→256K, plus
per-project LoRA cache training.

1. **Dry-run the pipeline on Kaggle T4 first** (smaller scale) to avoid
   wasting A100 hours on bugs.
2. **Full fine-tune** on expanded UltRes corpus (~8-16 hrs on A100 80GB).
3. **YaRN extension:** apply 4x rope-scaling, fine-tune on 64K-128K sequences
   using Unsloth's long-context gradient checkpointing (~20-40 hrs).
4. **Per-project LoRA:** QLoRA-train small adapters on user-revisited topics
   (feasible on the user's 4060 Ti 8GB at seq 2K, or on Kaggle).
5. **Publish** `ultres/UltRes-Base-27B-Long-GGUF` + pin in registry.

## Files to touch for common changes

| Change | Files |
|---|---|
| Add a search backend | `ultres/search/base.py` (Protocol), new provider file, `ultres/search/__init__`, `ultres/cli.py` (provider construction in `run`) |
| Add a tool | `ultres/agent/tools.py` (schema + system prompt), `ultres/agent/research_loop.py` (`ToolExecutor.execute` + `_extract_tool_call`) |
| Change hot-window eviction | `ultres/agent/research_loop.py` (`HotWindow._evict`) |
| Add a model | `ultres/models/registry.py` (`_REGISTRY`), optionally `ultres/models/loader.py` |
| Change store layout | `ultres/memory/store.py` (all paths derived from `qdir`) |
