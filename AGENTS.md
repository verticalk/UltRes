# AGENTS.md

Guidance for AI agents (and humans) working on the UltRes codebase.

**Repo:** https://github.com/verticalk/UltRes (private)

> **Keep this file up to date.** If you make changes to the codebase — new
> features, fixed bugs, changed dependencies, updated configs, new verification
> results, roadmap progress — update the relevant sections of AGENTS.md in the
> same commit. Do not let this file go stale. It is the single source of truth
> for project status, build/test commands, and architecture decisions.

## Project status

**v1.1 — GPU-accelerated with native tool calling and self-critique.**

| Milestone | Status |
|---|---|
| Package scaffold + CLI | Done |
| Config (Pydantic + TOML) | Done |
| Model registry + loader (GGUF + llama.cpp) | Done |
| Search layer (SearXNG / Tavily / Brave + fetch) | Done |
| Memory layer (store / vector / hierarchical / KV cache) | Done |
| Agent layer (tools / planner / research loop / reasoner) | Done |
| LoRA cache stub + trajectory accumulation (v1.5/v2) | Done |
| GPU acceleration (CUDA 12.1 wheel, RTX 4060 Ti) | Done — 43.6 tokens/sec |
| Native tool calling (Qwen2.5-7B-Instruct) | Done |
| Synthesis pass (evidence-cited answers) | Done |
| Self-critique loop (reasoning about reasoning) | Done |
| Trajectory saving for v1.5 QLoRA | Done |
| Tests (32 unit tests) | Done — all passing |
| End-to-end pipeline verified | Done |
| Git repo + pushed to GitHub | Done |

**v1.1 changes from v1:**
- **GPU**: Switched from CPU wheel to CUDA 12.1 wheel. 43.6 tokens/sec (was ~2-3 on CPU).
- **Model**: Switched from Qwen2.5-Coder-7B to Qwen2.5-7B-Instruct (general reasoning + native tool calling). Coder model kept as `coder` registry entry.
- **Native tool calling**: Tools passed via `tools=` API parameter instead of prompt-based JSON. Fallback JSON parser kept for robustness.
- **Synthesis pass**: Before `finish`, model is forced to load recalled content and write an evidence-cited answer.
- **Self-critique**: After `finish`, answer is critiqued against evidence. If gaps found, re-researches and re-synthesizes.
- **Trajectory accumulation**: Every query saves (query, steps, answer, doc_ids, code_ids) to `.ultres/trajectories/` for v1.5 QLoRA training.

**Base model:** Qwen2.5-7B-Instruct Q4_K_M (4.7 GB, Apache 2.0, 32K context).
General-purpose instruct model with native function-calling support.
Note: Qwen2.5-Coder-7B is also available as `ultres --model coder` for code-specific tasks.

**Roadmap:**
- v1.1 (done): GPU + Qwen2.5-7B-Instruct + native tool calling + self-critique + trajectories
- v1.5 (next): QLoRA-specialize on accumulated UltRes trajectories → UltRes-Base-7B (Kaggle free T4)
- v2: full fine-tune + YaRN long-context extension (32K→256K) → UltRes-Base-7B-Long (cloud GPU credits)

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
pytest                      # unit tests in tests/ (32 tests, no GPU/model needed)
```

Tests cover: store CRUD, hierarchical summary tree, vector recall, tool-call
JSON parsing (native + fallback), config TOML round-trip, LoRA cache stub,
trajectory save/load. They do NOT require a GPU, downloaded model, or network access.

## Run

```bash
# 1. Start SearXNG (one-time):
docker run -d --name searxng -p 8080:8080 -v ./searxng_settings.yml:/etc/searxng/settings.yml:ro searxng/searxng

# 2. Pull the model (~4.7 GB, multi-part GGUF):
ultres models pull

# 3. Run a query (GPU-accelerated):
ultres "build me a complex C++ calculator app"

# Code-specific query (uses Qwen2.5-Coder-7B):
ultres --model coder "optimize this C++ function"
```

Requires a running SearXNG instance (see README) or a Tavily/Brave API key.

## Verified working

- **32/32 tests pass** on Python 3.12.8, Windows 11.
- **GPU**: RTX 4060 Ti 8GB, CUDA 12.1 wheel, **43.6 tokens/sec** (was ~2-3 on CPU).
- **Model**: Qwen2.5-7B-Instruct Q4_K_M (4.7 GB, multi-part GGUF) — native tool calling confirmed.
- **Native tool calling**: Qwen2.5-7B-Instruct uses `tools=` API parameter (not prompt-based JSON).
- **Search**: SearXNG in Docker with JSON format + limiter disabled.
- **Fetch**: trafilatura + httpx extraction works (fixed `bare_extraction` returning Document object instead of dict in trafilatura 2.x).
- **Self-critique**: After `finish`, answer is critiqued against evidence; gaps trigger re-research.
- **Trajectories**: Saved to `.ultres/trajectories/` for v1.5 QLoRA training.

## Architecture (one-paragraph)

A user query goes to the **Planner** (one model call → research subtasks), then
the **Research Loop** (model picks tool calls via native function calling:
`search`/`visit`/`recall`/`load_summary`/`load_slice`/`load_code`/`finish`).
Each `visit` fetches a page, extracts text + code blocks, and ingests them into
the **Knowledge Store** (disk: raw/notes/summaries/code) + **Vector Index**
(Chroma). The model only sees compact confirmations inline; it pulls full
content via `load_*` tools as needed. A **HotWindow** evicts old messages when
the soft token budget is exceeded, keeping the model within its 32K native
window while the disk store holds everything. On `finish`, a **Synthesis Pass**
forces the model to load recalled content and write an evidence-cited answer.
Then a **Self-Critique Loop** checks the answer against evidence and re-researches
gaps. Finally, the trajectory is saved to `.ultres/trajectories/` for v1.5 QLoRA.

## Key constraints

- **v1.1 ships no training.** The `lora/` module is a loader stub + trajectory accumulator. Training is v1.5.
- **v1.1 uses Qwen2.5-7B-Instruct** (32K native ctx, native tool calling). `coder` selection uses Qwen2.5-Coder-7B. `ultres-base` and `ultres-base-long` point at not-yet-published checkpoints — they will 404 until v1.5/v2.
- **GPU requires CUDA 12.1 wheel** (v0.3.4). The cu124 wheel (v0.3.35) has an illegal instruction issue on this CPU. CPU wheel works as fallback.
- **KV-cache reuse is best-effort.** `memory/kv_cache.py` tracks attended chunks but does not persist tensors. Falls back to re-inference. v2 wires in real persistence.
- **Hot window budget defaults to 28K** (under the 32K native limit) to leave headroom for generation. v2 raises this when the native window becomes 256K.
- **Self-critique adds latency** (1 extra model call per round, default 1 round). Disable via `enable_self_critique=false` in config.

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
