# AGENTS.md

Guidance for AI agents (and humans) working on the UltRes codebase.

## Build & install

```bash
pip install -e .            # install package + ultres CLI
pip install -e ".[dev]"     # + pytest, pytest-asyncio
pip install -e ".[inference]"  # + llama-cpp-python (needs C++ toolchain OR use prebuilt wheel)
# Prebuilt wheel (no compiler needed):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
playwright install chromium # for the page fetcher
```

## Test

```bash
pytest                      # unit tests in tests/ (31 tests, no GPU/model needed)
```

Tests cover: store CRUD, hierarchical summary tree, vector recall, tool-call
JSON parsing, config TOML round-trip, LoRA cache stub. They do NOT require a
GPU, downloaded model, or network access.

## Run

```bash
# 1. Start SearXNG (one-time):
docker run -d --name searxng -p 8080:8080 -v ./searxng_settings.yml:/etc/searxng/settings.yml:ro searxng/searxng

# 2. Pull the model (~4.7 GB):
ultres models pull

# 3. Run a query:
ultres "build me a complex C++ calculator app"
```

Requires a running SearXNG instance (see README) or a Tavily/Brave API key.

## Verified working

- **31/31 tests pass** on Python 3.12.8, Windows 11, no GPU.
- **End-to-end pipeline verified**: `ultres "What is the difference between C++ std::vector and std::list?"` successfully searched SearXNG, visited 2 cppreference.com pages, extracted 8 code blocks + 2 docs into the knowledge store, and produced a research-backed answer in 11 steps.
- **Model**: Qwen2.5-Coder-7B-Instruct Q4_K_M (4.7 GB) — confirmed working with llama-cpp-python 0.3.35 (prebuilt CPU wheel).
- **Search**: SearXNG in Docker with JSON format + limiter disabled.
- **Fetch**: trafilatura + httpx extraction works (fixed `bare_extraction` returning Document object instead of dict in trafilatura 2.x).

## Architecture (one-paragraph)

A user query goes to the **Planner** (one model call → research subtasks), then
the **Research Loop** (model picks tool calls: `search`/`visit`/`recall`/
`load_summary`/`load_slice`/`load_code`/`finish`). Each `visit` fetches a page,
extracts text + code blocks, and ingests them into the **Knowledge Store**
(disk: raw/notes/summaries/code) + **Vector Index** (Chroma). The model only
sees compact confirmations inline; it pulls full content via `load_*` tools as
needed. A **HotWindow** evicts old messages when the soft token budget is
exceeded, keeping the model within its 32K (v1) / 256K (v2) native window while
the disk store holds everything. On `finish`, the answer is streamed and saved.

## Key constraints

- **v1 ships no training.** The `lora/` module is a loader stub. Training is v2.
- **v1 uses the stock Qwen2.5-Coder-7B** (32K native ctx). `ultres-base` and
  `ultres-base-long` model selections exist in the registry but point at
  not-yet-published checkpoints — they will 404 until v1.5/v2.
- **KV-cache reuse is best-effort in v1.** `memory/kv_cache.py` tracks attended
  chunks but does not persist tensors (llama-cpp-python's stable API doesn't
  expose this cleanly yet). Falls back to re-inference. v2 wires in real
  persistence.
- **Hot window budget defaults to 28K** (under the 32K native limit) to leave
  headroom for the model's generation. v2 raises this when the native window
  becomes 256K.

## v1.5 training (Kaggle free T4×2)

Goal: produce `UltRes-Base-7B` via QLoRA on UltRes agent trajectories.

1. **Generate trajectories locally** (no GPU): run v1's agent loop on a curated
   set of coding/research tasks; record (query → plan → tool calls → answer)
   pairs in Qwen3 chat format. Target ~5-10K examples.
2. **Upload as a Kaggle dataset** (CPU session, internet on).
3. **Train on Kaggle T4×2** (GPU session, internet off):
   - Base: `unsloth/qwen2.5-coder-7b-instruct-bnb-4bit`
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
