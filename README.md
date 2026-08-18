# UltRes

**UltRes** is a lightweight, locally-runnable AI that performs like a flagship model on a user's specific request by aggressively researching the web, building a disk-backed knowledge store for that exact task, and reasoning over it with a small agentic model — cheap, private, and dynamic.

The base model is a custom fine-tuned + long-context-extended derivative of [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) (Apache 2.0), built using only free GPU compute. v1.1 ships as a wrapper around the stock model with GPU acceleration, native tool calling, and self-critique; v1.5/v2 add the custom UltRes checkpoints trained on accumulated agent trajectories.

## Why UltRes

The core idea: instead of training a giant model on everything, use a small model that **researches the specific task on demand** and stores what it learns in a disk-backed knowledge store it can navigate via tool calls. This gives **effectively unlimited logical context** bounded only by disk — not by any fixed token limit — while running on a consumer laptop.

This pattern is proven in 2026 by projects like [DR-Venus-4B](https://github.com/inclusionAI/DR-Venus) and [AgentCPM-Explore-4B](https://github.com/OpenBMB/AgentCPM), small on-device deep-research agents that entered GAIA/BrowseComp/HLE. UltRes wraps the same pattern around a 7B instruct model with native tool calling.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/verticalk/UltRes
cd UltRes
pip install -e .
playwright install chromium
```

### GPU acceleration (NVIDIA, recommended)

```bash
pip install llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --force-reinstall --no-deps
pip install nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-nvrtc-cu12==12.4.127
```

This gives ~40+ tokens/sec on an RTX 4060 Ti 8GB (vs ~2-3 on CPU).

## Search backend setup

UltRes supports three search backends. The default is **SearXNG** (local, free, no API key).

### SearXNG (default, free)

Run a local SearXNG instance with JSON output enabled:

```bash
docker run -d --name searxng -p 8080:8080 \
    -e SEARXNG_BASE_URL=http://localhost:8080 \
    -v ./searxng:/etc/searxng \
    searxng/searxng
```

Then enable JSON output in `./searxng/settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

### Tavily (paid, higher quality)

```bash
export ULTRES_SEARCH_BACKEND=tavily
export ULTRES_SEARCH_API_KEY=tvly-xxxxxxxx
```

### Brave (paid)

```bash
export ULTRES_SEARCH_BACKEND=brave
export ULTRES_SEARCH_API_KEY=BSAxxxxxxxx
```

## Usage

```bash
# Pull the model first (~4.7 GB download for Q4_K_M, multi-part GGUF)
ultres models pull

# Run a query (GPU-accelerated, native tool calling, self-critique)
ultres "build me a complex C++ calculator app"

# Deep research (up to 100 steps)
ultres --deep "explain how LLVM's instruction selection works"

# Use the code-focused model for code-specific tasks
ultres --model coder "optimize this C++ function"

# Use a specific search backend
ultres --search tavily "compare Rust async runtimes"

# List past queries
ultres list

# Show a past answer + research tree
ultres show <query-id>

# Edit config
ultres config

# Clean all stored data (keeps models + config)
ultres clean
```

## Hardware requirements (v1.1)

- **CPU:** modern multi-core (tested on i7-13700K)
- **RAM:** 16 GB minimum (model is ~4.7 GB at Q4_K_M + OS + KV cache)
- **GPU:** NVIDIA with 8GB+ VRAM recommended. RTX 4060 Ti 8GB fully offloads the 7B Q4_K_M for ~40+ tok/s. CPU-only works at ~2-3 tok/s.
- **Disk:** ~5 GB for the model + 50-200 MB per deep research query

## How "effectively unlimited context" works

The 32K native context window (v1) / 256K (v2) is only the **hot working area**. The real context is the **disk-backed Knowledge Store**, which the model navigates via tool calls:

| Tier | Where | What |
|---|---|---|
| Hot | GPU KV cache (32K/256K) | current reasoning slice: planner output + relevant chunks + working notes |
| Warm | Chroma vector index (disk) | embeddings of every retrieved page's notes + summaries; `recall(query)` pulls slices back into hot |
| Cold | disk (raw pages) | full raw markdown of every fetched page; `load_slice(doc_id, section)` drills in |

Plus:
- **Hierarchical summaries** — each doc → extractive notes → subtopic summary → topic summary. `load_summary(topic)` gives a compressed view.
- **Verbatim code preservation** — code blocks are *never* summarized; stored as first-class objects, retrieved via `load_code(code_id)`.
- **KV-cache reuse** — (v2) KV tensors for retrieved chunks are cached to disk and paged back in on re-attention.

This is the [MemGPT/Letta](https://github.com/cpacker/MemGPT) pattern + the DR-Venus agent loop + disk KV paging. To the user it feels like the model "knows everything it researched"; physically, only a slice is in attention at once.

## Roadmap

| Phase | What | Compute |
|---|---|---|
| **v1.1** (this release) | GPU-accelerated agent wrapper around stock Qwen2.5-7B-Instruct + native tool calling + self-critique + trajectory accumulation | Your hardware (no training) |
| **v1.5** | QLoRA-specialize Qwen2.5-7B-Instruct on accumulated UltRes agent trajectories → `UltRes-Base-7B` | Kaggle free T4×2 (30 hrs/week) |
| **v2** | Full fine-tune + YaRN long-context extension (32K→256K) → `UltRes-Base-7B-Long` + per-project LoRA cache training | Vultr $250 free A100 80GB credits (~68 hrs) |
| Future | Continued pretraining; UltRes Lite (4B); shared adapter hub | NVIDIA Inception credits (optional) |

See [the full plan](.devin/plans/) for architecture details and verification criteria.

## Project layout

```
ultres/
  cli.py                 # Typer CLI entrypoint
  config.py              # Pydantic settings, TOML I/O
  models/
    loader.py            # GGUF download + llama.cpp init
    registry.py          # Model URLs + quant presets
  search/
    base.py              # SearchProvider interface + data types
    searxng.py           # Local SearXNG provider
    tavily.py            # Tavily API provider
    brave.py             # Brave API provider
    fetch.py             # Playwright + trafilatura page extraction
  memory/
    store.py             # Disk knowledge store (raw/notes/summaries/code)
    vector.py            # Chroma in-process index
    kv_cache.py          # KV cache reuse wrapper (v2: real persistence)
    hierarchical.py      # Summary tree builder
  agent/
    tools.py             # Tool schemas (native function calling)
    planner.py           # Query decomposition
    research_loop.py     # Agentic research loop + synthesis + self-critique
    reasoner.py          # Final answer synthesis
  lora/
    cache.py             # Per-project adapter loader + trajectory accumulation
```

## License

Apache 2.0 (matches the Qwen2.5 base model license).
