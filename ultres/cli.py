"""UltRes CLI (Typer).

Commands:
    ultres "<query>"                Run the deep research pipeline (default).
    ultres --fast "<query>"         Run the fast agentic loop (30 steps).
    ultres models pull              Download the active model.
    ultres models list              List supported models.
    ultres config                   Show / edit config.
    ultres list                     List past queries.
    ultres show <query-id>          Print a past answer + research tree.
    ultres trajectories list        List saved trajectories.
    ultres trajectories stats       Show trajectory statistics.
    ultres trajectories export      Export training-ready JSONL.
    ultres trajectories validate    Check trajectory completeness.
    ultres clean                    Remove all stored data (keeps models + config).

Flags:
    --fast                          Use fast agentic loop instead of deep pipeline.
    --search searxng|tavily|brave   Override search backend for this run.
    --model stock|coder|ultres-base|...   Override model selection for this run.
    --max-pages N                   Max pages to crawl (deep mode, default 2000).
    --no-critique                   Skip self-critique pass.
    --no-stream                     Disable live streaming output.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError with
# rich's Unicode characters (▶, ✓, █, ░, etc.).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from ultres.config import UltResConfig
from ultres.models.loader import detect_ram_gb, detect_vram_gb, download_model, ensure_model, resolve_spec
from ultres.models.registry import list_specs
from ultres import __version__

app = typer.Typer(
    help="UltRes — lightweight self-researching local LLM.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Model management.")
app.add_typer(models_app, name="models")
trajectories_app = typer.Typer(help="Trajectory management for v1.5 QLoRA training.")
app.add_typer(trajectories_app, name="trajectories")

# Force UTF-8 console to avoid cp1252 encoding errors on Windows.
console = Console(force_terminal=True, legacy_windows=False) if sys.platform == "win32" else Console()

# Known subcommand names — if the first CLI arg is NOT one of these, treat all
# args as a bare query and route to `run`.
_KNOWN_SUBCOMMANDS = {"run", "models", "config", "list", "show", "clean", "trajectories"}


def _maybe_redirect_to_run():
    """If the first positional arg isn't a known subcommand, rewrite sys.argv
    to route through the `run` subcommand so `ultres "my query"` works."""
    # Skip program name; collect positional args (not options).
    args = sys.argv[1:]
    # Find the first positional arg (not starting with -).
    first_pos = None
    for a in args:
        if not a.startswith("-"):
            first_pos = a
            break
    if first_pos is not None and first_pos not in _KNOWN_SUBCOMMANDS:
        # Insert "run" before the args so Typer sees: ultres run "my query" ...
        sys.argv = [sys.argv[0], "run"] + args


# ---------------------------------------------------------------------------
# Main: run a query
# ---------------------------------------------------------------------------

@app.command()
def run(
    query: str = typer.Argument(..., help='The query, e.g. "build me a complex C++ calculator app"'),
    fast: bool = typer.Option(False, "--fast", help="Use fast agentic loop (30 steps) instead of deep pipeline"),
    search: Optional[str] = typer.Option(None, "--search", help="searxng|tavily|brave"),
    model: Optional[str] = typer.Option(None, "--model", help="stock|coder|ultres-base|ultres-base-long"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", help="Max pages to crawl (deep mode)"),
    no_critique: bool = typer.Option(False, "--no-critique", help="Skip self-critique"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable live streaming"),
    lite: bool = typer.Option(False, "--lite", help="(reserved, future 4B tier)"),
):
    """Run the full UltRes research pipeline for a query."""
    if lite:
        console.print("[yellow]--lite is reserved for a future 4B tier; ignoring.[/yellow]")

    cfg = UltResConfig.load()
    cfg.ensure_dirs()

    # Apply CLI overrides.
    if search:
        cfg.search.backend = search  # type: ignore[assignment]
    if model:
        cfg.model.selection = model  # type: ignore[assignment]
    if max_pages:
        cfg.deep_research.max_pages = max_pages
    if no_critique:
        cfg.agent.enable_self_critique = False
    if no_stream:
        cfg.deep_research.enable_streaming = False

    # Hardware report.
    ram = detect_ram_gb()
    vram = detect_vram_gb()
    spec = resolve_spec(cfg)
    mode = "fast agentic" if fast else "deep research"
    console.print(
        Panel(
            f"[bold]UltRes v{__version__}[/bold]\n"
            f"Mode: {mode}\n"
            f"Model: {spec.label} ({cfg.model.quant})\n"
            f"Context: {cfg.model.extended_ctx if cfg.model.use_extended_context else spec.native_ctx} tokens\n"
            f"Search: {cfg.search.backend}\n"
            f"RAM: {ram:.1f} GB | VRAM: {vram:.1f} GB\n"
            + (f"Max pages: {cfg.deep_research.max_pages}\n" if not fast else f"Max steps: {cfg.agent.max_steps}\n")
            + f"Streaming: {'on' if cfg.deep_research.enable_streaming else 'off'}",
            title="Configuration",
            border_style="cyan",
        )
    )

    # Ensure model is present.
    console.print("[cyan]Ensuring model is downloaded...[/cyan]")
    model_path = ensure_model(cfg)
    console.print(f"[green]Model:[/green] {model_path}")

    # Lazy import to avoid loading llama-cpp-python on every command.
    from ultres.models.loader import ModelManager
    from ultres.search.base import get_provider

    # Provider.
    if cfg.search.backend == "searxng":
        provider = get_provider("searxng", base_url=cfg.search.searxng_base_url)
    elif cfg.search.backend == "tavily":
        provider = get_provider("tavily", api_key=cfg.search.tavily_api_key)
    elif cfg.search.backend == "brave":
        provider = get_provider("brave", api_key=cfg.search.brave_api_key)
    else:
        raise typer.BadParameter(f"Unknown search backend: {cfg.search.backend}")

    if fast:
        # --- Fast mode: use the v1.1 agentic loop ---
        from ultres.agent.research_loop import run_research_loop
        from ultres.agent.reasoner import stream_answer

        console.print("[cyan]Loading model into llama.cpp...[/cyan]")
        model_mgr = ModelManager(cfg)
        llm = model_mgr.load_instruct()
        console.print("[green]Model loaded.[/green]")

        result = asyncio.run(run_research_loop(llm, cfg, query, provider=provider, console=console))

        answer_path = cfg.answers_dir / f"{result.query_id}.md"
        stream_answer(result.answer, answer_path, console=console)
        console.print(
            f"\n[dim]query_id={result.query_id} | steps={result.steps} | "
            f"sources={len(result.visited_urls)}[/dim]"
        )
        model_mgr.unload_current()
    else:
        # --- Deep mode: v1.4 deep research pipeline ---
        from ultres.research.pipeline import run_deep_research
        from ultres.streaming import ResearchStreamer

        model_mgr = ModelManager(cfg)

        streamer = ResearchStreamer(console=console, enabled=cfg.deep_research.enable_streaming)

        result = asyncio.run(run_deep_research(
            model_mgr=model_mgr,
            cfg=cfg,
            user_query=query,
            provider=provider,
            console=console,
            streamer=streamer,
        ))

        # For deep mode, the answer is already saved by the pipeline.
        console.print(
            f"\n[dim]query_id={result.query_id} | pages={result.pages_crawled} | "
            f"clusters={result.clusters_found} | sources={len(result.visited_urls)}[/dim]"
        )
        model_mgr.unload_current()


# Default command: `ultres "query"` (no subcommand).
# We intercept sys.argv in _maybe_redirect_to_run() (called at import time)
# to route bare queries through the `run` subcommand.
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
):
    if ctx.invoked_subcommand is not None:
        return
    console.print("[red]Please provide a query:[/red] ultres \"your query\"")
    console.print("[dim]Or use a subcommand: ultres models pull, ultres config, etc.[/dim]")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# models subcommands
# ---------------------------------------------------------------------------

@models_app.command("pull")
def models_pull(
    model: Optional[str] = typer.Option(None, "--model", help="stock|ultres-base|ultres-base-long"),
    quant: Optional[str] = typer.Option(None, "--quant", help="Quantization, e.g. Q4_K_M"),
):
    """Download the active (or specified) model."""
    cfg = UltResConfig.load()
    if model:
        cfg.model.selection = model  # type: ignore[assignment]
    if quant:
        cfg.model.quant = quant
    spec = resolve_spec(cfg)
    console.print(f"[cyan]Pulling[/cyan] {spec.label} ({cfg.model.quant})")
    path = download_model(spec, cfg.model.quant, cfg.model_cache_dir)
    console.print(f"[green]Done:[/green] {path}")


@models_app.command("list")
def models_list():
    """List supported models."""
    table = Table(title="Supported models")
    table.add_column("key", style="cyan")
    table.add_column("label")
    table.add_column("params", justify="right")
    table.add_column("native ctx", justify="right")
    table.add_column("ext ctx", justify="right")
    table.add_column("default quant")
    for key, spec in list_specs().items():
        ext_ctx = spec.extended_ctx if hasattr(spec, "extended_ctx") and spec.extended_ctx else "-"
        table.add_row(
            key,
            spec.label,
            f"{spec.params_b}B",
            f"{spec.native_ctx:,}",
            f"{ext_ctx:,}" if isinstance(ext_ctx, int) else ext_ctx,
            spec.default_quant,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@app.command("config")
def config_cmd():
    """Show current config and open an interactive editor for key fields."""
    cfg = UltResConfig.load()
    console.print(Panel(json.dumps(cfg.model_dump(mode="json"), indent=2), title="Current config"))

    field = Prompt.ask(
        "Edit field",
        choices=["search.backend", "search.searxng_base_url", "search.tavily_api_key",
                 "search.brave_api_key", "model.selection", "model.quant",
                 "agent.max_steps", "agent.hot_window_token_budget",
                 "deep_research.max_pages", "deep_research.crawl_concurrency",
                 "deep_research.enable_two_pass", "deep_research.enable_gap_detection",
                 "deep_research.gap_research_rounds", "deep_research.pipeline_timeout_min",
                 "save", "quit"],
        default="quit",
    )
    if field == "quit":
        return
    if field == "save":
        cfg.save()
        console.print("[green]Saved.[/green]")
        return

    value = Prompt.ask(f"New value for {field}")
    # Navigate dotted path.
    obj = cfg
    parts = field.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    last = parts[-1]
    cur = getattr(obj, last)
    if isinstance(cur, int):
        setattr(obj, last, int(value))
    elif isinstance(cur, float):
        setattr(obj, last, float(value))
    else:
        setattr(obj, last, value)
    cfg.save()
    console.print(f"[green]Updated {field} -> {value}[/green]")


# ---------------------------------------------------------------------------
# list / show / clean
# ---------------------------------------------------------------------------

@app.command("list")
def list_queries():
    """List past research queries in this project (fast + deep mode)."""
    cfg = UltResConfig.load()
    table = Table(title="Past queries")
    table.add_column("query_id", style="cyan")
    table.add_column("mode")
    table.add_column("query")
    table.add_column("docs/pages", justify="right")
    table.add_column("code", justify="right")

    # Scan store (fast mode queries).
    seen_ids: set[str] = set()
    if cfg.store_dir.exists():
        for qdir in sorted(cfg.store_dir.iterdir()):
            if not qdir.is_dir():
                continue
            meta_path = qdir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text("utf-8"))
            except json.JSONDecodeError:
                continue
            qid = meta.get("query_id", qdir.name)
            seen_ids.add(qid)
            table.add_row(
                qid,
                "fast",
                (meta.get("user_query", "") or "")[:60],
                str(len(meta.get("docs", {}))),
                str(len(meta.get("code", {}))),
            )

    # Scan trajectories (deep mode queries).
    traj_dir = cfg.ultres_dir / "trajectories"
    if traj_dir.exists():
        for traj_file in sorted(traj_dir.iterdir()):
            if not traj_file.suffix == ".jsonl":
                continue
            qid = traj_file.stem
            if qid in seen_ids:
                continue
            try:
                record = json.loads(traj_file.read_text("utf-8").strip().split("\n")[0])
            except (json.JSONDecodeError, IndexError):
                continue
            mode = record.get("mode", "?")
            pages = record.get("pages_crawled", 0)
            code_count = len(record.get("code_ids", []))
            table.add_row(
                qid,
                mode,
                (record.get("query", "") or "")[:60],
                str(pages),
                str(code_count),
            )

    if table.row_count == 0:
        console.print("[dim]No queries yet.[/dim]")
    else:
        console.print(table)


@app.command("show")
def show_query(query_id: str = typer.Argument(..., help="The query_id to show.")):
    """Print a past answer + research tree (works for fast and deep mode)."""
    cfg = UltResConfig.load()

    # Show answer.
    answer_path = cfg.answers_dir / f"{query_id}.md"
    if answer_path.exists():
        console.print(Panel(answer_path.read_text("utf-8"), title=f"Answer {query_id}", border_style="green"))
    else:
        console.print(f"[red]No answer found for {query_id}[/red]")

    # Show store meta (fast mode).
    meta_path = cfg.store_dir / query_id / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text("utf-8"))
        console.print(Panel(json.dumps(meta, indent=2), title="Research tree (fast mode)", border_style="cyan"))

    # Show trajectory (deep mode).
    traj_path = cfg.ultres_dir / "trajectories" / f"{query_id}.jsonl"
    if traj_path.exists():
        record = json.loads(traj_path.read_text("utf-8").strip().split("\n")[0])
        # Show plan + brief if available.
        if record.get("plan"):
            console.print(Panel(record["plan"][:2000], title="Implementation Plan (Pass 1)", border_style="blue"))
        if record.get("brief"):
            console.print(Panel(record["brief"][:2000], title="Master Research Brief", border_style="magenta"))
        # Show trajectory metadata.
        summary = {k: v for k, v in record.items() if k not in ("answer", "plan", "brief")}
        console.print(Panel(json.dumps(summary, indent=2), title="Trajectory (deep mode)", border_style="cyan"))


@app.command("clean")
def clean():
    """Remove all stored data (keeps config + models)."""
    cfg = UltResConfig.load()
    import shutil
    if cfg.store_dir.exists():
        shutil.rmtree(cfg.store_dir)
        console.print("[green]Cleared store.[/green]")
    if cfg.index_dir.exists():
        shutil.rmtree(cfg.index_dir)
        console.print("[green]Cleared index.[/green]")
    if cfg.answers_dir.exists():
        shutil.rmtree(cfg.answers_dir)
        console.print("[green]Cleared answers.[/green]")
    traj_dir = cfg.ultres_dir / "trajectories"
    if traj_dir.exists():
        shutil.rmtree(traj_dir)
        console.print("[green]Cleared trajectories.[/green]")


# ---------------------------------------------------------------------------
# trajectories subcommands (v1.4: v1.5 QLoRA training prep)
# ---------------------------------------------------------------------------

@trajectories_app.command("list")
def trajectories_list():
    """List all saved trajectories (fast + deep mode)."""
    cfg = UltResConfig.load()
    from ultres.lora.cache import list_trajectories
    trajectories = list_trajectories(cfg.ultres_dir / "trajectories")
    if not trajectories:
        console.print("[dim]No trajectories yet. Run some queries first.[/dim]")
        return
    table = Table(title=f"Trajectories ({len(trajectories)})")
    table.add_column("query_id", style="cyan")
    table.add_column("mode")
    table.add_column("type")
    table.add_column("query")
    table.add_column("pages", justify="right")
    table.add_column("answer chars", justify="right")
    for t in trajectories:
        table.add_row(
            t.get("query_id", "?"),
            t.get("mode", "?"),
            t.get("query_type", "?"),
            (t.get("query", "") or "")[:50],
            str(t.get("pages_crawled", 0) or 0),
            str(len(t.get("answer", "") or "")),
        )
    console.print(table)


@trajectories_app.command("stats")
def trajectories_stats():
    """Show trajectory statistics for v1.5 QLoRA planning."""
    cfg = UltResConfig.load()
    from ultres.lora.cache import trajectory_stats
    stats = trajectory_stats(cfg.ultres_dir / "trajectories")
    if stats["total"] == 0:
        console.print("[dim]No trajectories yet.[/dim]")
        return
    console.print(Panel(
        f"Total trajectories: [bold]{stats['total']}[/bold]\n"
        f"  Fast mode: {stats['by_mode'].get('fast', 0)}\n"
        f"  Deep mode: {stats['by_mode'].get('deep', 0)}\n\n"
        f"By query type:\n"
        f"  Code: {stats['by_query_type'].get('code', 0)}\n"
        f"  General: {stats['by_query_type'].get('general', 0)}\n"
        f"  Mixed: {stats['by_query_type'].get('mixed', 0)}\n\n"
        f"Total pages crawled: {stats['total_pages']}\n"
        f"Avg pages per deep query: {stats['avg_pages_deep']:.1f}\n"
        f"Avg answer length: {stats['avg_answer_chars']:.0f} chars\n"
        f"Total estimated tokens: {stats['total_estimated_tokens']:,}",
        title="Trajectory Statistics",
        border_style="cyan",
    ))


@trajectories_app.command("export")
def trajectories_export(
    output: str = typer.Option("train.jsonl", "--output", "-o", help="Output file path"),
    format: str = typer.Option("jsonl", "--format", "-f", help="jsonl or json"),
):
    """Export trajectories as training-ready JSONL for v1.5 QLoRA."""
    cfg = UltResConfig.load()
    from ultres.lora.cache import export_trajectories
    output_path = Path(output)
    count = export_trajectories(
        cfg.ultres_dir / "trajectories", output_path, format=format,
    )
    if count == 0:
        console.print("[red]No valid trajectories to export.[/red]")
        return
    console.print(f"[green]Exported {count} training examples to {output_path}[/green]")
    console.print(f"[dim]Format: {format} | Ready for v1.5 QLoRA training[/dim]")


@trajectories_app.command("show")
def trajectories_show(
    query_id: str = typer.Argument(..., help="The query_id to show."),
):
    """Show a single trajectory in detail."""
    cfg = UltResConfig.load()
    traj_path = cfg.ultres_dir / "trajectories" / f"{query_id}.jsonl"
    if not traj_path.exists():
        console.print(f"[red]No trajectory found for {query_id}[/red]")
        return
    record = json.loads(traj_path.read_text("utf-8").strip().split("\n")[0])
    # Show key fields.
    for key in ("query_id", "query", "mode", "query_type", "pages_crawled",
                "clusters", "steps", "timestamp"):
        if key in record and record[key] is not None:
            console.print(f"[cyan]{key}:[/cyan] {record[key]}")
    # Show plan (truncated).
    if record.get("plan"):
        console.print(Panel(record["plan"][:2000], title="Plan (Pass 1)", border_style="blue"))
    # Show brief (truncated).
    if record.get("brief"):
        console.print(Panel(record["brief"][:2000], title="Master Brief", border_style="magenta"))
    # Show answer (truncated).
    if record.get("answer"):
        console.print(Panel(record["answer"][:3000], title="Answer", border_style="green"))
    # Show sources.
    if record.get("visited_urls"):
        urls = record["visited_urls"][:20]
        console.print(f"[dim]Sources ({len(record['visited_urls'])} total):[/dim]")
        for u in urls:
            console.print(f"  [dim]- {u}[/dim]")


@trajectories_app.command("validate")
def trajectories_validate():
    """Validate all trajectories for training readiness."""
    cfg = UltResConfig.load()
    from ultres.lora.cache import validate_all_trajectories
    results = validate_all_trajectories(cfg.ultres_dir / "trajectories")
    if not results:
        console.print("[dim]No trajectories to validate.[/dim]")
        return
    valid_count = sum(1 for _, is_valid, _ in results if is_valid)
    console.print(f"[green]{valid_count}/{len(results)} trajectories valid[/green]")
    for qid, is_valid, issues in results:
        if is_valid:
            console.print(f"  [green]✓[/green] {qid}")
        else:
            console.print(f"  [red]✗[/red] {qid}: {'; '.join(issues)}")


# Redirect bare queries to `run` before Typer processes args.
_maybe_redirect_to_run()

if __name__ == "__main__":
    app()
