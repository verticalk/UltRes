"""UltRes CLI (Typer).

Commands:
    ultres "<query>"                Run the full research pipeline.
    ultres models pull              Download the active model.
    ultres models list              List supported models.
    ultres config                   Show / edit config.
    ultres list                     List past queries.
    ultres show <query-id>          Print a past answer + research tree.
    ultres clean                    Remove all stored queries + index.

Flags:
    --deep                          max_steps=100
    --search searxng|tavily|brave   Override search backend for this run.
    --model stock|ultres-base|...   Override model selection for this run.
    --lite                          (reserved for future 4B tier; not in v1)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from ultres.config import UltResConfig
from ultres.models.loader import detect_ram_gb, detect_vram_gb, download_model, ensure_model, resolve_spec
from ultres.models.registry import list_specs

app = typer.Typer(
    help="UltRes — lightweight self-researching local LLM.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Model management.")
app.add_typer(models_app, name="models")

console = Console()

# Known subcommand names — if the first CLI arg is NOT one of these, treat all
# args as a bare query and route to `run`.
_KNOWN_SUBCOMMANDS = {"run", "models", "config", "list", "show", "clean"}


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
    deep: bool = typer.Option(False, "--deep", help="max_steps=100"),
    search: Optional[str] = typer.Option(None, "--search", help="searxng|tavily|brave"),
    model: Optional[str] = typer.Option(None, "--model", help="stock|ultres-base|ultres-base-long"),
    lite: bool = typer.Option(False, "--lite", help="(reserved, future 4B tier)"),
):
    """Run the full UltRes research pipeline for a query."""
    if lite:
        console.print("[yellow]--lite is reserved for a future 4B tier; ignoring in v1.[/yellow]")

    cfg = UltResConfig.load()
    cfg.ensure_dirs()

    # Apply CLI overrides.
    if search:
        cfg.search.backend = search  # type: ignore[assignment]
    if model:
        cfg.model.selection = model  # type: ignore[assignment]
    if deep:
        cfg.agent.max_steps = 100

    # Hardware report.
    ram = detect_ram_gb()
    vram = detect_vram_gb()
    spec = resolve_spec(cfg)
    console.print(
        Panel(
            f"[bold]UltRes v0.1[/bold]\n"
            f"Model: {spec.label} ({cfg.model.quant})\n"
            f"Search: {cfg.search.backend}\n"
            f"RAM: {ram:.1f} GB | VRAM: {vram:.1f} GB\n"
            f"Max steps: {cfg.agent.max_steps} | Hot window: {cfg.agent.hot_window_token_budget} tokens",
            title="Configuration",
            border_style="cyan",
        )
    )

    # Ensure model is present.
    console.print("[cyan]Ensuring model is downloaded...[/cyan]")
    model_path = ensure_model(cfg)
    console.print(f"[green]Model:[/green] {model_path}")

    # Lazy import to avoid loading llama-cpp-python on every command.
    from ultres.models.loader import load_llama
    from ultres.agent.research_loop import run_research_loop
    from ultres.agent.reasoner import stream_answer
    from ultres.search.base import get_provider

    console.print("[cyan]Loading model into llama.cpp...[/cyan]")
    llm = load_llama(cfg, spec=spec, model_path=model_path)
    console.print("[green]Model loaded.[/green]")

    # Provider.
    if cfg.search.backend == "searxng":
        provider = get_provider("searxng", base_url=cfg.search.searxng_base_url)
    elif cfg.search.backend == "tavily":
        provider = get_provider("tavily", api_key=cfg.search.tavily_api_key)
    elif cfg.search.backend == "brave":
        provider = get_provider("brave", api_key=cfg.search.brave_api_key)
    else:
        raise typer.BadParameter(f"Unknown search backend: {cfg.search.backend}")

    # Run the loop.
    result = asyncio.run(run_research_loop(llm, cfg, query, provider=provider, console=console))

    # Stream + persist answer.
    answer_path = cfg.answers_dir / f"{result.query_id}.md"
    stream_answer(result.answer, answer_path, console=console)

    console.print(
        f"\n[dim]query_id={result.query_id} | steps={result.steps} | "
        f"sources={len(result.visited_urls)}[/dim]"
    )


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
    table.add_column("default quant")
    for key, spec in list_specs().items():
        table.add_row(
            key,
            spec.label,
            f"{spec.params_b}B",
            f"{spec.native_ctx:,}",
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
                 "agent.max_steps", "agent.hot_window_token_budget", "save", "quit"],
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
    """List past research queries in this project."""
    cfg = UltResConfig.load()
    if not cfg.store_dir.exists():
        console.print("[dim]No queries yet.[/dim]")
        return
    table = Table(title="Past queries")
    table.add_column("query_id", style="cyan")
    table.add_column("query")
    table.add_column("docs", justify="right")
    table.add_column("code", justify="right")
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
        table.add_row(
            meta.get("query_id", qdir.name),
            (meta.get("user_query", "") or "")[:60],
            str(len(meta.get("docs", {}))),
            str(len(meta.get("code", {}))),
        )
    console.print(table)


@app.command("show")
def show_query(query_id: str = typer.Argument(..., help="The query_id to show.")):
    """Print a past answer + research tree."""
    cfg = UltResConfig.load()
    answer_path = cfg.answers_dir / f"{query_id}.md"
    if answer_path.exists():
        console.print(Panel(answer_path.read_text("utf-8"), title=f"Answer {query_id}", border_style="green"))
    else:
        console.print(f"[red]No answer found for {query_id}[/red]")

    meta_path = cfg.store_dir / query_id / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text("utf-8"))
        console.print(Panel(json.dumps(meta, indent=2), title="Research tree", border_style="cyan"))


@app.command("clean")
def clean():
    """Remove all stored queries + vector index (keeps config + models)."""
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


# Redirect bare queries to `run` before Typer processes args.
_maybe_redirect_to_run()

if __name__ == "__main__":
    app()
