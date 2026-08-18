"""Final-answer reasoner.

After the research loop ends, this produces the streamed final answer. In v1
the loop's `finish` tool already produces the answer, so this module is a thin
wrapper that handles streaming to the terminal and writing to disk.

v1.2 adds two_pass_implement() for the deep research pipeline:
  Pass 1 (Instruct): synthesize research into an implementation plan
  Pass 2 (Coder): implement from the plan, forced to follow research
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ultres.research.compressor import MasterBrief
from ultres.streaming import ResearchStreamer


def stream_answer(
    answer: str,
    answer_path: Path,
    console: Console | None = None,
) -> None:
    """Render the final answer to the terminal and persist it to disk."""
    console = console or Console()
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(answer, "utf-8")
    console.print()
    console.print(Panel(Markdown(answer), title="UltRes answer", border_style="green"))
    console.print(f"\n[dim]Saved to {answer_path}[/dim]")


def final_reasoning_pass(
    llm: Any,
    user_query: str,
    topic_summaries: list[str],
    loaded_code: list[str],
    temperature: float = 0.4,
) -> str:
    """Optional v2-style final pass: synthesize across all topic summaries + code.

    v1 does not call this (the loop's `finish` is the answer), but it's here
    for v2 when the long-context model can hold more at once.
    """
    context_parts = []
    if topic_summaries:
        context_parts.append("## Topic summaries\n" + "\n\n".join(topic_summaries))
    if loaded_code:
        context_parts.append("## Code blocks\n" + "\n\n---\n\n".join(loaded_code))
    context = "\n\n".join(context_parts) or "(no additional context)"
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Synthesize a complete answer from the provided research."},
            {"role": "user", "content": f"Query: {user_query}\n\nContext:\n{context[:60000]}"},
        ],
        temperature=temperature,
        max_tokens=2048,
    )
    return resp["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# v1.2: Two-pass implementation
# ---------------------------------------------------------------------------

_PASS1_SYSTEM = """\
You are the architecture planner for UltRes, a research-driven AI. \
You have been given a master research brief derived from analyzing \
thousands of web pages. Your job is to write a detailed implementation \
plan based on this research.

The plan must:
- Specify the exact architecture and file structure
- Name the specific algorithms and patterns to use (from the research)
- Reference specific code examples that should be adapted
- List all libraries and tools recommended by the research
- Identify potential pitfalls and how to handle them
- Be specific enough that a coder can implement it without making \
  architecture decisions

Do NOT write code. Write a PLAN that a coder will follow.
"""

_PASS2_SYSTEM = """\
You are the implementation engine for UltRes. You MUST implement based on \
the plan below. The plan is derived from web research of thousands of pages.

CRITICAL RULES:
1. Do NOT substitute your own pretrained knowledge for research-based \
   architecture decisions. The plan specifies which algorithms, patterns, \
   and libraries to use — USE THOSE.
2. Use the provided code examples as reference and adapt them.
3. If the plan specifies an algorithm, use that algorithm.
4. If the plan specifies a library, use that library.
5. Your job is to write clean, working code that follows the plan — not to \
   redesign the architecture.

Write complete, production-quality code. Include error handling, comments, \
and proper structure as specified in the plan.
"""


def two_pass_implement(
    model_mgr: Any,
    user_query: str,
    brief: MasterBrief,
    streamer: ResearchStreamer | None = None,
    temperature: float = 0.4,
) -> tuple[str, str]:
    """Two-pass implementation: Instruct writes plan, Coder implements.

    Uses the ModelManager to swap between models, ensuring only one is
    loaded at a time (required for 8GB VRAM).

    Args:
        model_mgr: ModelManager instance for loading/unloading models.
        user_query: The user's original request.
        brief: Master research brief with code examples.
        streamer: Optional streamer for live token display.
        temperature: Generation temperature.

    Returns:
        Tuple of (plan, implementation) strings.
    """
    # Build code examples text.
    code_text = "\n\n---\n\n".join(
        f"```{cb.language or ''}\n{cb.content}\n```"
        for cb in brief.code_examples
    )

    # --- Pass 1: Instruct model writes the plan ---
    llm_instruct = model_mgr.load_instruct()

    pass1_prompt = (
        f"User request: {user_query}\n\n"
        f"Master research brief ({brief.total_pages} pages, "
        f"{brief.total_clusters} clusters, {brief.total_code_blocks} code blocks):\n\n"
        f"{brief.text}\n\n"
        f"Best code examples from research:\n\n{code_text[:30000]}\n\n"
        f"Write a detailed implementation plan for this request. "
        f"Be specific about architecture, algorithms, file structure, and libraries."
    )

    if streamer and streamer.enabled:
        streamer.print("\n[bold cyan]Pass 1: Writing implementation plan (Instruct model)...[/bold cyan]")

    if streamer and streamer.enabled:
        plan = streamer.token_stream(
            llm_instruct,
            messages=[
                {"role": "system", "content": _PASS1_SYSTEM},
                {"role": "user", "content": pass1_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
    else:
        resp = llm_instruct.create_chat_completion(
            messages=[
                {"role": "system", "content": _PASS1_SYSTEM},
                {"role": "user", "content": pass1_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        plan = resp["choices"][0]["message"]["content"].strip()

    # --- Pass 2: Coder model implements from the plan ---
    # Swap to coder model (unloads instruct first to free VRAM).
    coder = None
    coder_available = False
    try:
        coder = model_mgr.load_coder()
        coder_available = True
    except Exception as e:
        if streamer and streamer.enabled:
            streamer.print(f"[yellow]Coder model unavailable ({e}); trying Instruct fallback...[/yellow]")
        # Try to fall back to Instruct model for Pass 2.
        try:
            coder = model_mgr.load_instruct()
            coder_available = False
        except Exception as e2:
            if streamer and streamer.enabled:
                streamer.print(f"[red]Instruct fallback also failed ({e2}); using plan as answer.[/red]")
            # Last resort: return the plan as the implementation.
            return plan, plan

    pass2_prompt = (
        f"User request: {user_query}\n\n"
        f"Implementation plan (derived from research — FOLLOW THIS):\n\n{plan}\n\n"
        f"Reference code examples from research:\n\n{code_text[:30000]}\n\n"
        f"Now implement the complete solution based on this plan. "
        f"Write all the code. Follow the plan exactly."
    )

    if streamer and streamer.enabled:
        model_label = "Coder model" if coder_available else "Instruct model (fallback)"
        streamer.print(f"\n[bold cyan]Pass 2: Implementing from plan ({model_label})...[/bold cyan]")

    if streamer and streamer.enabled:
        implementation = streamer.token_stream(
            coder,
            messages=[
                {"role": "system", "content": _PASS2_SYSTEM},
                {"role": "user", "content": pass2_prompt},
            ],
            temperature=temperature,
            max_tokens=8192,
        )
    else:
        resp = coder.create_chat_completion(
            messages=[
                {"role": "system", "content": _PASS2_SYSTEM},
                {"role": "user", "content": pass2_prompt},
            ],
            temperature=temperature,
            max_tokens=8192,
        )
        implementation = resp["choices"][0]["message"]["content"].strip()

    return plan, implementation
