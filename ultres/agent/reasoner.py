"""Final-answer reasoner.

After the research loop ends, this produces the streamed final answer. In v1
the loop's `finish` tool already produces the answer, so this module is a thin
wrapper that handles streaming to the terminal and writing to disk.

In v2 (with the long-context UltRes-Base-7B-Long), this may do an extra
`recall`-backed final check over the full topic summaries + loaded code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


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
