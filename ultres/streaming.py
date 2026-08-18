"""Live streaming for the deep research pipeline.

Provides real-time terminal display with per-stage timers, progress bars,
and token-by-token streaming of model output. Also includes an optional
SSE (Server-Sent Events) endpoint for web UI integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------

@dataclass
class StageInfo:
    """Timing + stats for a single pipeline stage."""
    name: str
    label: str
    start_time: float = 0.0
    end_time: float = 0.0
    done: bool = False
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        if self.start_time > 0:
            return time.time() - self.start_time
        return 0.0

    @property
    def elapsed_str(self) -> str:
        s = int(self.elapsed)
        return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Terminal streamer
# ---------------------------------------------------------------------------

class ResearchStreamer:
    """Live terminal display for the deep research pipeline.

    Shows per-stage timers, progress, and streams model tokens in real-time.
    Falls back to simple console.print if rich Live is not desired.
    """

    def __init__(self, console: Console | None = None, enabled: bool = True):
        self.console = console or Console()
        self.enabled = enabled
        self.start_time = time.time()
        self.stages: list[StageInfo] = []
        self._stage_map: dict[str, StageInfo] = {}
        self._current: StageInfo | None = None
        self._live: Live | None = None

    @property
    def total_elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def total_elapsed_str(self) -> str:
        s = int(self.total_elapsed)
        return f"{s // 60:02d}:{s % 60:02d}"

    def stage_start(self, name: str, label: str) -> None:
        """Begin a new stage."""
        info = StageInfo(name=name, label=label, start_time=time.time())
        self.stages.append(info)
        self._stage_map[name] = info
        self._current = info
        if self.enabled:
            self.console.print(f"[cyan]▶ {label}...[/cyan]")

    def stage_progress(self, name: str, current: int, total: int, detail: str = "") -> None:
        """Update progress for a stage.

        v1.4: Uses console.print with a progress bar instead of \\r-based
        rendering, which was causing display issues in some terminals.
        """
        info = self._stage_map.get(name)
        if not info:
            return
        info.stats["current"] = current
        info.stats["total"] = total
        info.stats["detail"] = detail
        if self.enabled and total > 0:
            pct = current * 100 // total
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            # Use a simple print with the progress info.
            # We don't use \r because it conflicts with rich's Live display.
            # Instead, print a dim progress line that gets overwritten by the
            # next stage_done or stage_progress call.
            self.console.print(
                f"  [dim]{bar} {current}/{total} {detail}[/dim]",
            )

    def stage_done(self, name: str, stats: dict[str, Any] | None = None) -> None:
        """Mark a stage as done."""
        info = self._stage_map.get(name)
        if not info:
            return
        info.end_time = time.time()
        info.done = True
        if stats:
            info.stats.update(stats)
        self._current = None
        if self.enabled:
            stat_str = " | ".join(f"{k}={v}" for k, v in (stats or {}).items())
            self.console.print(
                f"[green]✓ {info.label}[/green] "
                f"[dim][{info.elapsed_str}]{f' — {stat_str}' if stat_str else ''}[/dim]"
            )

    def token_stream(
        self,
        llm: Any,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Stream model output token-by-token to the terminal.

        Uses llama-cpp-python's stream=True. Returns the full text.
        """
        kwargs["stream"] = True
        full_text = ""
        try:
            for chunk in llm.create_chat_completion(messages=messages, **kwargs):
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_text += token
                    if self.enabled:
                        self.console.print(token, end="")
        except Exception as e:
            if self.enabled:
                self.console.print(f"\n[red]Stream error: {e}[/red]")
            # Fallback: non-streaming.
            kwargs.pop("stream", None)
            resp = llm.create_chat_completion(messages=messages, **kwargs)
            full_text = resp["choices"][0]["message"]["content"] or ""
            if self.enabled:
                self.console.print(full_text, end="")
        if self.enabled:
            self.console.print()  # newline
        return full_text

    def print(self, msg: str, style: str = "") -> None:
        """Print a message to the console."""
        if self.enabled:
            self.console.print(msg, style=style) if style else self.console.print(msg)

    def summary(self) -> dict[str, Any]:
        """Return timing summary for all stages."""
        return {
            "total_elapsed": self.total_elapsed,
            "stages": [
                {
                    "name": s.name,
                    "label": s.label,
                    "elapsed": s.elapsed,
                    "stats": s.stats,
                }
                for s in self.stages
            ],
        }

    def display_summary(self) -> None:
        """Print a final timing summary table."""
        if not self.enabled:
            return
        self.console.print()
        self.console.print(Panel(
            "\n".join(
                f"{'✓' if s.done else '○'} {s.label:40s} [{s.elapsed_str}]"
                + (f" — {', '.join(f'{k}={v}' for k, v in s.stats.items() if k not in ('current', 'total', 'detail'))}"
                   if any(k not in ('current', 'total', 'detail') for k in s.stats) else "")
                for s in self.stages
            ) + f"\n{'─' * 50}\nTotal elapsed: {self.total_elapsed_str}",
            title="Pipeline Summary",
            border_style="cyan",
        ))


# ---------------------------------------------------------------------------
# SSE streamer (optional, for web UI)
# ---------------------------------------------------------------------------

class SSEStreamer:
    """Server-Sent Events streamer for web UI integration.

    Yields SSE-formatted events that can be consumed by a browser EventSource.
    """

    def __init__(self):
        self.events: list[str] = []

    def event(self, event_type: str, data: dict[str, Any]) -> str:
        """Format an SSE event."""
        import json
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        self.events.append(msg)
        return msg

    def stage_start(self, name: str, label: str) -> str:
        return self.event("stage_start", {"name": name, "label": label})

    def stage_done(self, name: str, stats: dict[str, Any]) -> str:
        return self.event("stage_done", {"name": name, "stats": stats})

    def token(self, text: str) -> str:
        return self.event("token", {"text": text})

    def final_answer(self, answer: str) -> str:
        return self.event("final_answer", {"answer": answer})
