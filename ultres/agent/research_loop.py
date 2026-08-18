"""Agentic research loop: plan -> search -> visit -> extract -> summarize -> store.

The model drives the loop via tool calls. The orchestrator executes each tool,
writes results to the knowledge store, and returns a *compact* confirmation
(not full content) to the model's hot context. On `recall`/`load_*`, the
requested slice is injected into hot context (evicting oldest if over the soft
token budget).

This is the heart of UltRes: the disk-backed store is the "cold tier", the
hot context window is just the working area, and the model navigates between
them via tool calls — giving effectively-unlimited logical context.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ultres.agent.planner import plan_query
from ultres.agent.tools import TOOL_SCHEMAS, system_prompt
from ultres.memory.hierarchical import build_tree
from ultres.memory.kv_cache import KVCacheManager
from ultres.memory.store import KnowledgeStore
from ultres.memory.vector import VectorIndex
from ultres.search.base import SearchProvider, get_provider


# ---------------------------------------------------------------------------
# Hot context window management
# ---------------------------------------------------------------------------

@dataclass
class HotWindow:
    """Manages the messages in the model's hot context with eviction."""

    budget: int  # soft token budget
    messages: list[dict[str, str]] = field(default_factory=list)
    _tokens: int = 0

    @staticmethod
    def _est_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def total_tokens(self) -> int:
        return self._tokens

    def append(self, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        self.messages.append(msg)
        self._tokens += self._est_tokens(content)
        self._evict()

    def _evict(self) -> None:
        """Evict oldest tool-result messages until under budget.

        Always keeps the system prompt (index 0) and the last user message.
        """
        while self._tokens > self.budget and len(self.messages) > 3:
            # Remove the oldest non-system message that isn't the last user turn.
            for i in range(1, len(self.messages) - 1):
                role = self.messages[i]["role"]
                if role in ("tool", "assistant"):
                    removed = self.messages.pop(i)
                    self._tokens -= self._est_tokens(removed["content"])
                    break
            else:
                break


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes tool calls against the search provider + knowledge store."""

    def __init__(
        self,
        provider: SearchProvider,
        store: KnowledgeStore,
        index: VectorIndex,
        kv: KVCacheManager,
        summarizer=None,
    ):
        self.provider = provider
        self.store = store
        self.index = index
        self.kv = kv
        self.summarizer = summarizer

    async def execute(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool and return a compact confirmation string."""
        if name == "search":
            return await self._search(args)
        if name == "visit":
            return await self._visit(args)
        if name == "recall":
            return await self._recall(args)
        if name == "load_summary":
            return self._load_summary(args)
        if name == "load_slice":
            return self._load_slice(args)
        if name == "load_code":
            return self._load_code(args)
        if name == "finish":
            return args.get("answer", "")
        return f"Unknown tool: {name}"

    # -- tool implementations -----------------------------------------

    async def _search(self, args: dict[str, Any]) -> str:
        query = args.get("query", "")
        n = int(args.get("n", 5))
        try:
            results = await self.provider.search(query, n=n)
        except Exception as e:
            return f"search error: {e}"
        if not results:
            return f"No results for: {query}"
        lines = [f"Found {len(results)} results for '{query}'. Call visit(url) to fetch the most relevant pages:"]
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r.title}\n     {r.url}")
        lines.append("NEXT STEP: Call visit(url) on the most relevant URL to extract its content.")
        return "\n".join(lines)

    async def _visit(self, args: dict[str, Any]) -> str:
        url = args.get("url", "")
        try:
            page = await self.provider.fetch(url)
        except Exception as e:
            return f"visit error for {url}: {e}"
        if page.fetch_error and not page.text:
            return f"visit failed for {url}: {page.fetch_error}"
        # Ingest into store + index.
        info = build_tree(self.store, page, summarizer=self.summarizer)
        # Index notes + summaries + code snippets.
        notes = self.store.get_notes(info["doc_id"]) or {}
        self.index.add_note(
            info["doc_id"],
            "\n".join(notes.get("key_paragraphs", [])),
            url=page.url,
            title=page.title,
        )
        self.index.add_summary(
            info["topic_id"],
            self.store.get_topic_summary(info["topic_id"]),
            kind="topic",
            name=info["topic_name"],
        )
        self.index.add_summary(
            info["subtopic_id"],
            self.store.get_subtopic_summary(info["subtopic_id"]),
            kind="subtopic",
            name=info["subtopic_name"],
        )
        for cid in info["code_ids"]:
            code_text = self.store.get_code(cid) or ""
            meta = self.store.meta.code.get(cid, {})
            self.index.add_code(cid, code_text, language=meta.get("language"))
        headings = notes.get("headings", [])[:10]
        return (
            f"VISITED {url}\n"
            f"  doc_id={info['doc_id']}  <-- USE THIS EXACT doc_id with load_slice\n"
            f"  title: {page.title}\n"
            f"  topic: {info['topic_name']} ({info['topic_id']})\n"
            f"  headings: {headings}\n"
            f"  code_blocks: {info['code_ids'] or 'none'}\n"
            f"  NEXT: Call load_slice with doc_id={info['doc_id']} to read content."
        )

    def _recall(self, args: dict[str, Any]) -> str:
        query = args.get("query", "")
        k = int(args.get("k", 8))
        hits = self.index.recall(query, k=k)
        if not hits:
            return f"recall: nothing stored yet for '{query}'"
        lines = [f"recall '{query}' -> {len(hits)} hits (use doc_id with load_slice):"]
        for h in hits:
            kind = h.metadata.get("kind", "?")
            ref = h.metadata.get("doc_id") or h.metadata.get("summary_id") or h.metadata.get("code_id", "")
            preview = h.text[:160].replace("\n", " ")
            lines.append(f"  [{kind}] doc_id={ref} (score={h.score:.2f}): {preview}")
        return "\n".join(lines)

    def _load_summary(self, args: dict[str, Any]) -> str:
        topic = args.get("topic", "")
        # Find closest topic by name.
        best = None
        best_score = -1.0
        for t in self.store.list_topics():
            name = t.get("name", "")
            # Simple overlap score.
            score = len(set(name.lower().split()) & set(topic.lower().split()))
            if score > best_score:
                best_score = score
                best = t
        if not best:
            return f"No topics stored yet."
        summary = self.store.get_topic_summary(best["topic_id"])
        self.kv.mark_attended(f"summary:{best['topic_id']}", summary)
        return f"Topic summary for '{best['name']}':\n{summary}"

    def _load_slice(self, args: dict[str, Any]) -> str:
        doc_id = args.get("doc_id", "")
        section = args.get("section")
        text = self.store.get_doc_slice(doc_id, section)
        if not text:
            return f"No doc found for doc_id={doc_id}"
        # Truncate to a reasonable slice for the hot window.
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated; use load_slice with a section to drill in]"
        self.kv.mark_attended(f"slice:{doc_id}:{section or 'all'}", text)
        return f"Slice of {doc_id} (section={section or 'all'}):\n{text}"

    def _load_code(self, args: dict[str, Any]) -> str:
        code_id = args.get("code_id", "")
        text = self.store.get_code(code_id)
        if not text:
            return f"No code block found for code_id={code_id}"
        self.kv.mark_attended(f"code:{code_id}", text)
        return f"Code block {code_id}:\n{text}"


# ---------------------------------------------------------------------------
# Research loop
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    answer: str
    steps: int
    query_id: str
    visited_urls: list[str]


def _extract_tool_call(resp: dict[str, Any]) -> tuple[str | None, dict[str, Any], str]:
    """Parse a model response into (tool_name, args, content_text).

    Supports both native tool_calls and JSON-in-content fallback.
    The JSON-in-content path handles:
      - ```json ... ``` blocks
      - Bare {"name": "...", "arguments": {...}} objects
      - Objects with nested arguments dict
    """
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    # Native tool_calls (OpenAI-style).
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0]
        fn = tc["function"]
        name = fn["name"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        return name, args, content

    # Fallback: parse JSON from content.
    import re

    # Try ```json ... ``` block first.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            name = data.get("name")
            if name:
                args = data.get("arguments", data.get("args", {}))
                return name, args, content
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object with "name" in the content.
    # Use a greedy approach: find all { ... } blocks and try to parse.
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL):
        try:
            data = json.loads(m.group(0))
            name = data.get("name")
            if name:
                args = data.get("arguments", data.get("args", {}))
                return name, args, content
        except json.JSONDecodeError:
            continue

    return None, {}, content


async def run_research_loop(
    llm: Any,
    cfg: Any,
    user_query: str,
    provider: SearchProvider | None = None,
    console: Console | None = None,
) -> LoopResult:
    """Run the full research loop for a user query.

    Returns the final answer + metadata.
    """
    console = console or Console()
    cfg.ensure_dirs()

    # Set up per-query store + index.
    query_id = uuid.uuid4().hex[:12]
    store = KnowledgeStore(cfg.store_dir, query_id=query_id)
    store.set_user_query(user_query)
    index = VectorIndex(cfg.index_dir, query_id=query_id, embedding_model=cfg.memory.embedding_model)
    kv = KVCacheManager(cfg.index_dir / "kv", enabled=cfg.memory.kv_cache_reuse)

    # Search provider.
    if provider is None:
        provider = _make_provider(cfg)

    # Summarizer (uses the same llm, short context).
    def summarizer(text: str, max_words: int) -> str:
        if not text.strip():
            return ""
        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": f"Summarize in <= {max_words} words."},
                    {"role": "user", "content": text[:8000]},
                ],
                temperature=0.3,
                max_tokens=max_words * 2,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return text[: max_words * 5]

    executor = ToolExecutor(provider, store, index, kv, summarizer=summarizer)

    # --- Plan ---
    with Live(Panel(Text("Planning...", style="cyan")), console=console, refresh_per_second=4) as live:
        subtasks = plan_query(llm, user_query, temperature=cfg.agent.temperature)
        live.update(Panel(Text(f"Plan: {len(subtasks)} subtasks", style="green")))

    # --- Hot window ---
    window = HotWindow(budget=cfg.agent.hot_window_token_budget)
    window.append("system", system_prompt())
    window.append(
        "user",
        f"User request: {user_query}\n\nSuggested research subtasks:\n"
        + "\n".join(f"- {s['query']}" for s in subtasks),
    )

    visited_urls: list[str] = []
    steps = 0
    max_steps = cfg.agent.max_steps
    answer = ""

    with Live(console=console, refresh_per_second=4) as live:
        while steps < max_steps:
            steps += 1
            live.update(
                Panel(
                    Text(
                        f"Step {steps}/{max_steps} | hot ctx ~{window.total_tokens()} tokens | "
                        f"store: {len(store.meta.docs)} docs, {len(store.meta.code)} code blocks",
                        style="cyan",
                    )
                )
            )

            # Ask the model for the next action.
            # We use prompt-based tool calling (tools described in system prompt)
            # rather than the native `tools` API parameter, because Qwen2.5
            # doesn't reliably use native tool-calling with chatml format.
            try:
                resp = llm.create_chat_completion(
                    messages=window.messages,
                    temperature=cfg.agent.temperature,
                    max_tokens=1024,
                )
            except Exception as e:
                window.append("assistant", f"(internal error: {e})")
                continue

            name, args, content = _extract_tool_call(resp)
            if content:
                window.append("assistant", content)

            if name is None:
                # No tool call — nudge the model with a stronger instruction.
                window.append(
                    "user",
                    "You must call a tool. Respond with ONLY a JSON tool call like:\n"
                    '{"name": "search", "arguments": {"query": "..."}}\n'
                    "Do not answer directly. Research first.",
                )
                continue

            if name == "finish":
                if len(visited_urls) < 2:
                    window.append(
                        "tool",
                        f"BLOCKED: You have visited {len(visited_urls)} pages. "
                        "You MUST visit at least 2 pages before finishing. "
                        "Call visit(url) on a search result URL next.",
                    )
                    continue
                answer = args.get("answer", content)
                break

            # Block recall before any visits — the store is empty.
            if name == "recall" and len(visited_urls) == 0:
                window.append(
                    "tool",
                    "BLOCKED: recall searches stored content, but no pages have been "
                    "visited yet. Call visit(url) first to populate the store.",
                )
                continue

            # Execute the tool.
            try:
                result = await executor.execute(name, args)
            except Exception as e:
                result = f"tool error ({name}): {e}"

            if name == "visit":
                url = args.get("url", "")
                if url and url not in visited_urls:
                    visited_urls.append(url)

            # Return a compact confirmation to the model.
            window.append("tool", result)

        if not answer:
            # Ran out of steps — force a final answer.
            window.append(
                "user",
                "You've reached the step limit. Call finish now with your best answer.",
            )
            try:
                resp = llm.create_chat_completion(
                    messages=window.messages,
                    temperature=cfg.agent.final_temperature,
                    max_tokens=2048,
                )
                _, fargs, fcontent = _extract_tool_call(resp)
                answer = fargs.get("answer", fcontent)
            except Exception:
                answer = "(failed to produce a final answer)"

    # Persist answer.
    answer_path = cfg.answers_dir / f"{query_id}.md"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(
        f"# Query\n{user_query}\n\n# Answer\n{answer}\n\n# Sources\n"
        + "\n".join(f"- {u}" for u in visited_urls),
        "utf-8",
    )

    return LoopResult(
        answer=answer,
        steps=steps,
        query_id=query_id,
        visited_urls=visited_urls,
    )


def _make_provider(cfg: Any) -> SearchProvider:
    """Instantiate the configured search provider."""
    backend = cfg.search.backend
    if backend == "searxng":
        return get_provider("searxng", base_url=cfg.search.searxng_base_url)
    if backend == "tavily":
        return get_provider("tavily", api_key=cfg.search.tavily_api_key)
    if backend == "brave":
        return get_provider("brave", api_key=cfg.search.brave_api_key)
    raise ValueError(f"Unknown search backend: {backend}")
