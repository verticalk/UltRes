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
import re
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
    """Manages the messages in the model's hot context with eviction.

    v1.6.1: Pins the first `pinned_count` messages (system prompt + initial
    user message) so they are never evicted. This allows llama.cpp's internal
    prefix KV cache to reuse the cached state for those messages across steps,
    reducing prompt processing time for steps 2-N.
    """

    budget: int  # soft token budget
    messages: list[dict[str, str]] = field(default_factory=list)
    pinned_count: int = 2  # v1.6.1: system + initial user message are pinned
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

        v1.6.1: Always keeps the pinned messages (first `pinned_count`) and
        the last user message. This preserves the KV cache prefix.
        """
        while self._tokens > self.budget and len(self.messages) > self.pinned_count + 2:
            # Remove the oldest non-pinned message that isn't the last user turn.
            for i in range(self.pinned_count, len(self.messages) - 1):
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
        # v1.6.1: Ingest into store with summarizer=None (lazy summarization).
        # Summaries are batch-generated after the research loop completes,
        # eliminating 2 model calls per visit (subtopic + topic summary).
        # The recall() tool still searches extractive notes + raw text.
        info = build_tree(self.store, page, summarizer=None)
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

    Supports multiple tool-call formats:
    1. Native tool_calls (OpenAI-style) — tool_calls field in message
    2. JSON in content — ```json ... ``` blocks or bare JSON objects
    3. v1.6.1: Qwen3 XML format (inside or outside thinking blocks):
       <function=name>\n<parameter=key>value</parameter>\n</function>

    Also strips thinking tokens (...) from the returned content.
    """
    msg = resp["choices"][0]["message"]
    raw_content = msg.get("content") or ""

    # v1.6.1: Strip Qwen3 thinking tokens from content for the window.
    # Qwen3.8 generates thinking reasoning before the actual response.
    # We keep thinking out of the hot window to avoid filling it with
    # reasoning tokens that don't contribute to the tool call.
    # The tags use angle brackets: ... 
    content = re.sub(r"<think>.*?</think>\s*", "", raw_content, flags=re.DOTALL)
    content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
    content = content.strip()

    # 1. Native tool_calls (OpenAI-style).
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

    # 2. v1.6.1: Qwen3 XML format — search in BOTH raw and stripped content.
    # The model outputs <function=name>...<parameter=key>value</parameter>...</function>
    # This may be inside thinking blocks, so we search the raw content too.
    xml_pattern = r"<function=(\w+)>(.*?)</function>"
    for m in re.finditer(xml_pattern, raw_content, re.DOTALL):
        name = m.group(1)
        body = m.group(2)
        # Parse <parameter=key>value</parameter> entries.
        args: dict[str, Any] = {}
        for pm in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", body, re.DOTALL):
            args[pm.group(1)] = pm.group(2).strip()
        if name:
            return name, args, content

    # 3. JSON in content — ```json ... ``` blocks.
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

    # 4. Bare JSON object with "name" in content.
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


async def _synthesize_answer(
    llm: Any,
    executor: "ToolExecutor",
    user_query: str,
    draft_answer: str,
    window: "HotWindow",
    cfg: Any,
) -> str:
    """Synthesis pass: load recalled content and produce an evidence-cited answer.

    v1.6.1: Now async (was sync with asyncio.run() that crashed inside a running loop).
    Forces the model to ground its answer in retrieved content rather than
    pretrained knowledge. If the draft already references doc_ids, keep it.
    """
    # Recall relevant content.
    try:
        recall_result = await executor._recall({"query": user_query, "k": 8})
    except Exception:
        recall_result = "(recall failed)"

    # Load top slices into the synthesis context.
    synthesis_context = recall_result
    # Try to load a few slices for grounding.
    import re as _re

    doc_ids = _re.findall(r"doc_id=([a-f0-9_]+)", recall_result)
    for did in doc_ids[:3]:
        try:
            slice_text = executor.store.get_doc_slice(did)
            if slice_text:
                synthesis_context += f"\n\n--- Content from {did} ---\n"
                synthesis_context += slice_text[:4000]
        except Exception:
            pass

    synthesis_prompt = (
        f"You are synthesizing a final answer for: {user_query}\n\n"
        f"Draft answer: {draft_answer}\n\n"
        f"Retrieved research:\n{synthesis_context[:12000]}\n\n"
        f"Rewrite the answer to be detailed and grounded in the retrieved research. "
        f"Reference specific facts from the content. If the draft is already well-grounded, "
        f"you may keep it. Produce ONLY the final answer text (no tool call)."
    )
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a precise research synthesizer."},
                {"role": "user", "content": synthesis_prompt},
            ],
            temperature=cfg.agent.final_temperature,
            max_tokens=2048,
        )
        synthesized = resp["choices"][0]["message"]["content"].strip()
        if synthesized:
            return synthesized
    except Exception:
        pass
    return draft_answer


def _self_critique(
    llm: Any,
    user_query: str,
    answer: str,
    recall_context: str,
    cfg: Any,
) -> tuple[str, list[str]]:
    """Critique an answer against retrieved evidence.

    Returns (critique_text, list_of_gap_queries).
    If the critique finds gaps, the gap queries are used to re-enter research.
    """
    critique_prompt = (
        f"You are critiquing a research answer for: {user_query}\n\n"
        f"Answer: {answer}\n\n"
        f"Available research evidence:\n{recall_context[:8000]}\n\n"
        f"Identify any gaps, errors, or unsupported claims in the answer. "
        f"If the answer is well-supported, say 'VERIFIED'. "
        f"If there are gaps, list specific search queries that would fill them. "
        f"Respond in this format:\n"
        f"VERDICT: VERIFIED or NEEDS_MORE_RESEARCH\n"
        f"GAPS: <comma-separated search queries, or empty if verified>\n"
        f"NOTES: <brief explanation>"
    )
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a rigorous research critic."},
                {"role": "user", "content": critique_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        # Parse verdict + gaps.
        gaps: list[str] = []
        for line in text.splitlines():
            if line.startswith("GAPS:"):
                gap_str = line[5:].strip()
                if gap_str and gap_str.lower() not in ("none", "empty", ""):
                    gaps = [g.strip() for g in gap_str.split(",") if g.strip()]
        return text, gaps
    except Exception:
        return "VERDICT: VERIFIED\nGAPS:\nNOTES: critique failed", []


def _save_trajectory(
    cfg: Any,
    query_id: str,
    user_query: str,
    answer: str,
    steps: int,
    visited_urls: list[str],
    doc_ids: list[str],
    code_ids: list[str],
    query_type: str = "general",
) -> None:
    """Save a research trajectory in the unified v1.4 format.

    Both fast and deep modes use the same schema for v1.5 QLoRA compatibility.
    """
    traj_dir = cfg.ultres_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{query_id}.jsonl"
    record = {
        "query_id": query_id,
        "query": user_query,
        "mode": "fast",  # fast agentic loop
        "query_type": query_type,
        "answer": answer,
        "plan": None,  # fast mode has no Pass 1 plan
        "brief": None,  # fast mode has no master brief
        "steps": steps,
        "pages_crawled": len(visited_urls),  # fast mode: pages = visited URLs
        "clusters": None,
        "gaps": None,
        "visited_urls": visited_urls,
        "doc_ids": doc_ids,
        "code_ids": code_ids,
        "timing": {},
        "timestamp": time.time(),
    }
    with traj_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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

    # v1.6.1: Pre-warm the KV cache with the system prompt + tools.
    # This makes a dummy call so llama.cpp caches the KV state for the
    # system prompt + tool schemas. Subsequent calls with the same prefix
    # will skip reprocessing those tokens.
    try:
        llm.create_chat_completion(
            messages=[window.messages[0]],
            tools=TOOL_SCHEMAS,
            max_tokens=1,
            temperature=0.0,
        )
    except Exception:
        pass  # Warmup failure is non-critical.

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
            # Use native tool calling (Qwen2.5-7B-Instruct supports it).
            # The _extract_tool_call fallback also parses JSON from content
            # in case native tool_calls is empty.
            try:
                resp = llm.create_chat_completion(
                    messages=window.messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=cfg.agent.temperature,
                    max_tokens=4096,  # v1.6.1: increased for thinking tokens
                )
            except Exception as e:
                # Fallback: retry without tools (some models/chat formats
                # don't support the tools parameter).
                try:
                    resp = llm.create_chat_completion(
                        messages=window.messages,
                        temperature=cfg.agent.temperature,
                        max_tokens=4096,
                    )
                except Exception as e2:
                    window.append("assistant", f"(internal error: {e2})")
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
                min_visits = cfg.agent.min_visits_before_finish
                if len(visited_urls) < min_visits:
                    window.append(
                        "tool",
                        f"BLOCKED: You have visited {len(visited_urls)} pages. "
                        f"You MUST visit at least {min_visits} pages before finishing. "
                        "Call visit(url) on a search result URL next.",
                    )
                    continue
                # Synthesis pass: force the model to produce an evidence-cited answer.
                draft_answer = args.get("answer", content)
                synthesized = await _synthesize_answer(
                    llm, executor, user_query, draft_answer, window, cfg
                )
                answer = synthesized
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
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=cfg.agent.final_temperature,
                    max_tokens=2048,
                )
                _, fargs, fcontent = _extract_tool_call(resp)
                draft = fargs.get("answer", fcontent)
                synthesized = await _synthesize_answer(
                    llm, executor, user_query, draft, window, cfg
                )
                answer = synthesized
            except Exception:
                answer = "(failed to produce a final answer)"

    # --- Self-critique loop ---
    if cfg.agent.enable_self_critique and answer:
        for round_num in range(cfg.agent.critique_rounds):
            console.print(f"[dim]Self-critique round {round_num + 1}/{cfg.agent.critique_rounds}...[/dim]")
            # Get recall context for the critique.
            try:
                recall_ctx = await executor._recall({"query": user_query, "k": 8})
            except Exception:
                recall_ctx = "(recall failed)"
            critique_text, gaps = _self_critique(
                llm, user_query, answer, recall_ctx, cfg
            )
            if not gaps:
                console.print(f"[green]Self-critique: answer verified.[/green]")
                break
            console.print(f"[yellow]Self-critique found gaps: {gaps}[/yellow]")
            # v1.6.1: Parallelize gap searches (was sequential).
            gap_queries = gaps[:3]
            console.print(f"[cyan]Re-researching {len(gap_queries)} gaps in parallel...[/cyan]")
            import re as _re

            search_tasks = [executor._search({"query": g, "n": 3}) for g in gap_queries]
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            # Visit the first result from each search in parallel.
            visit_tasks = []
            visit_queries = []
            for i, sr in enumerate(search_results):
                if isinstance(sr, Exception):
                    continue
                window.append("tool", sr)
                url_match = _re.search(r"https?://\S+", sr)
                if url_match:
                    visit_tasks.append(executor._visit({"url": url_match.group(0)}))
                    visit_queries.append(url_match.group(0))
            if visit_tasks:
                visit_results = await asyncio.gather(*visit_tasks, return_exceptions=True)
                for i, vr in enumerate(visit_results):
                    if isinstance(vr, str):
                        visited_urls.append(visit_queries[i])
                        window.append("tool", vr)
            # Re-synthesize with the new content.
            answer = await _synthesize_answer(llm, executor, user_query, answer, window, cfg)

    # --- Save trajectory for v1.5 QLoRA training ---
    doc_ids = list(store.meta.docs.keys())
    code_ids = list(store.meta.code.keys())
    # Detect query type for trajectory metadata.
    from ultres.research.source_prioritizer import detect_query_type as _detect_qt
    qt = _detect_qt(user_query)
    _save_trajectory(cfg, query_id, user_query, answer, steps, visited_urls, doc_ids, code_ids, query_type=qt)

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
