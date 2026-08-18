"""Deep research pipeline orchestrator (v1.2).

Ties together all stages: plan → crawl → cluster → gap detect → re-crawl →
summarize → compress → two-pass implement → critique → save trajectory.

This is the heart of UltRes v1.2's deep research mode.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel

from ultres.agent.planner import plan_deep
from ultres.agent.reasoner import two_pass_implement
from ultres.memory.store import KnowledgeStore
from ultres.memory.vector import VectorIndex
from ultres.research.clusterer import Cluster, cluster_pages
from ultres.research.compressor import MasterBrief, batch_summarize, hierarchical_compress
from ultres.research.crawler import CrawlResult, bulk_crawl
from ultres.research.gap_detector import detect_gaps
from ultres.research.source_prioritizer import detect_query_type
from ultres.streaming import ResearchStreamer


@dataclass
class DeepResearchResult:
    """Result of a deep research pipeline run."""
    answer: str
    query_id: str
    plan: str
    brief: str
    pages_crawled: int
    clusters_found: int
    gaps_detected: list[str] = field(default_factory=list)
    visited_urls: list[str] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    query_type: str = "general"


async def run_deep_research(
    model_mgr: Any,
    cfg: Any,
    user_query: str,
    provider: Any,
    console: Console | None = None,
    streamer: ResearchStreamer | None = None,
) -> DeepResearchResult:
    """Run the full deep research pipeline.

    Uses ModelManager to swap between Instruct and Coder models as needed,
    ensuring only one model is loaded at a time (required for 8GB VRAM).

    Args:
        model_mgr: ModelManager for loading/unloading models.
        cfg: UltResConfig.
        user_query: The user's request.
        provider: Search provider.
        console: Rich console for output.
        streamer: Optional streamer for live display.

    Returns:
        DeepResearchResult with the final answer + metadata.
    """
    console = console or Console()
    streamer = streamer or ResearchStreamer(console=console, enabled=cfg.deep_research.enable_streaming)
    cfg.ensure_dirs()

    query_id = uuid.uuid4().hex[:12]
    dr = cfg.deep_research
    pipeline_start = time.time()

    # Set up per-query store + index.
    store = KnowledgeStore(cfg.store_dir, query_id=query_id)
    store.set_user_query(user_query)
    index = VectorIndex(cfg.index_dir, query_id=query_id, embedding_model=cfg.memory.embedding_model)

    try:
        # Load Instruct model for stages 1-7.
        llm = model_mgr.load_instruct()

        # =========================================================================
        # Stage 1: Plan + Query Expansion
        # =========================================================================
        streamer.stage_start("plan", "Plan + Query Expansion")
        plan_result = plan_deep(llm, user_query, temperature=cfg.agent.temperature)
        queries = plan_result["expanded_queries"]
        query_type = plan_result["query_type"]
        streamer.stage_done("plan", {
            "queries": len(queries),
            "type": query_type,
        })

        if not queries:
            queries = [user_query]

        # =========================================================================
        # Stage 2: Bulk Crawl Round 1 (persist to disk store)
        # =========================================================================
        streamer.stage_start("crawl1", "Bulk Crawl Round 1")
        crawl_result = await bulk_crawl(
            provider=provider,
            queries=queries,
            max_pages=dr.max_pages,
            concurrency=dr.crawl_concurrency,
            crawl_depth=dr.crawl_depth,
            query_type=query_type,
            enable_quality_filter=dr.enable_quality_filter,
            min_quality=dr.min_page_quality,
            streamer=streamer,
            stage_name="crawl1",
            store=store,
            index=index,
        )
        streamer.stage_done("crawl1", {
            "pages": crawl_result.success_count,
            "failed": len(crawl_result.errors),
            "deduped": crawl_result.deduped,
        })

        all_pages = list(crawl_result.pages)
        all_urls = list(crawl_result.urls_fetched)

        # Bug 5: Handle empty crawl gracefully.
        if not all_pages:
            streamer.stage_done("crawl1", {"pages": 0, "error": "No pages crawled"})
            console.print("[red]No pages crawled. Check that SearXNG is running and accessible.[/red]")
            console.print(f"[dim]SearXNG URL: {cfg.search.searxng_base_url}[/dim]")
            # Save error trajectory.
            _save_trajectory_unified(
                cfg, query_id, user_query, answer="(no answer — crawl failed)",
                plan="", brief="", mode="deep", query_type=query_type,
                pages_crawled=0, clusters=0, gaps=[], visited_urls=[],
                doc_ids=[], code_ids=[], timing=streamer.summary(),
            )
            return DeepResearchResult(
                answer="(no answer — crawl failed)",
                query_id=query_id, plan="", brief="",
                pages_crawled=0, clusters_found=0,
                visited_urls=[], timing=streamer.summary(),
                query_type=query_type,
            )

        # Check pipeline timeout.
        if time.time() - pipeline_start > dr.pipeline_timeout_min * 60:
            console.print(f"[red]Pipeline timeout ({dr.pipeline_timeout_min} min). Aborting.[/red]")
            raise TimeoutError("Pipeline timeout exceeded")

        # =========================================================================
        # Stage 3: Cluster Round 1
        # =========================================================================
        streamer.stage_start("cluster1", "Cluster Round 1")
        clusters = cluster_pages(all_pages, index, threshold=dr.cluster_threshold)
        streamer.stage_done("cluster1", {
            "clusters": len(clusters),
        })

        # =========================================================================
        # Stage 4-5: Gap Detection + Re-crawl (if enabled)
        # =========================================================================
        gaps: list[str] = []
        if dr.enable_gap_detection and clusters:
            for gap_round in range(dr.gap_research_rounds):
                streamer.stage_start(f"gap{gap_round+1}", f"Gap Detection Round {gap_round+1}")
                gaps = detect_gaps(llm, clusters, user_query, query_type)
                streamer.stage_done(f"gap{gap_round+1}", {
                    "gaps": len(gaps),
                })

                if not gaps:
                    break

                streamer.stage_start(f"recrawl{gap_round+1}", f"Re-crawl for Gaps Round {gap_round+1}")
                gap_crawl = await bulk_crawl(
                    provider=provider,
                    queries=gaps,
                    max_pages=500,
                    concurrency=dr.crawl_concurrency,
                    crawl_depth=1,
                    query_type=query_type,
                    enable_quality_filter=dr.enable_quality_filter,
                    min_quality=dr.min_page_quality,
                    streamer=streamer,
                    stage_name=f"recrawl{gap_round+1}",
                    store=store,
                    index=index,
                )
                streamer.stage_done(f"recrawl{gap_round+1}", {
                    "pages": gap_crawl.success_count,
                })

                all_pages.extend(gap_crawl.pages)
                all_urls.extend(gap_crawl.urls_fetched)
                streamer.stage_start(f"recluster{gap_round+1}", f"Re-cluster Round {gap_round+1}")
                clusters = cluster_pages(all_pages, index, threshold=dr.cluster_threshold)
                streamer.stage_done(f"recluster{gap_round+1}", {
                    "clusters": len(clusters),
                })

        # =========================================================================
        # Stage 6: Batch Summarize
        # =========================================================================
        streamer.stage_start("summarize", "Batch Summarize")
        clusters = batch_summarize(
            clusters, llm,
            max_tokens=dr.batch_summarize_max_tokens,
            streamer=streamer,
        )
        streamer.stage_done("summarize", {
            "clusters": len(clusters),
        })

        # =========================================================================
        # Stage 7: Hierarchical Compression
        # =========================================================================
        streamer.stage_start("compress", "Hierarchical Compression")
        brief = hierarchical_compress(
            clusters, llm, user_query,
            max_words=dr.master_brief_max_words,
            streamer=streamer,
        )
        streamer.stage_done("compress", {
            "brief_words": len(brief.text.split()),
            "code_examples": len(brief.code_examples),
        })

        # =========================================================================
        # Stage 8: Two-Pass Implementation (swaps models via ModelManager)
        # =========================================================================
        streamer.stage_start("implement", "Two-Pass Implementation")
        plan, answer = two_pass_implement(
            model_mgr=model_mgr,
            user_query=user_query,
            brief=brief,
            streamer=streamer,
            temperature=cfg.agent.final_temperature,
        )
        streamer.stage_done("implement", {
            "plan_chars": len(plan),
            "answer_chars": len(answer),
        })

        # =========================================================================
        # Stage 9: Self-Critique (reload Instruct if Coder was used)
        # =========================================================================
        if cfg.agent.enable_self_critique and answer:
            # Ensure Instruct is loaded for critique.
            model_mgr.load_instruct()
            llm = model_mgr.current()
            streamer.stage_start("critique", "Self-Critique")
            answer = _critique_and_refine(
                llm, user_query, answer, brief.text,
                cfg.agent.critique_rounds, streamer,
            )
            streamer.stage_done("critique", {})

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Saving partial results...[/yellow]")
        answer = answer if 'answer' in dir() else "(interrupted)"
        plan = plan if 'plan' in dir() else ""
        brief_text = brief.text if 'brief' in dir() else ""
        clusters_count = len(clusters) if 'clusters' in dir() else 0
    except TimeoutError:
        console.print("\n[yellow]Pipeline timed out. Saving partial results...[/yellow]")
        answer = answer if 'answer' in dir() else "(timeout)"
        plan = plan if 'plan' in dir() else ""
        brief_text = brief.text if 'brief' in dir() else ""
        clusters_count = len(clusters) if 'clusters' in dir() else 0
    else:
        brief_text = brief.text
        clusters_count = len(clusters)

    # =========================================================================
    # Stage 10: Save Trajectory + Answer (unified format)
    # =========================================================================
    streamer.stage_start("save", "Save Trajectory")
    timing = streamer.summary()

    doc_ids = list(store.meta.docs.keys())
    code_ids = list(store.meta.code.keys())

    _save_trajectory_unified(
        cfg, query_id, user_query, answer=answer,
        plan=plan if 'plan' in dir() else "",
        brief=brief_text if 'brief_text' in dir() else "",
        mode="deep", query_type=query_type if 'query_type' in dir() else "general",
        pages_crawled=len(all_pages), clusters=clusters_count,
        gaps=gaps if 'gaps' in dir() else [],
        visited_urls=all_urls, doc_ids=doc_ids, code_ids=code_ids,
        timing=timing,
    )

    # Save answer.
    answer_path = cfg.answers_dir / f"{query_id}.md"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(
        f"# Query\n{user_query}\n\n# Implementation Plan\n{plan if 'plan' in dir() else ''}\n\n"
        f"# Answer\n{answer}\n\n# Research Brief\n{brief_text if 'brief_text' in dir() else ''}\n\n"
        f"# Research Stats\n"
        f"Pages crawled: {len(all_pages)}\n"
        f"Clusters: {clusters_count}\n"
        f"Code examples: {len(brief.code_examples) if 'brief' in dir() else 0}\n"
        f"Gaps detected: {len(gaps) if 'gaps' in dir() else 0}\n\n# Sources\n"
        + "\n".join(f"- {u}" for u in all_urls[:100]),
        "utf-8",
    )
    streamer.stage_done("save", {"trajectory": str(cfg.ultres_dir / "trajectories" / f"{query_id}.jsonl")})

    # Display summary.
    streamer.display_summary()

    return DeepResearchResult(
        answer=answer,
        query_id=query_id,
        plan=plan if 'plan' in dir() else "",
        brief=brief_text if 'brief_text' in dir() else "",
        pages_crawled=len(all_pages),
        clusters_found=clusters_count,
        gaps_detected=gaps if 'gaps' in dir() else [],
        visited_urls=all_urls,
        timing=timing,
        query_type=query_type if 'query_type' in dir() else "general",
    )


def _critique_and_refine(
    llm: Any,
    user_query: str,
    answer: str,
    brief_text: str,
    rounds: int,
    streamer: ResearchStreamer | None = None,
) -> str:
    """Critique the answer against the research brief and refine if needed.

    Only replaces the answer if the critique contains actual code blocks
    that represent a corrected implementation. Does NOT replace with raw
    critique prose.
    """
    import re as _re

    for round_num in range(rounds):
        prompt = (
            f"You are critiquing an implementation for: {user_query}\n\n"
            f"Research brief:\n{brief_text[:8000]}\n\n"
            f"Implementation:\n{answer[:8000]}\n\n"
            f"Does the implementation follow the research-derived plan? "
            f"Are there research-identified pitfalls not handled? "
            f"If the implementation is good, respond with ONLY 'VERIFIED'. "
            f"If there are issues, describe them briefly, then provide the "
            f"COMPLETE corrected implementation in a single code block."
        )
        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a rigorous code reviewer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            critique = resp["choices"][0]["message"]["content"].strip()

            # Check for VERIFIED verdict.
            if critique.upper().startswith("VERIFIED"):
                if streamer and streamer.enabled:
                    streamer.print("[green]Critique: implementation verified.[/green]")
                return answer

            # Extract code blocks from the critique.
            code_blocks = _re.findall(r"```[\w]*\n(.*?)```", critique, _re.DOTALL)
            if code_blocks:
                # Use the largest code block as the corrected implementation.
                best_block = max(code_blocks, key=len)
                if len(best_block) > len(answer) * 0.3:
                    if streamer and streamer.enabled:
                        streamer.print("[yellow]Critique: refining implementation with corrected code...[/yellow]")
                    return best_block.strip()
            else:
                if streamer and streamer.enabled:
                    streamer.print("[dim]Critique: no code corrections needed.[/dim]")
        except Exception:
            pass
    return answer


def _save_trajectory_unified(
    cfg: Any,
    query_id: str,
    user_query: str,
    answer: str,
    plan: str,
    brief: str,
    mode: str,
    query_type: str,
    pages_crawled: int,
    clusters: int,
    gaps: list[str],
    visited_urls: list[str],
    doc_ids: list[str],
    code_ids: list[str],
    timing: dict[str, Any],
) -> None:
    """Save a trajectory in the unified v1.4 format.

    Both fast and deep modes use this same schema for v1.5 QLoRA compatibility.
    """
    traj_dir = cfg.ultres_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{query_id}.jsonl"
    record = {
        "query_id": query_id,
        "query": user_query,
        "mode": mode,  # "fast" or "deep"
        "query_type": query_type,  # "code", "general", or "mixed"
        "answer": answer,
        "plan": plan or None,  # deep mode only (Pass 1 plan)
        "brief": brief or None,  # deep mode only (master brief)
        "steps": None,  # fast mode only, set by fast loop
        "pages_crawled": pages_crawled,
        "clusters": clusters if mode == "deep" else None,
        "gaps": gaps if mode == "deep" else None,
        "visited_urls": visited_urls,
        "doc_ids": doc_ids,
        "code_ids": code_ids,
        "timing": timing,
        "timestamp": time.time(),
    }
    with traj_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
