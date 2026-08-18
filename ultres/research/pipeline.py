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


async def run_deep_research(
    llm: Any,
    llm_coder: Any | None,
    cfg: Any,
    user_query: str,
    provider: Any,
    console: Console | None = None,
    streamer: ResearchStreamer | None = None,
) -> DeepResearchResult:
    """Run the full deep research pipeline.

    Args:
        llm: Instruct model (for planning, summarizing, critique).
        llm_coder: Coder model (for implementation). If None, uses llm.
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

    # Set up per-query store + index.
    store = KnowledgeStore(cfg.store_dir, query_id=query_id)
    store.set_user_query(user_query)
    index = VectorIndex(cfg.index_dir, query_id=query_id, embedding_model=cfg.memory.embedding_model)

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
    # Stage 2: Bulk Crawl Round 1
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
    )
    streamer.stage_done("crawl1", {
        "pages": crawl_result.success_count,
        "failed": len(crawl_result.errors),
        "deduped": crawl_result.deduped,
    })

    all_pages = list(crawl_result.pages)
    all_urls = list(crawl_result.urls_fetched)

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
                crawl_depth=1,  # Shallower for gap fill.
                query_type=query_type,
                enable_quality_filter=dr.enable_quality_filter,
                min_quality=dr.min_page_quality,
                streamer=streamer,
            )
            streamer.stage_done(f"recrawl{gap_round+1}", {
                "pages": gap_crawl.success_count,
            })

            # Merge new pages and re-cluster.
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
    # Stage 8: Two-Pass Implementation
    # =========================================================================
    streamer.stage_start("implement", "Two-Pass Implementation")
    answer = two_pass_implement(
        llm_instruct=llm,
        llm_coder=llm_coder,
        user_query=user_query,
        brief=brief,
        streamer=streamer,
        temperature=cfg.agent.final_temperature,
    )
    streamer.stage_done("implement", {
        "answer_chars": len(answer),
    })

    # =========================================================================
    # Stage 9: Self-Critique (if enabled)
    # =========================================================================
    if cfg.agent.enable_self_critique and answer:
        streamer.stage_start("critique", "Self-Critique")
        answer = _critique_and_refine(
            llm, user_query, answer, brief.text,
            cfg.agent.critique_rounds, streamer,
        )
        streamer.stage_done("critique", {})

    # =========================================================================
    # Stage 10: Save Trajectory + Answer
    # =========================================================================
    streamer.stage_start("save", "Save Trajectory")
    timing = streamer.summary()

    # Save trajectory.
    traj_dir = cfg.ultres_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{query_id}.jsonl"
    record = {
        "query_id": query_id,
        "query": user_query,
        "answer": answer,
        "brief": brief.text,
        "pages_crawled": len(all_pages),
        "clusters": len(clusters),
        "gaps": gaps,
        "visited_urls": all_urls,
        "timing": timing,
        "timestamp": time.time(),
    }
    with traj_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Save answer.
    answer_path = cfg.answers_dir / f"{query_id}.md"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(
        f"# Query\n{user_query}\n\n# Answer\n{answer}\n\n# Research Stats\n"
        f"Pages crawled: {len(all_pages)}\n"
        f"Clusters: {len(clusters)}\n"
        f"Code examples: {len(brief.code_examples)}\n"
        f"Gaps detected: {len(gaps)}\n\n# Sources\n"
        + "\n".join(f"- {u}" for u in all_urls[:100]),
        "utf-8",
    )
    streamer.stage_done("save", {"trajectory": str(traj_path)})

    # Display summary.
    streamer.display_summary()

    return DeepResearchResult(
        answer=answer,
        query_id=query_id,
        plan=brief.text,
        brief=brief.text,
        pages_crawled=len(all_pages),
        clusters_found=len(clusters),
        gaps_detected=gaps,
        visited_urls=all_urls,
        timing=timing,
    )


def _critique_and_refine(
    llm: Any,
    user_query: str,
    answer: str,
    brief_text: str,
    rounds: int,
    streamer: ResearchStreamer | None = None,
) -> str:
    """Critique the answer against the research brief and refine if needed."""
    for round_num in range(rounds):
        prompt = (
            f"You are critiquing an implementation for: {user_query}\n\n"
            f"Research brief:\n{brief_text[:8000]}\n\n"
            f"Implementation:\n{answer[:8000]}\n\n"
            f"Does the implementation follow the research-derived plan? "
            f"Are there research-identified pitfalls not handled? "
            f"If the implementation is good, say 'VERIFIED'. "
            f"If there are issues, describe them and provide the corrected implementation."
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
            if critique.upper().startswith("VERIFIED"):
                if streamer and streamer.enabled:
                    streamer.print("[green]Critique: implementation verified.[/green]")
                return answer
            # Check if the critique contains a corrected implementation.
            if "```" in critique and len(critique) > len(answer) * 0.5:
                if streamer and streamer.enabled:
                    streamer.print("[yellow]Critique: refining implementation...[/yellow]")
                return critique
        except Exception:
            pass
    return answer
