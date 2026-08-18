"""Per-project LoRA adapter cache (v1: loader stub, v2: trains).

v1: scans `.ultres/adapters/*.safetensors`; if an adapter's metadata matches
the current query topic (cosine on topic embedding >= threshold), it's loaded
via llama.cpp's `lora_path`. v1 ships NO training path — adapters are only
loaded if present (e.g. placed manually or produced by v2).

v2 will add `ultres --train-adapter --topic <name>` which QLoRA-trains a small
adapter on the accumulated research + Q&A for a topic, on the user's GPU or
Kaggle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AdapterMeta:
    name: str
    topic: str
    base_model: str
    created_at: float
    path: Path
    # Cosine similarity threshold for matching (default 0.7).
    match_threshold: float = 0.7


def list_adapters(adapters_dir: Path) -> list[AdapterMeta]:
    """List all `.safetensors` adapters in `adapters_dir` with their metadata."""
    adapters_dir = Path(adapters_dir)
    if not adapters_dir.exists():
        return []
    out: list[AdapterMeta] = []
    for p in sorted(adapters_dir.glob("*.safetensors")):
        meta_path = p.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text("utf-8"))
            except json.JSONDecodeError:
                pass
        out.append(
            AdapterMeta(
                name=meta.get("name", p.stem),
                topic=meta.get("topic", p.stem),
                base_model=meta.get("base_model", ""),
                created_at=float(meta.get("created_at", 0.0)),
                path=p,
                match_threshold=float(meta.get("match_threshold", 0.7)),
            )
        )
    return out


def topic_similarity(a: str, b: str) -> float:
    """Cheap token-overlap similarity for topic matching (v1).

    v2 will replace this with real embeddings via the VectorIndex.
    """
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_matching_adapter(
    adapters_dir: Path,
    topic: str,
) -> AdapterMeta | None:
    """Return the best-matching adapter for `topic`, or None."""
    adapters = list_adapters(adapters_dir)
    best: AdapterMeta | None = None
    best_score = 0.0
    for a in adapters:
        score = topic_similarity(a.topic, topic)
        if score > best_score:
            best_score = score
            best = a
    if best and best_score >= best.match_threshold:
        return best
    return None


def lora_path_for_query(adapters_dir: Path, topic: str) -> str | None:
    """Return a path string suitable for llama.cpp's `lora_path` arg, or None.

    v1: only loads if a matching adapter is already on disk. No training.
    """
    match = find_matching_adapter(adapters_dir, topic)
    return str(match.path) if match else None


# ---------------------------------------------------------------------------
# Trajectory accumulation for v1.5 QLoRA training
# ---------------------------------------------------------------------------

def list_trajectories(trajectories_dir: Path) -> list[dict[str, Any]]:
    """List all saved research trajectories in .ultres/trajectories/.

    Each trajectory is a JSONL file with (query, answer, steps, visited_urls,
    doc_ids, code_ids). These become the training dataset for v1.5 QLoRA.
    """
    trajectories_dir = Path(trajectories_dir)
    if not trajectories_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(trajectories_dir.glob("*.jsonl")):
        try:
            import json as _json

            record = _json.loads(p.read_text("utf-8").strip().splitlines()[0])
            out.append(record)
        except Exception:
            continue
    return out


def trajectory_count(trajectories_dir: Path) -> int:
    """Count saved trajectories (for display in CLI)."""
    trajectories_dir = Path(trajectories_dir)
    if not trajectories_dir.exists():
        return 0
    return len(list(trajectories_dir.glob("*.jsonl")))


# ---------------------------------------------------------------------------
# v1.4: Trajectory export, validation, and statistics for v1.5 QLoRA
# ---------------------------------------------------------------------------

def export_trajectories(
    trajectories_dir: Path,
    output_path: Path,
    format: str = "jsonl",
) -> int:
    """Export trajectories as training-ready JSONL for v1.5 QLoRA.

    Produces instruction/response pairs suitable for Unsloth/Qwen QLoRA:
        {"instruction": "...", "input": "", "output": "...", "context": "..."}

    For deep mode trajectories, the brief is included as context.
    For fast mode trajectories, the visited URLs are included as context.

    Args:
        trajectories_dir: Directory containing .jsonl trajectory files.
        output_path: Where to write the exported dataset.
        format: "jsonl" (default) or "json" (list format).

    Returns:
        Number of exported examples.
    """
    trajectories = list_trajectories(trajectories_dir)
    examples: list[dict[str, Any]] = []
    for traj in trajectories:
        query = traj.get("query", "")
        answer = traj.get("answer", "")
        if not query or not answer:
            continue
        # Build context from brief (deep mode) or URLs (fast mode).
        context_parts: list[str] = []
        if traj.get("brief"):
            context_parts.append(f"Research brief:\n{traj['brief']}")
        if traj.get("plan"):
            context_parts.append(f"Implementation plan:\n{traj['plan']}")
        if traj.get("visited_urls"):
            urls = traj["visited_urls"][:20]
            context_parts.append("Sources:\n" + "\n".join(f"- {u}" for u in urls))
        context = "\n\n".join(context_parts)
        examples.append({
            "instruction": query,
            "input": "",
            "output": answer,
            "context": context,
            "mode": traj.get("mode", "unknown"),
            "query_type": traj.get("query_type", "unknown"),
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        import json as _json
        output_path.write_text(_json.dumps(examples, indent=2), "utf-8")
    else:  # jsonl
        with output_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
    return len(examples)


def validate_trajectory(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single trajectory record for training readiness.

    Returns (is_valid, list_of_issues).
    """
    issues: list[str] = []
    # Required fields.
    required = ["query_id", "query", "answer", "mode", "timestamp"]
    for field_name in required:
        if field_name not in record:
            issues.append(f"missing field: {field_name}")
    # Non-empty answer.
    if record.get("answer") and len(record["answer"].strip()) < 10:
        issues.append("answer too short (<10 chars)")
    # Non-empty query.
    if record.get("query") and len(record["query"].strip()) < 3:
        issues.append("query too short (<3 chars)")
    # Minimum content.
    mode = record.get("mode", "")
    if mode == "fast":
        if not record.get("visited_urls"):
            issues.append("fast mode: no visited URLs")
    elif mode == "deep":
        pages = record.get("pages_crawled", 0)
        if pages < 10:
            issues.append(f"deep mode: only {pages} pages crawled (min 10)")
    else:
        issues.append(f"unknown mode: {mode}")
    return (len(issues) == 0, issues)


def validate_all_trajectories(
    trajectories_dir: Path,
) -> list[tuple[str, bool, list[str]]]:
    """Validate all trajectories in a directory.

    Returns list of (query_id, is_valid, issues) tuples.
    """
    trajectories = list_trajectories(trajectories_dir)
    results: list[tuple[str, bool, list[str]]] = []
    for traj in trajectories:
        qid = traj.get("query_id", "unknown")
        is_valid, issues = validate_trajectory(traj)
        results.append((qid, is_valid, issues))
    return results


def trajectory_stats(trajectories_dir: Path) -> dict[str, Any]:
    """Compute statistics over all trajectories.

    Returns a dict with:
        - total: total count
        - by_mode: {"fast": N, "deep": N}
        - by_query_type: {"code": N, "general": N, "mixed": N}
        - total_pages: sum of pages_crawled
        - avg_pages_deep: average pages per deep query
        - avg_answer_chars: average answer length in chars
        - total_estimated_tokens: estimated total tokens
    """
    trajectories = list_trajectories(trajectories_dir)
    stats: dict[str, Any] = {
        "total": len(trajectories),
        "by_mode": {"fast": 0, "deep": 0, "unknown": 0},
        "by_query_type": {"code": 0, "general": 0, "mixed": 0, "unknown": 0},
        "total_pages": 0,
        "avg_pages_deep": 0.0,
        "avg_answer_chars": 0.0,
        "total_estimated_tokens": 0,
    }
    deep_count = 0
    deep_pages = 0
    total_answer_chars = 0
    total_tokens = 0
    for traj in trajectories:
        mode = traj.get("mode", "unknown")
        stats["by_mode"][mode] = stats["by_mode"].get(mode, 0) + 1
        qtype = traj.get("query_type", "unknown")
        stats["by_query_type"][qtype] = stats["by_query_type"].get(qtype, 0) + 1
        pages = traj.get("pages_crawled", 0) or 0
        stats["total_pages"] += pages
        if mode == "deep":
            deep_count += 1
            deep_pages += pages
        answer = traj.get("answer", "") or ""
        total_answer_chars += len(answer)
        # Rough token estimate (~4 chars/token).
        total_tokens += len(answer) // 4
        if traj.get("brief"):
            total_tokens += len(traj["brief"]) // 4
        if traj.get("plan"):
            total_tokens += len(traj["plan"]) // 4
    if deep_count > 0:
        stats["avg_pages_deep"] = deep_pages / deep_count
    if len(trajectories) > 0:
        stats["avg_answer_chars"] = total_answer_chars / len(trajectories)
    stats["total_estimated_tokens"] = total_tokens
    return stats
