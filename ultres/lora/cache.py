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
