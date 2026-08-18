"""KV-cache reuse wrapper around llama.cpp.

Goal: persist KV tensors for retrieved chunks to disk and page them back in
when the same chunk is re-attended, avoiding recompute.

v1 status: best-effort. llama-cpp-python's KV cache API is limited; if it
doesn't expose clean save/restore, we fall back to plain re-inference and
document this as a v2 optimization. The hierarchical memory still works
without it — just slower on re-attention.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class KVCacheManager:
    """Best-effort on-disk KV cache for re-attended chunks.

    The actual KV tensor persistence depends on llama-cpp-python exposing
    `llama_get_kv_cache` / `llama_set_kv_cache` (or equivalent). As of late
    2026, the stable API does not expose this directly, so this manager
    currently acts as a no-op tracker: it records which chunks have been
    attended and their hashes, so a future implementation can wire in real
    persistence without changing call sites.
    """

    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        # chunk_id -> sha1(content) for dedup / invalidation
        self._seen: dict[str, str] = {}

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:16]

    def mark_attended(self, chunk_id: str, content: str) -> None:
        """Record that `chunk_id` with `content` has been attended."""
        if not self.enabled:
            return
        self._seen[chunk_id] = self._hash(content)

    def has_attended(self, chunk_id: str, content: str) -> bool:
        """True if the exact same content was attended before."""
        if not self.enabled:
            return False
        return self._seen.get(chunk_id) == self._hash(content)

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "chunks_seen": len(self._seen)}

    # The hooks below are placeholders for when llama-cpp-python exposes
    # KV save/restore. They are intentionally no-ops in v1.

    def save_kv(self, llama: Any, chunk_id: str) -> None:
        """(v2) Persist the current KV cache state for `chunk_id`."""
        return None

    def load_kv(self, llama: Any, chunk_id: str) -> bool:
        """(v2) Restore KV cache state for `chunk_id`. Returns True if restored."""
        return False
