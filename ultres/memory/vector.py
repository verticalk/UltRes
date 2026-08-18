"""Vector index (Chroma, in-process, persistent on disk).

Indexes the structured notes + topic/subtopic summaries for the current query.
The agent calls `recall(query, k)` to pull semantically-relevant slices back
into the hot context window.

Uses an ONNX MiniLM embedding model by default so it runs on CPU with no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


@dataclass
class RecallHit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorIndex:
    """Per-query Chroma collection for semantic recall."""

    def __init__(
        self,
        index_dir: Path,
        query_id: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_function=None,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.query_id = query_id
        self.embedding_model = embedding_model
        self._client = chromadb.PersistentClient(
            path=str(self.index_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        # One collection per query so recalls are scoped.
        # If a custom embedding_function is provided (e.g. for tests), use it;
        # otherwise fall back to Chroma's bundled ONNX MiniLM (downloads on first use).
        collection_kwargs: dict[str, Any] = {"metadata": {"hnsw:space": "cosine"}}
        if embedding_function is not None:
            collection_kwargs["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(
            name=f"ultres_{query_id}",
            **collection_kwargs,
        )

    # -- ingest --------------------------------------------------------

    def add_note(self, doc_id: str, text: str, url: str = "", title: str = "") -> None:
        if not text.strip():
            return
        chunk_id = f"note:{doc_id}"
        self._collection.upsert(
            ids=[chunk_id],
            documents=[text],
            metadatas=[{"kind": "note", "doc_id": doc_id, "url": url, "title": title}],
        )

    def add_summary(
        self,
        summary_id: str,
        text: str,
        kind: str = "topic",
        name: str = "",
    ) -> None:
        if not text.strip():
            return
        chunk_id = f"summary:{kind}:{summary_id}"
        self._collection.upsert(
            ids=[chunk_id],
            documents=[text],
            metadatas=[{"kind": f"summary:{kind}", "summary_id": summary_id, "name": name}],
        )

    def add_code(self, code_id: str, text: str, language: str | None = None) -> None:
        if not text.strip():
            return
        # Index a truncated version for recall; full content is fetched via load_code.
        snippet = text[:2000]
        chunk_id = f"code:{code_id}"
        self._collection.upsert(
            ids=[chunk_id],
            documents=[snippet],
            metadatas=[{"kind": "code", "code_id": code_id, "language": language or ""}],
        )

    # -- query ---------------------------------------------------------

    def recall(self, query: str, k: int = 8) -> list[RecallHit]:
        if self._collection.count() == 0:
            return []
        res = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )
        hits: list[RecallHit] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            hits.append(
                RecallHit(
                    chunk_id=cid,
                    text=doc or "",
                    score=score,
                    metadata=meta or {},
                )
            )
        return hits

    def recall_filtered(
        self,
        query: str,
        k: int = 8,
        exclude_kinds: set[str] | None = None,
    ) -> list[RecallHit]:
        """Recall but filter out chunks whose 'kind' metadata is in exclude_kinds.

        Used to exclude temporary crawl docs from recall results.
        """
        if exclude_kinds is None:
            exclude_kinds = set()
        # Fetch more than k so we have enough after filtering.
        fetch_k = min(k * 3, self._collection.count()) if self._collection.count() > 0 else 0
        if fetch_k == 0:
            return []
        res = self._collection.query(
            query_texts=[query],
            n_results=fetch_k,
        )
        hits: list[RecallHit] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            kind = (meta or {}).get("kind", "")
            if kind in exclude_kinds:
                continue
            score = 1.0 - float(dist)
            hits.append(
                RecallHit(
                    chunk_id=cid,
                    text=doc or "",
                    score=score,
                    metadata=meta or {},
                )
            )
            if len(hits) >= k:
                break
        return hits

    def delete_by_kind(self, kind: str) -> int:
        """Delete all chunks with the given 'kind' metadata value.

        Used to clean up temporary crawl docs after clustering.
        Returns the number of deleted chunks.
        """
        try:
            # Get all IDs with this kind.
            results = self._collection.get(where={"kind": kind})
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def delete_by_id_prefix(self, prefix: str) -> int:
        """Delete all chunks whose ID starts with the given prefix.

        Used to clean up temporary crawl docs (IDs like 'crawl:...').
        """
        try:
            results = self._collection.get()
            ids = results.get("ids", [])
            to_delete = [i for i in ids if i.startswith(prefix)]
            if to_delete:
                self._collection.delete(ids=to_delete)
            return len(to_delete)
        except Exception:
            return 0

    # -- lifecycle -----------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop this query's collection (used by `ultres clean`)."""
        try:
            self._client.delete_collection(self._collection.name)
        except Exception:
            pass
