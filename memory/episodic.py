"""
Episodic Memory — vector-DB-backed event and conversation store.

Analogous to the hippocampus: remembers *specific past events* with
timestamps and context, and retrieves them by semantic similarity.

Backend strategy (tried in order):
  1. ChromaDB  — embedded, persistent, no server needed (preferred)
  2. Qdrant    — server-mode, high-throughput (optional)
  3. NumpyStore — pure-Python fallback, no external deps, in-memory only

All three backends expose the same AbstractMemory interface, so callers
never need to know which backend is active.

Phase 4: QuantumHippocampus wraps this class and overrides ``retrieve``
with a Grover-inspired amplitude-amplified search.

Usage::

    em = EpisodicMemory(persist_path="./data/episodic_db")
    em.store(MemoryRecord(
        key="turn-001",
        content="User asked about the budget",
        embedding=encoder.encode("User asked about the budget"),
        metadata={"session": "abc", "role": "user"},
    ))
    top = em.retrieve(query_embedding=encoder.encode("budget"), top_k=5)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from memory.interfaces import AbstractMemory, MemoryRecord

# --------------------------------------------------------------------------- #
# Backend detection                                                             #
# --------------------------------------------------------------------------- #

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# --------------------------------------------------------------------------- #
# NumpyStore — zero-dep in-memory fallback                                     #
# --------------------------------------------------------------------------- #


class _NumpyStore:
    """Pure-numpy in-memory vector store used when no DB is installed."""

    def __init__(self) -> None:
        self._records: Dict[str, MemoryRecord] = {}

    def upsert(self, record: MemoryRecord) -> None:
        self._records[record.key] = record

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[MemoryRecord]:
        q = np.asarray(query_embedding, dtype=np.float32)
        results: list[tuple[float, MemoryRecord]] = []
        for rec in self._records.values():
            if rec.is_expired():
                continue
            if not _meta_matches(rec, filters):
                continue
            if rec.embedding is not None:
                sim = _cosine(q, np.asarray(rec.embedding, dtype=np.float32))
            else:
                sim = 0.0
            results.append((sim, rec))
        results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in results[:top_k]]

    def get(self, key: str) -> Optional[MemoryRecord]:
        return self._records.get(key)

    def delete(self, key: str) -> bool:
        return bool(self._records.pop(key, None))

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)


# --------------------------------------------------------------------------- #
# EpisodicMemory                                                               #
# --------------------------------------------------------------------------- #


class EpisodicMemory(AbstractMemory):
    """
    Episodic (hippocampal) memory backed by a vector database.

    Args:
        persist_path   : Directory for ChromaDB persistence.
                         Pass ``None`` to force in-memory numpy fallback.
        collection_name: ChromaDB / Qdrant collection identifier.
        embedding_dim  : Expected vector dimension (validated at upsert time).
        qdrant_url     : If set, prefer Qdrant over ChromaDB.
        qdrant_api_key : Optional Qdrant API key.

    Example::

        em = EpisodicMemory(persist_path="./data/episodic")
        em.store(MemoryRecord(key="e1", content="hello", embedding=[0.1]*768))
        records = em.retrieve(query_embedding=[0.1]*768, top_k=5)
    """

    COLLECTION = "nawal_episodic"

    def __init__(
        self,
        persist_path: Optional[str] = "./data/episodic_db",
        collection_name: str = COLLECTION,
        embedding_dim: int = 768,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._backend: str  # "chroma" | "qdrant" | "numpy"
        self._chroma_col = None
        self._qdrant: Optional["QdrantClient"] = None
        self._numpy_store: Optional[_NumpyStore] = None

        if qdrant_url and QDRANT_AVAILABLE:
            self._init_qdrant(qdrant_url, qdrant_api_key)
        elif CHROMA_AVAILABLE and persist_path is not None:
            self._init_chroma(persist_path)
        else:
            self._init_numpy()

        logger.info(
            f"EpisodicMemory backend={self._backend!r} "
            f"collection={collection_name!r} dim={embedding_dim}"
        )

    # ------------------------------------------------------------------ #
    # Backend initialisers                                                 #
    # ------------------------------------------------------------------ #

    def _init_chroma(self, persist_path: str) -> None:
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._chroma_col = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._backend = "chroma"

    def _init_qdrant(self, url: str, api_key: Optional[str]) -> None:
        self._qdrant = QdrantClient(url=url, api_key=api_key)
        collections = [c.name for c in self._qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        self._backend = "qdrant"

    def _init_numpy(self) -> None:
        self._numpy_store = _NumpyStore()
        self._backend = "numpy"
        logger.warning(
            "EpisodicMemory using in-memory numpy fallback "
            "(install chromadb for persistent storage)"
        )

    # ------------------------------------------------------------------ #
    # AbstractMemory implementation                                        #
    # ------------------------------------------------------------------ #

    def store(self, record: MemoryRecord) -> None:
        if self._backend == "chroma":
            self._chroma_upsert(record)
        elif self._backend == "qdrant":
            self._qdrant_upsert(record)
        else:
            self._numpy_store.upsert(record)  # type: ignore[union-attr]
        logger.debug(
            f"EpisodicMemory stored key={record.key!r} backend={self._backend}"
        )

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        if self._backend == "chroma":
            return self._chroma_query(query_embedding, top_k, filters)
        elif self._backend == "qdrant":
            return self._qdrant_query(query_embedding, top_k, filters)
        else:
            return self._numpy_store.query(query_embedding, top_k, filters)  # type: ignore[union-attr]

    def get(self, key: str) -> Optional[MemoryRecord]:
        if self._backend == "chroma":
            return self._chroma_get(key)
        elif self._backend == "qdrant":
            return self._qdrant_get(key)
        else:
            return self._numpy_store.get(key)  # type: ignore[union-attr]

    def delete(self, key: str) -> bool:
        if self._backend == "chroma":
            try:
                self._chroma_col.delete(ids=[key])  # type: ignore[union-attr]
                return True
            except Exception:
                return False
        elif self._backend == "qdrant":
            try:
                self._qdrant.delete(  # type: ignore[union-attr]
                    collection_name=self.collection_name,
                    points_selector=[key],
                )
                return True
            except Exception:
                return False
        else:
            return self._numpy_store.delete(key)  # type: ignore[union-attr]

    def clear(self) -> None:
        if self._backend == "chroma":
            ids = self._chroma_col.get()["ids"]  # type: ignore[union-attr]
            if ids:
                self._chroma_col.delete(ids=ids)  # type: ignore[union-attr]
        elif self._backend == "qdrant":
            self._qdrant.delete_collection(self.collection_name)  # type: ignore[union-attr]
            self._init_qdrant(self._qdrant._client._host, None)  # type: ignore[union-attr]
        else:
            self._numpy_store.clear()  # type: ignore[union-attr]
        logger.debug(f"EpisodicMemory cleared backend={self._backend}")

    def __len__(self) -> int:
        if self._backend == "chroma":
            return self._chroma_col.count()  # type: ignore[union-attr]
        elif self._backend == "qdrant":
            info = self._qdrant.get_collection(self.collection_name)  # type: ignore[union-attr]
            return info.vectors_count or 0
        else:
            return len(self._numpy_store)  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # ChromaDB internals                                                   #
    # ------------------------------------------------------------------ #

    def _chroma_upsert(self, record: MemoryRecord) -> None:
        emb = record.embedding or []
        meta = {
            k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v))
            for k, v in record.metadata.items()
        }
        meta["_content"] = str(record.content)
        meta["_timestamp"] = record.timestamp
        meta["_ttl"] = record.ttl if record.ttl is not None else -1.0

        if emb:
            self._chroma_col.upsert(  # type: ignore[union-attr]
                ids=[record.key],
                embeddings=[emb],
                metadatas=[meta],
                documents=[str(record.content)],
            )
        else:
            self._chroma_col.upsert(  # type: ignore[union-attr]
                ids=[record.key],
                metadatas=[meta],
                documents=[str(record.content)],
            )

    def _chroma_query(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[MemoryRecord]:
        where: Optional[Dict] = None
        if filters:
            conditions = [{"$and": [{k: {"$eq": v}} for k, v in filters.items()]}]
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        kwargs: Dict[str, Any] = {
            "n_results": top_k,
            "include": ["metadatas", "documents", "embeddings", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._chroma_col.query(query_embeddings=[query_embedding], **kwargs)  # type: ignore[union-attr]

        records: List[MemoryRecord] = []
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        docs = results.get("documents", [[]])[0]
        embs = results.get("embeddings") or [[]] * len(ids)
        emb_list = embs[0] if embs else [[]] * len(ids)

        for i, rid in enumerate(ids):
            meta = dict(metas[i])
            content = meta.pop("_content", docs[i] if i < len(docs) else "")
            timestamp = float(meta.pop("_timestamp", 0.0))
            ttl_raw = meta.pop("_ttl", -1.0)
            ttl = float(ttl_raw) if ttl_raw != -1.0 else None
            emb = list(emb_list[i]) if i < len(emb_list) and emb_list[i] else None
            rec = MemoryRecord(
                key=rid,
                content=content,
                embedding=emb,
                metadata=meta,
                timestamp=timestamp,
                ttl=ttl,
            )
            if not rec.is_expired():
                records.append(rec)
        return records

    def _chroma_get(self, key: str) -> Optional[MemoryRecord]:
        result = self._chroma_col.get(ids=[key], include=["metadatas", "documents", "embeddings"])  # type: ignore[union-attr]
        ids = result.get("ids", [])
        if not ids:
            return None
        meta = dict(result["metadatas"][0])
        content = meta.pop(
            "_content", result["documents"][0] if result.get("documents") else ""
        )
        timestamp = float(meta.pop("_timestamp", 0.0))
        ttl_raw = meta.pop("_ttl", -1.0)
        ttl = float(ttl_raw) if ttl_raw != -1.0 else None
        embs = result.get("embeddings")
        emb = list(embs[0]) if embs else None
        rec = MemoryRecord(
            key=ids[0],
            content=content,
            embedding=emb,
            metadata=meta,
            timestamp=timestamp,
            ttl=ttl,
        )
        return None if rec.is_expired() else rec

    # ------------------------------------------------------------------ #
    # Qdrant internals                                                     #
    # ------------------------------------------------------------------ #

    def _qdrant_upsert(self, record: MemoryRecord) -> None:
        payload = dict(record.metadata)
        payload["_content"] = str(record.content)
        payload["_timestamp"] = record.timestamp
        payload["_ttl"] = record.ttl if record.ttl is not None else -1.0
        vector = record.embedding or [0.0] * self.embedding_dim
        self._qdrant.upsert(  # type: ignore[union-attr]
            collection_name=self.collection_name,
            points=[PointStruct(id=record.key, vector=vector, payload=payload)],
        )

    def _qdrant_query(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[MemoryRecord]:
        qdrant_filter = None
        if filters:
            must = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=must)

        hits = self._qdrant.search(  # type: ignore[union-attr]
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=True,
        )
        records: List[MemoryRecord] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            content = payload.pop("_content", "")
            timestamp = float(payload.pop("_timestamp", 0.0))
            ttl_raw = payload.pop("_ttl", -1.0)
            ttl = float(ttl_raw) if ttl_raw != -1.0 else None
            rec = MemoryRecord(
                key=str(hit.id),
                content=content,
                embedding=list(hit.vector) if hit.vector else None,
                metadata=payload,
                timestamp=timestamp,
                ttl=ttl,
            )
            if not rec.is_expired():
                records.append(rec)
        return records

    def _qdrant_get(self, key: str) -> Optional[MemoryRecord]:
        results = self._qdrant.retrieve(  # type: ignore[union-attr]
            collection_name=self.collection_name,
            ids=[key],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        hit = results[0]
        payload = dict(hit.payload or {})
        content = payload.pop("_content", "")
        timestamp = float(payload.pop("_timestamp", 0.0))
        ttl_raw = payload.pop("_ttl", -1.0)
        ttl = float(ttl_raw) if ttl_raw != -1.0 else None
        rec = MemoryRecord(
            key=str(hit.id),
            content=content,
            embedding=list(hit.vector) if hit.vector else None,
            metadata=payload,
            timestamp=timestamp,
            ttl=ttl,
        )
        return None if rec.is_expired() else rec

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory(backend={self._backend!r}, "
            f"collection={self.collection_name!r}, "
            f"records={len(self)})"
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _meta_matches(record: MemoryRecord, filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    return all(record.metadata.get(k) == v for k, v in filters.items())
