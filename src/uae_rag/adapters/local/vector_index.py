"""Local vector index adapter — ChromaDB PersistentClient.

Implements ``VectorIndexPort`` (ports/vector_index.py). Stores the
embedder's ``model_id`` and ``dimension`` in the collection's metadata at
first upsert so that a later constructor pointing at the same collection
can detect an incompatible swap and refuse to corrupt the index.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb

from uae_rag.ports.vector_index import QueryHit, VectorRecord

logger = logging.getLogger(__name__)

_METADATA_DIM_KEY = "embedding_dim"
_METADATA_MODEL_KEY = "embedding_model_id"
_HNSW_SPACE_KEY = "hnsw:space"
_HNSW_SPACE_VALUE = "cosine"


class DimensionMismatchError(RuntimeError):
    """Raised when an embedder's dimension doesn't match the on-disk index's stored dim."""


class ChromaVectorIndex:
    """Persistent vector index backed by a ChromaDB collection.

    The constructor reconciles the embedder's identity (``embedder_model_id``,
    ``embedder_dimension``) against the collection's stored metadata:

    - Collection not yet populated → metadata is written on the first upsert.
    - Stored dimension differs → ``DimensionMismatchError`` immediately.
    - Stored dimension matches, model id differs → WARNING, proceed.
    - Both match → silent.
    """

    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str,
        embedder_model_id: str,
        embedder_dimension: int,
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name
        self._embedder_model_id = embedder_model_id
        self._embedder_dimension = embedder_dimension

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={_HNSW_SPACE_KEY: _HNSW_SPACE_VALUE},
        )
        self._reconcile_metadata()

    def _reconcile_metadata(self) -> None:
        stored = self._collection.metadata or {}
        stored_dim = stored.get(_METADATA_DIM_KEY)
        stored_model = stored.get(_METADATA_MODEL_KEY)

        if stored_dim is None:
            return  # fresh collection — metadata is written on first upsert.

        if int(stored_dim) != self._embedder_dimension:
            raise DimensionMismatchError(
                f"Embedder dimension {self._embedder_dimension} != "
                f"index dimension {stored_dim}. The current index was built "
                f"with a different embedding model. "
                f"Re-run with --reset to drop and rebuild."
            )

        if stored_model and stored_model != self._embedder_model_id:
            logger.warning(
                "model id changed for collection %s (stored=%s, current=%s); "
                "vector space may have shifted — consider --reset",
                self._collection_name,
                stored_model,
                self._embedder_model_id,
            )

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        batch = list(records)
        if not batch:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for r in batch:
            if len(r.embedding) != self._embedder_dimension:
                raise DimensionMismatchError(
                    f"Embedding length {len(r.embedding)} != "
                    f"expected dimension {self._embedder_dimension} (record id={r.id})."
                )
            ids.append(r.id)
            embeddings.append(r.embedding)
            documents.append(r.document)
            metadatas.append(dict(r.metadata))

        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        self._ensure_metadata_recorded()

    def _ensure_metadata_recorded(self) -> None:
        current = self._collection.metadata or {}
        if (
            current.get(_METADATA_DIM_KEY) == self._embedder_dimension
            and current.get(_METADATA_MODEL_KEY) == self._embedder_model_id
        ):
            return
        # ChromaDB rejects modify() if hnsw:space is still in the dict — the
        # distance function is fixed at creation. Strip it before persisting.
        merged = {k: v for k, v in current.items() if k != _HNSW_SPACE_KEY}
        merged[_METADATA_DIM_KEY] = self._embedder_dimension
        merged[_METADATA_MODEL_KEY] = self._embedder_model_id
        self._collection.modify(metadata=merged)

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[QueryHit]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            QueryHit(
                id=i,
                document=doc or "",
                metadata=dict(meta or {}),
                score=1.0 - float(dist),
            )
            for i, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=False)
        ]

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={_HNSW_SPACE_KEY: _HNSW_SPACE_VALUE},
        )


__all__ = ["ChromaVectorIndex", "DimensionMismatchError"]
