"""Dense retriever — glue over ``EmbeddingsPort`` and ``VectorIndexPort``.

Per ADR-0004, the dense leg supplies semantic recall; the hybrid retriever fuses
it with the BM25 leg's lexical recall via RRF. Per ADR-0002, this module imports
only from ``ports/`` — never from ``adapters/``.
"""

from __future__ import annotations

from uae_rag.ports.embeddings import EmbeddingsPort
from uae_rag.ports.retrieval import RetrievalHit
from uae_rag.ports.vector_index import VectorIndexPort


class DenseRetriever:
    """Embed the query and look it up in the vector index."""

    def __init__(
        self,
        *,
        embedder: EmbeddingsPort,
        vector_index: VectorIndexPort,
    ) -> None:
        self._embedder = embedder
        self._index = vector_index

    def retrieve(self, query: str, *, top_k: int = 50) -> list[RetrievalHit]:
        embedding = self._embedder.embed_query(query)
        query_hits = self._index.query(embedding, top_k=top_k)
        return [
            RetrievalHit(
                chunk_id=h.id,
                text=h.document,
                metadata=dict(h.metadata),
                score=float(h.score),
                rank=rank,
                source="dense",
            )
            for rank, h in enumerate(query_hits, start=1)
        ]


__all__ = ["DenseRetriever"]
