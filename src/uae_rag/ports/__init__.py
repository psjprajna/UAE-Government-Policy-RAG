"""Ports package — interfaces the domain layer depends on (ADR-0002).

Re-exports the two phase-3 protocols so consumers can write a single import:

    from uae_rag.ports import EmbeddingsPort, VectorIndexPort, VectorRecord, QueryHit
"""

from __future__ import annotations

from uae_rag.ports.embeddings import EmbeddingsPort
from uae_rag.ports.vector_index import QueryHit, VectorIndexPort, VectorRecord

__all__ = ["EmbeddingsPort", "QueryHit", "VectorIndexPort", "VectorRecord"]
