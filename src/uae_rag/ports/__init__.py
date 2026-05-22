"""Ports package — interfaces the domain layer depends on (ADR-0002).

Re-exports the protocols so consumers can write a single import:

    from uae_rag.ports import (
        EmbeddingsPort, VectorIndexPort, VectorRecord, QueryHit,
        RetrievalPort, RetrievalHit, RetrievalSource,
        RerankerPort,
        LLMPort, LLMUnavailableError,
    )
"""

from __future__ import annotations

from uae_rag.ports.embeddings import EmbeddingsPort
from uae_rag.ports.llm import LLMPort, LLMUnavailableError
from uae_rag.ports.reranker import RerankerPort
from uae_rag.ports.retrieval import RetrievalHit, RetrievalPort, RetrievalSource
from uae_rag.ports.vector_index import QueryHit, VectorIndexPort, VectorRecord

__all__ = [
    "EmbeddingsPort",
    "LLMPort",
    "LLMUnavailableError",
    "QueryHit",
    "RerankerPort",
    "RetrievalHit",
    "RetrievalPort",
    "RetrievalSource",
    "VectorIndexPort",
    "VectorRecord",
]
