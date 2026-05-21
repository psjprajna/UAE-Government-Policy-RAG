"""Port: cross-encoder reranker over a candidate hit list.

The domain layer depends on this protocol; concrete rerankers under
``uae_rag.adapters.*`` implement it. Per ADR-0002, no domain module imports
an adapter directly; per ADR-0007, the local implementation uses
``BAAI/bge-reranker-v2-m3`` via ``sentence_transformers.CrossEncoder``.

Modeling the reranker as its own port (instead of overloading ``RetrievalPort``)
keeps the candidate-set contract explicit: ``rerank(query, hits, top_k)`` is a
re-scorer over an existing list, not a primary retriever. The wrapper that
composes a ``RetrievalPort`` with a ``RerankerPort`` lives in
``uae_rag.retrieval.rerank`` and itself satisfies ``RetrievalPort`` so callers
remain agnostic to whether reranking is enabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from uae_rag.ports.retrieval import RetrievalHit


@runtime_checkable
class RerankerPort(Protocol):
    """Interface for cross-encoder reranking over a candidate hit list.

    Attributes:
        model_id: Provider-qualified model identifier (e.g. ``"BAAI/bge-reranker-v2-m3"``).
    """

    model_id: str

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        """Return at most ``top_k`` hits, re-scored, sorted by descending score.

        Returned hits carry ``source="reranked"`` and 1-based ``rank`` over the
        reranked subset. ``chunk_id``, ``text``, and ``metadata`` are preserved
        from the input (``metadata`` is copied so callers can mutate it freely).
        """
        ...


__all__ = ["RerankerPort"]
