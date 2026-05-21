"""Port: retrieval (sparse, dense, or hybrid).

The domain layer depends on this protocol; concrete retrievers under
``uae_rag.retrieval.*`` implement it. Per ADR-0002, no domain module imports
an adapter directly; per ADR-0004, the hybrid implementation fuses a BM25 leg
and a dense leg via Reciprocal Rank Fusion (k=60).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

RetrievalSource = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One ranked retrieval result.

    Result lists are sorted by descending ``score``. ``score`` units depend on
    ``source``: cosine similarity for ``dense``, raw BM25 score for ``bm25``,
    Reciprocal Rank Fusion score for ``hybrid``. Do not compare scores across
    sources — compare ranks instead.
    """

    chunk_id: str
    text: str
    metadata: dict[str, str | int | None]
    score: float
    rank: int  # 1-based rank in the producing leg/fusion.
    source: RetrievalSource


@runtime_checkable
class RetrievalPort(Protocol):
    """Interface for any retriever — sparse, dense, or hybrid."""

    def retrieve(self, query: str, *, top_k: int = 20) -> list[RetrievalHit]:
        """Return the top-``top_k`` hits for ``query``, ordered by descending score."""
        ...


__all__ = ["RetrievalHit", "RetrievalPort", "RetrievalSource"]
