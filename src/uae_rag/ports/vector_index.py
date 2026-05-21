"""Port: vector index (upsert + similarity query).

The domain layer depends on this protocol; adapters under
``uae_rag.adapters.*`` implement it. Per ADR-0002, no domain module imports
an adapter directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One row to upsert into the index."""

    id: str
    embedding: list[float]
    document: str
    metadata: dict[str, str | int | None]


@dataclass(frozen=True, slots=True)
class QueryHit:
    """One row returned by a similarity query, ordered by descending ``score``."""

    id: str
    document: str
    metadata: dict[str, Any]
    score: float  # cosine similarity in [0, 1]; higher is closer.


@runtime_checkable
class VectorIndexPort(Protocol):
    """Interface for a persistent or in-memory vector index."""

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        """Insert or replace records by id. Deterministic on identical input."""
        ...

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[QueryHit]:
        """Return the top-``top_k`` hits, optionally filtered by metadata equality."""
        ...

    def count(self) -> int:
        """Return the number of records currently stored."""
        ...

    def reset(self) -> None:
        """Drop every record; subsequent ``count()`` returns 0."""
        ...


__all__ = ["QueryHit", "VectorIndexPort", "VectorRecord"]
