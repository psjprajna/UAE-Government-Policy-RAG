"""Unit tests for ``VectorIndexPort`` — Protocol surface and semantic contract.

A dict-backed ``FakeVectorIndex`` exercises upsert/count/query/reset and the
``where`` metadata-equality filter that ChromaDB provides, with no third-party
dependency. Orthogonal-basis vectors give deterministic top-k ordering.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import pytest

from uae_rag.ports import QueryHit, VectorIndexPort, VectorRecord


class FakeVectorIndex:
    """In-memory ``VectorIndexPort`` — dict-backed, brute-force cosine query."""

    def __init__(self) -> None:
        self._store: dict[str, VectorRecord] = {}

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        for r in records:
            self._store[r.id] = r

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[QueryHit]:
        candidates = [r for r in self._store.values() if _matches(r.metadata, where)]
        scored = [
            (sum(a * b for a, b in zip(r.embedding, embedding, strict=True)), r) for r in candidates
        ]
        scored.sort(key=lambda t: -t[0])
        return [
            QueryHit(id=r.id, document=r.document, metadata=dict(r.metadata), score=s)
            for s, r in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._store)

    def reset(self) -> None:
        self._store.clear()


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    return all(metadata.get(k) == v for k, v in where.items())


def _unit(slot: int, dim: int = 8) -> list[float]:
    """Unit vector with 1.0 in ``slot`` — orthogonal to any other slot."""
    v = [0.0] * dim
    v[slot] = 1.0
    return v


def _record(
    rid: str,
    slot: int,
    *,
    source_slug: str = "labour-law-en",
    language: str = "en",
) -> VectorRecord:
    return VectorRecord(
        id=rid,
        embedding=_unit(slot),
        document=f"doc for {rid}",
        metadata={"source_slug": source_slug, "language": language},
    )


@pytest.fixture
def index() -> FakeVectorIndex:
    return FakeVectorIndex()


def test_fake_vector_index_satisfies_port(index: FakeVectorIndex) -> None:
    assert isinstance(index, VectorIndexPort)


def test_upsert_then_count_matches_record_count(index: FakeVectorIndex) -> None:
    index.upsert(_record(f"r{i}", slot=i) for i in range(5))
    assert index.count() == 5


def test_upsert_replaces_existing_record_by_id(index: FakeVectorIndex) -> None:
    """A second upsert with the same id overwrites; count is unchanged."""
    index.upsert([_record("r0", slot=0)])
    index.upsert([_record("r0", slot=3)])  # same id, different slot

    assert index.count() == 1
    hits = index.query(_unit(3), top_k=1)
    assert hits[0].id == "r0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-9)


def test_query_orders_results_by_cosine_similarity(index: FakeVectorIndex) -> None:
    index.upsert(_record(f"r{i}", slot=i) for i in range(5))

    hits = index.query(_unit(2), top_k=3)

    assert len(hits) == 3
    assert hits[0].id == "r2"
    assert hits[0].score == pytest.approx(1.0, abs=1e-9)
    # Non-matching slots return cosine == 0; ordering among ties is implementation-defined,
    # but the top hit is guaranteed and the score sequence is monotonic non-increasing.
    for prev, nxt in pairwise(hits):
        assert prev.score >= nxt.score


def test_where_filter_on_source_slug_narrows_candidates(index: FakeVectorIndex) -> None:
    index.upsert(
        [
            _record("en-1", slot=0, source_slug="labour-law-en"),
            _record("mo-1", slot=0, source_slug="mohre-resolutions"),
            _record("en-2", slot=0, source_slug="labour-law-en"),
        ]
    )

    hits = index.query(_unit(0), top_k=10, where={"source_slug": "mohre-resolutions"})

    assert {h.id for h in hits} == {"mo-1"}


def test_where_filter_on_language_narrows_candidates(index: FakeVectorIndex) -> None:
    index.upsert(
        [
            _record("en-1", slot=0, language="en"),
            _record("ar-1", slot=0, language="ar"),
            _record("ar-2", slot=0, language="ar"),
        ]
    )

    hits = index.query(_unit(0), top_k=10, where={"language": "ar"})

    assert {h.id for h in hits} == {"ar-1", "ar-2"}


def test_reset_clears_state(index: FakeVectorIndex) -> None:
    index.upsert(_record(f"r{i}", slot=i) for i in range(3))
    assert index.count() == 3

    index.reset()

    assert index.count() == 0
    assert index.query(_unit(0), top_k=5) == []
