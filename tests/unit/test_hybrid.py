"""Unit tests for HybridRetriever — RRF math, fallback, validation, wiring smoke.

Self-contained: spy retrievers stand in for the BM25 and dense legs so the
fusion logic is exercised in isolation from real models or vector stores.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any, ClassVar

import pytest

from uae_rag import config
from uae_rag.ingestion.chunker import Chunk
from uae_rag.ports import (
    QueryHit,
    RetrievalHit,
    RetrievalPort,
    VectorRecord,
)
from uae_rag.retrieval.hybrid import HybridRetriever


def _hit(chunk_id: str, rank: int, source: str = "bm25") -> RetrievalHit:
    """Build a stub hit. ``score`` is not used by RRF — rank is what matters."""
    return RetrievalHit(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"source_slug": "labour-law-en"},
        score=0.0,
        rank=rank,
        source=source,  # type: ignore[arg-type]
    )


class _SpyRetriever:
    """RetrievalPort fake; records the ``top_k`` it was called with."""

    def __init__(
        self,
        hits: list[RetrievalHit] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self._hits = hits or []
        self._raises = raises
        self.last_top_k: int | None = None

    def retrieve(self, query: str, *, top_k: int = 50) -> list[RetrievalHit]:
        self.last_top_k = top_k
        if self._raises is not None:
            raise self._raises
        return self._hits[:top_k]


def test_spy_satisfies_retrieval_port() -> None:
    assert isinstance(_SpyRetriever(), RetrievalPort)


def test_rrf_math_two_legs() -> None:
    """Overlapping rankings produce exact RRF scores; non-overlapping fall behind."""
    # BM25: a(1), b(2), c(3); Dense: b(1), a(2), d(3). rrf_k=60.
    # a: 1/61 + 1/62; b: 1/62 + 1/61; c: 1/63; d: 1/63.
    bm25 = _SpyRetriever([_hit("a", 1), _hit("b", 2), _hit("c", 3)])
    dense = _SpyRetriever([
        _hit("b", 1, "dense"),
        _hit("a", 2, "dense"),
        _hit("d", 3, "dense"),
    ])
    hybrid = HybridRetriever(bm25=bm25, dense=dense, per_leg_top_k=10, rrf_k=60)

    hits = hybrid.retrieve("q", top_k=4)

    expected_ab = 1.0 / 61 + 1.0 / 62
    expected_cd = 1.0 / 63
    assert [h.chunk_id for h in hits] == ["a", "b", "c", "d"]
    assert math.isclose(hits[0].score, expected_ab)
    assert math.isclose(hits[1].score, expected_ab)
    assert math.isclose(hits[2].score, expected_cd)
    assert math.isclose(hits[3].score, expected_cd)
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(h.source == "hybrid" for h in hits)


def test_tie_break_by_chunk_id() -> None:
    """When RRF scores tie, ascending chunk_id wins."""
    bm25 = _SpyRetriever([_hit("z", 1), _hit("a", 2)])
    dense = _SpyRetriever([_hit("a", 1, "dense"), _hit("z", 2, "dense")])
    hybrid = HybridRetriever(bm25=bm25, dense=dense, per_leg_top_k=10)

    hits = hybrid.retrieve("q", top_k=2)

    assert [h.chunk_id for h in hits] == ["a", "z"]


def test_empty_bm25_leg_equals_dense_ranking() -> None:
    """When BM25 returns nothing, fused output mirrors dense ordering."""
    bm25 = _SpyRetriever([])
    dense = _SpyRetriever([
        _hit("a", 1, "dense"),
        _hit("b", 2, "dense"),
        _hit("c", 3, "dense"),
    ])
    hybrid = HybridRetriever(bm25=bm25, dense=dense, per_leg_top_k=10)

    hits = hybrid.retrieve("q", top_k=3)

    assert [h.chunk_id for h in hits] == ["a", "b", "c"]
    assert all(h.source == "hybrid" for h in hits)


def test_both_legs_empty_returns_empty() -> None:
    hybrid = HybridRetriever(bm25=_SpyRetriever([]), dense=_SpyRetriever([]))

    assert hybrid.retrieve("q", top_k=5) == []


def test_dense_leg_exception_falls_back_to_bm25(caplog: pytest.LogCaptureFixture) -> None:
    """A raising dense leg is logged via logger.exception; BM25 still produces results."""
    bm25 = _SpyRetriever([_hit("a", 1), _hit("b", 2)])
    dense = _SpyRetriever(raises=RuntimeError("boom"))
    hybrid = HybridRetriever(bm25=bm25, dense=dense)

    with caplog.at_level(logging.ERROR, logger="uae_rag.retrieval.hybrid"):
        hits = hybrid.retrieve("q", top_k=5)

    assert [h.chunk_id for h in hits] == ["a", "b"]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "expected logger.exception to record an ERROR-level event"
    assert any("dense" in r.message.lower() for r in errors)


def test_per_leg_top_k_and_top_k_truncation() -> None:
    """Each leg is called with per_leg_top_k; the final list is capped at top_k."""
    bm25_hits = [_hit(f"b{i:02d}", i + 1) for i in range(50)]
    dense_hits = [_hit(f"d{i:02d}", i + 1, "dense") for i in range(50)]
    bm25 = _SpyRetriever(bm25_hits)
    dense = _SpyRetriever(dense_hits)
    hybrid = HybridRetriever(bm25=bm25, dense=dense, per_leg_top_k=50)

    hits = hybrid.retrieve("q", top_k=20)

    assert len(hits) == 20
    assert bm25.last_top_k == 50
    assert dense.last_top_k == 50


def test_top_k_must_be_positive() -> None:
    hybrid = HybridRetriever(bm25=_SpyRetriever(), dense=_SpyRetriever())

    with pytest.raises(ValueError, match="top_k"):
        hybrid.retrieve("q", top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        hybrid.retrieve("q", top_k=-1)


# --- Wiring smoke test for config.get_retriever ---------------------------------


def _unit(slot: int, dim: int = 4) -> list[float]:
    v = [0.0] * dim
    v[slot] = 1.0
    return v


class _FakeEmbeddings:
    """Tiny embedder: returns a unit vector keyed by the first token."""

    model_id: str = "fake"
    dimension: int = 4
    passage_prefix: str = ""
    query_prefix: str = ""

    _SLOTS: ClassVar[dict[str, int]] = {"annual": 0, "leave": 1, "article": 2}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vec(self, text: str) -> list[float]:
        words = text.lower().split()
        return _unit(self._SLOTS.get(words[0] if words else "annual", 0), self.dimension)


class _FakeVectorIndex:
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
        scored = [
            (sum(a * b for a, b in zip(r.embedding, embedding, strict=True)), r)
            for r in self._store.values()
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


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_slug="labour-law-en",
        breadcrumb=f"Article ({cid})",
        article_id=cid,
        language="en",
        page_start=1,
        page_end=1,
        text=text,
        mode="article",
    )


def test_get_retriever_wires_a_RetrievalPort() -> None:
    chunks = [
        _chunk("1", "annual leave entitlement is thirty days"),
        _chunk("2", "article 29 details annual leave policy"),
    ]
    embedder = _FakeEmbeddings()
    index = _FakeVectorIndex()
    index.upsert(
        VectorRecord(
            id=c.chunk_id,
            embedding=embedder.embed_query(c.text),
            document=c.text,
            metadata={"source_slug": c.source_slug},
        )
        for c in chunks
    )

    retriever = config.get_retriever(chunks=chunks, embedder=embedder, vector_index=index)

    assert isinstance(retriever, RetrievalPort)
    hits = retriever.retrieve("annual leave", top_k=5)
    assert hits, "wired retriever should surface at least one hit"
    assert all(h.source == "hybrid" for h in hits)
    assert {h.chunk_id for h in hits} <= {"1", "2"}
