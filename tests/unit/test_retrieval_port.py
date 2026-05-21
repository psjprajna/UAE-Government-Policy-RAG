"""Unit tests for ``RetrievalPort`` — Protocol surface + BM25/Dense conformance.

Self-contained: inline minimal fakes for embedder and vector index keep these
tests independent of ``test_embeddings_port`` / ``test_vector_index_port``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from typing import Any, ClassVar

import pytest

from uae_rag.ingestion.chunker import Chunk
from uae_rag.ports import (
    QueryHit,
    RetrievalHit,
    RetrievalPort,
    VectorIndexPort,
    VectorRecord,
)
from uae_rag.retrieval.bm25 import BM25Retriever
from uae_rag.retrieval.dense import DenseRetriever


def _unit(slot: int, dim: int = 8) -> list[float]:
    v = [0.0] * dim
    v[slot] = 1.0
    return v


class _FakeEmbeddings:
    """Tiny embedder: returns a unit vector keyed by the first token of the input."""

    model_id: str = "fake"
    dimension: int = 8
    passage_prefix: str = ""
    query_prefix: str = ""

    _SLOTS: ClassVar[dict[str, int]] = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vec(self, text: str) -> list[float]:
        first = text.split()[0].lower() if text.split() else "alpha"
        return _unit(self._SLOTS.get(first, 0), self.dimension)


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


def _chunk(cid: str, text: str, *, language: str = "en") -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_slug="labour-law-en",
        breadcrumb=f"Article ({cid})",
        article_id=cid,
        language=language,
        page_start=1,
        page_end=1,
        text=text,
        mode="article",
    )


def test_fake_vector_index_satisfies_port() -> None:
    """Sanity: the inline fake conforms to the vector index Protocol."""
    assert isinstance(_FakeVectorIndex(), VectorIndexPort)


def test_retrieval_hit_is_frozen_dataclass() -> None:
    hit = RetrievalHit(
        chunk_id="x",
        text="t",
        metadata={"source_slug": "labour-law-en"},
        score=0.5,
        rank=1,
        source="dense",
    )
    with pytest.raises(FrozenInstanceError):
        hit.score = 0.9  # type: ignore[misc]


def test_bm25_retriever_satisfies_retrieval_port() -> None:
    bm25 = BM25Retriever([_chunk("1", "annual leave entitlement")])
    assert isinstance(bm25, RetrievalPort)


def test_dense_retriever_satisfies_retrieval_port() -> None:
    dense = DenseRetriever(embedder=_FakeEmbeddings(), vector_index=_FakeVectorIndex())
    assert isinstance(dense, RetrievalPort)


def test_dense_retriever_returns_ranked_hits_with_source_and_rank() -> None:
    """Glue: embed_query → vector_index.query → RetrievalHit mapping is correct."""
    embedder = _FakeEmbeddings()
    index = _FakeVectorIndex()
    index.upsert(
        [
            VectorRecord(
                id="alpha-1",
                embedding=_unit(0),
                document="alpha doc",
                metadata={"source_slug": "labour-law-en"},
            ),
            VectorRecord(
                id="beta-1",
                embedding=_unit(1),
                document="beta doc",
                metadata={"source_slug": "labour-law-en"},
            ),
            VectorRecord(
                id="gamma-1",
                embedding=_unit(2),
                document="gamma doc",
                metadata={"source_slug": "labour-law-en"},
            ),
        ]
    )
    dense = DenseRetriever(embedder=embedder, vector_index=index)

    hits = dense.retrieve("beta query", top_k=3)

    assert len(hits) == 3
    assert hits[0].chunk_id == "beta-1"
    assert hits[0].rank == 1
    assert hits[1].rank == 2
    assert hits[2].rank == 3
    assert all(h.source == "dense" for h in hits)
    assert math.isclose(hits[0].score, 1.0, abs_tol=1e-9)
    # text is propagated from the index's document field
    assert hits[0].text == "beta doc"
    # metadata is propagated as a fresh dict
    assert hits[0].metadata == {"source_slug": "labour-law-en"}


def test_dense_retriever_respects_top_k() -> None:
    embedder = _FakeEmbeddings()
    index = _FakeVectorIndex()
    index.upsert(
        VectorRecord(
            id=f"r{i}",
            embedding=_unit(i % 4),
            document=f"doc {i}",
            metadata={"source_slug": "labour-law-en"},
        )
        for i in range(10)
    )
    dense = DenseRetriever(embedder=embedder, vector_index=index)

    hits = dense.retrieve("alpha", top_k=4)

    assert len(hits) == 4
