"""Unit tests for ``RerankRetriever`` — wrapper math, fallbacks, validation, wiring.

Self-contained: spy retriever stands in for the inner ``RetrievalPort``;
``FakeReranker`` is re-imported from ``test_reranker_port`` (matches the
Phase 4 cross-import precedent in ``test_hybrid.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from tests.unit.test_reranker_port import FakeReranker, _hit
from uae_rag import config
from uae_rag.ports import RerankerPort, RetrievalHit, RetrievalPort
from uae_rag.retrieval.rerank import RerankRetriever


class _SpyRetriever:
    """``RetrievalPort`` fake; records the ``top_k`` passed in by the wrapper."""

    def __init__(
        self,
        hits: list[RetrievalHit] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self._hits = hits or []
        self._raises = raises
        self.last_top_k: int | None = None
        self.last_query: str | None = None

    def retrieve(self, query: str, *, top_k: int = 20) -> list[RetrievalHit]:
        self.last_top_k = top_k
        self.last_query = query
        if self._raises is not None:
            raise self._raises
        return self._hits[:top_k]


class _RaisingReranker:
    """``RerankerPort`` fake that always raises — exercises the wrapper's fallback."""

    model_id: str = "raising-reranker"

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        raise RuntimeError("model unavailable")


# --- Protocol conformance --------------------------------------------------------


def test_rerank_retriever_satisfies_retrieval_port() -> None:
    """The wrapper itself implements ``RetrievalPort`` — callers stay agnostic."""
    wrapper = RerankRetriever(retriever=_SpyRetriever(), reranker=FakeReranker())

    assert isinstance(wrapper, RetrievalPort)


# --- Pass-through to inner retriever and reranker -------------------------------


def test_rerank_calls_inner_retriever_at_candidate_top_k() -> None:
    """Inner retriever is invoked at ``candidate_top_k``, not at the user's ``top_k``."""
    spy = _SpyRetriever([_hit(f"c{i}", "x" * (20 - i), rank=i + 1) for i in range(20)])
    wrapper = RerankRetriever(retriever=spy, reranker=FakeReranker(), candidate_top_k=15)

    wrapper.retrieve("q", top_k=5)

    assert spy.last_top_k == 15


def test_rerank_passes_query_to_reranker() -> None:
    """The same query string reaches both the inner retriever and the reranker."""
    spy = _SpyRetriever([_hit("a", "annual leave")])
    fake = FakeReranker()
    wrapper = RerankRetriever(retriever=spy, reranker=fake, candidate_top_k=20)

    wrapper.retrieve("annual leave entitlement", top_k=5)

    assert spy.last_query == "annual leave entitlement"
    assert fake.last_query == "annual leave entitlement"


def test_rerank_emits_source_reranked() -> None:
    """Every hit returned by the wrapper carries ``source="reranked"``."""
    spy = _SpyRetriever([_hit(f"c{i}", "x" * (10 - i)) for i in range(5)])
    wrapper = RerankRetriever(retriever=spy, reranker=FakeReranker(), candidate_top_k=20)

    hits = wrapper.retrieve("q", top_k=3)

    assert hits  # sanity
    assert all(h.source == "reranked" for h in hits)


# --- Behavior: re-scoring + truncation -------------------------------------------


def test_rerank_changes_order_per_reranker_scoring() -> None:
    """The FakeReranker re-orders by ``-len(text)``; the wrapper output reflects that.

    Inner retriever emits in RRF order (rank 1, 2, 3); the FakeReranker re-scores
    by passage length so the shortest text floats to rank 1 regardless of input rank.
    """
    inner_hits = [
        _hit("long", "x" * 200, rank=1),  # RRF rank 1 but longest text
        _hit("medium", "x" * 100, rank=2),
        _hit("short", "x" * 30, rank=3),  # RRF rank 3 but shortest text
    ]
    spy = _SpyRetriever(inner_hits)
    wrapper = RerankRetriever(retriever=spy, reranker=FakeReranker(), candidate_top_k=20)

    hits = wrapper.retrieve("q", top_k=3)

    assert [h.chunk_id for h in hits] == ["short", "medium", "long"]


def test_rerank_truncates_to_top_k() -> None:
    """Inner returns 20 candidates; reranker truncates to ``top_k``."""
    inner_hits = [_hit(f"c{i:02d}", "x" * (20 - i), rank=i + 1) for i in range(20)]
    spy = _SpyRetriever(inner_hits)
    wrapper = RerankRetriever(retriever=spy, reranker=FakeReranker(), candidate_top_k=20)

    hits = wrapper.retrieve("q", top_k=5)

    assert len(hits) == 5


# --- Edge cases + fallbacks ------------------------------------------------------


def test_empty_inner_result_returns_empty_without_calling_reranker() -> None:
    """When the inner retriever returns ``[]``, the reranker is never called."""
    fake = FakeReranker()
    wrapper = RerankRetriever(retriever=_SpyRetriever([]), reranker=fake, candidate_top_k=20)

    result = wrapper.retrieve("q", top_k=5)

    assert result == []
    assert fake.last_passages is None  # No call → no recording


def test_reranker_raises_falls_back_to_inner_hits(caplog: pytest.LogCaptureFixture) -> None:
    """A raising reranker → wrapper logs exception, returns inner hits truncated.

    Importantly, the fallback hits preserve ``source="hybrid"`` — downstream sees
    the true provenance, not a misleading ``"reranked"`` label.
    """
    inner_hits = [
        _hit("a", "text-a", rank=1, source="hybrid"),
        _hit("b", "text-b", rank=2, source="hybrid"),
        _hit("c", "text-c", rank=3, source="hybrid"),
    ]
    spy = _SpyRetriever(inner_hits)
    wrapper = RerankRetriever(retriever=spy, reranker=_RaisingReranker(), candidate_top_k=20)

    with caplog.at_level(logging.ERROR, logger="uae_rag.retrieval.rerank"):
        hits = wrapper.retrieve("q", top_k=2)

    assert [h.chunk_id for h in hits] == ["a", "b"]
    assert all(h.source == "hybrid" for h in hits)  # original provenance preserved
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "expected logger.exception to record an ERROR-level event"
    assert any("reranker" in r.message.lower() for r in errors)


# --- Input validation ------------------------------------------------------------


def test_top_k_zero_raises_value_error() -> None:
    wrapper = RerankRetriever(retriever=_SpyRetriever(), reranker=FakeReranker())

    with pytest.raises(ValueError, match="top_k"):
        wrapper.retrieve("q", top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        wrapper.retrieve("q", top_k=-1)


def test_candidate_top_k_less_than_top_k_logs_debug(caplog: pytest.LogCaptureFixture) -> None:
    """When ``candidate_top_k < top_k``, the wrapper logs at DEBUG and still runs.

    The reranked list is capped at ``candidate_top_k`` (the reranker can't invent
    new candidates); the DEBUG log surfaces the misconfiguration to operators.
    """
    spy = _SpyRetriever([_hit(f"c{i}", "x" * (5 - i), rank=i + 1) for i in range(5)])
    wrapper = RerankRetriever(retriever=spy, reranker=FakeReranker(), candidate_top_k=5)

    with caplog.at_level(logging.DEBUG, logger="uae_rag.retrieval.rerank"):
        hits = wrapper.retrieve("q", top_k=10)

    assert len(hits) == 5  # capped at candidate_top_k since reranker can't pad
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("candidate_top_k" in r.message for r in debugs)


# --- Config wiring smoke test ----------------------------------------------------


def test_get_reranker_returns_a_RerankerPort() -> None:
    """``config.get_reranker()`` wires the local adapter without loading the model."""
    reranker = config.get_reranker()

    assert isinstance(reranker, RerankerPort)
