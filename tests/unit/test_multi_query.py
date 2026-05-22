"""Unit tests for ``MultiQueryRetriever`` — variation parsing, RRF fusion, fallback.

Self-contained: a small ``_FakeRetriever`` stands in for the inner
``RetrievalPort``; ``FakeLLM`` is re-imported from ``test_llm_port`` (same
cross-test pattern Phase 4's ``test_hybrid`` and Phase 5's ``test_rerank``
already use). Zero network, zero daemon.
"""

from __future__ import annotations

import logging

import pytest

from tests.unit.test_llm_port import FakeLLM
from uae_rag.ports import RetrievalHit, RetrievalPort
from uae_rag.retrieval.multi_query import MultiQueryRetriever


def _hit(
    chunk_id: str,
    *,
    text: str = "text",
    rank: int = 1,
    source: str = "hybrid",
    score: float = 0.5,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_slug": "x"},
        score=score,
        rank=rank,
        source=source,  # type: ignore[arg-type]
    )


class _FakeRetriever:
    """``RetrievalPort`` fake with per-query canned hit lists."""

    def __init__(
        self,
        hits_by_query: dict[str, list[RetrievalHit]] | None = None,
        *,
        default: list[RetrievalHit] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._by = hits_by_query or {}
        self._default = default or []
        self._raises = raises
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int = 20) -> list[RetrievalHit]:
        self.calls.append((query, top_k))
        if self._raises is not None:
            raise self._raises
        return list(self._by.get(query, self._default))[:top_k]


_VAR_RESPONSE_TWO = "1. variation one\n2. variation two"


# --- Protocol conformance --------------------------------------------------------


def test_multi_query_satisfies_retrieval_port() -> None:
    """The wrapper itself implements ``RetrievalPort`` — callers stay agnostic."""
    wrapper = MultiQueryRetriever(
        retriever=_FakeRetriever(),
        llm=FakeLLM(canned_response=""),
    )

    assert isinstance(wrapper, RetrievalPort)


# --- Input validation ------------------------------------------------------------


def test_n_variations_zero_raises_in_init() -> None:
    with pytest.raises(ValueError, match="n_variations"):
        MultiQueryRetriever(
            retriever=_FakeRetriever(),
            llm=FakeLLM(canned_response=""),
            n_variations=0,
        )


def test_n_variations_negative_raises_in_init() -> None:
    with pytest.raises(ValueError, match="n_variations"):
        MultiQueryRetriever(
            retriever=_FakeRetriever(),
            llm=FakeLLM(canned_response=""),
            n_variations=-1,
        )


def test_top_k_zero_raises_value_error() -> None:
    wrapper = MultiQueryRetriever(
        retriever=_FakeRetriever(),
        llm=FakeLLM(canned_response=""),
    )

    with pytest.raises(ValueError, match="top_k"):
        wrapper.retrieve("q", top_k=0)


def test_top_k_negative_raises_value_error() -> None:
    wrapper = MultiQueryRetriever(
        retriever=_FakeRetriever(),
        llm=FakeLLM(canned_response=""),
    )

    with pytest.raises(ValueError, match="top_k"):
        wrapper.retrieve("q", top_k=-3)


# --- Empty-query short-circuit ---------------------------------------------------


def test_empty_query_propagates_to_inner_without_calling_llm() -> None:
    """Whitespace-only / empty query → inner retriever called once, LLM untouched."""
    fake_llm = FakeLLM(canned_response="should-not-be-used")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm)

    out = wrapper.retrieve("   ", top_k=5)

    assert [h.chunk_id for h in out] == ["A"]
    assert fake_llm.call_count == 0
    assert len(inner.calls) == 1
    assert inner.calls[0][0] == "   "


# --- LLM-driven variations + sequencing -----------------------------------------


def test_inner_retriever_is_called_once_per_unique_query() -> None:
    """Original + 2 variations → 3 inner calls; identical strings get deduped."""
    fake_llm = FakeLLM(canned_response=_VAR_RESPONSE_TWO)
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=2)

    wrapper.retrieve("original q", top_k=5)

    queries = [call[0] for call in inner.calls]
    assert queries == ["original q", "variation one", "variation two"]


def test_inner_top_k_matches_caller_top_k() -> None:
    """Each inner call uses the caller's ``top_k`` — fusion handles the trim."""
    fake_llm = FakeLLM(canned_response=_VAR_RESPONSE_TWO)
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=2)

    wrapper.retrieve("q", top_k=12)

    assert {call[1] for call in inner.calls} == {12}


def test_duplicate_variation_lines_are_deduped() -> None:
    """LLM emits a duplicate phrasing → wrapper calls the inner retriever once for it."""
    fake_llm = FakeLLM(canned_response="1. same line\n2. same line\n3. unique line")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=3)

    wrapper.retrieve("q", top_k=5)

    queries = [call[0] for call in inner.calls]
    assert queries == ["q", "same line", "unique line"]


def test_variation_equal_to_original_is_deduped() -> None:
    """If the LLM echoes the original query, the inner retriever is not called twice."""
    fake_llm = FakeLLM(canned_response="1. q\n2. rephrased q")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=2)

    wrapper.retrieve("q", top_k=5)

    queries = [call[0] for call in inner.calls]
    assert queries == ["q", "rephrased q"]


def test_blank_lines_in_llm_output_are_ignored() -> None:
    fake_llm = FakeLLM(canned_response="\n1. first\n\n2. second\n\n")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=2)

    wrapper.retrieve("q", top_k=5)

    queries = [call[0] for call in inner.calls]
    assert queries == ["q", "first", "second"]


def test_fewer_variations_than_requested_is_tolerated() -> None:
    """LLM returns 1 variation when 3 requested → wrapper uses what it got."""
    fake_llm = FakeLLM(canned_response="1. only one")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=3)

    wrapper.retrieve("q", top_k=5)

    queries = [call[0] for call in inner.calls]
    assert queries == ["q", "only one"]


def test_variation_caps_at_n_variations() -> None:
    """LLM over-produces → wrapper trims to ``n_variations``."""
    fake_llm = FakeLLM(canned_response="1. a\n2. b\n3. c\n4. d\n5. e")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=2)

    wrapper.retrieve("q", top_k=5)

    queries = [call[0] for call in inner.calls]
    assert queries == ["q", "a", "b"]


def test_variation_prefix_formats_are_stripped() -> None:
    """Numeric/punct prefixes (``1.``, ``2)``, ``3 -``, ``4``, bullet ``-``) are stripped."""
    fake_llm = FakeLLM(canned_response="1. one\n2) two\n3 - three\n4 four\n- five")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=5)

    wrapper.retrieve("q", top_k=5)

    queries = [call[0] for call in inner.calls]
    assert queries == ["q", "one", "two", "three", "four", "five"]


def test_variation_prompt_is_sent_to_llm() -> None:
    """The LLM receives a prompt containing the original query and the requested count."""
    fake_llm = FakeLLM(canned_response="1. v1\n2. v2\n3. v3")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=3)

    wrapper.retrieve("annual leave entitlement", top_k=5)

    assert fake_llm.last_prompt is not None
    assert "annual leave entitlement" in fake_llm.last_prompt
    assert "3" in fake_llm.last_prompt


def test_variation_prompt_uses_max_variation_tokens() -> None:
    """``max_variation_tokens`` is forwarded to the LLM as ``max_output_tokens``."""
    fake_llm = FakeLLM(canned_response="1. v1\n2. v2")
    inner = _FakeRetriever(default=[_hit("A", rank=1)])
    wrapper = MultiQueryRetriever(
        retriever=inner,
        llm=fake_llm,
        n_variations=2,
        max_variation_tokens=128,
    )

    wrapper.retrieve("q", top_k=5)

    assert fake_llm.last_max_output_tokens == 128


# --- RRF fusion math -------------------------------------------------------------


def test_rrf_fusion_math_against_hand_rolled_fixture() -> None:
    """Three legs with overlapping hits — fused order and scores match RRF arithmetic.

    Original  → [A(1), B(2), C(3)]
    Variation → [B(1), C(2)]
    Variation → [C(1)]

    With ``rrf_k=60``:
      A = 1/61
      B = 1/62 + 1/61
      C = 1/63 + 1/62 + 1/61
    Order: C > B > A.
    """
    fake_llm = FakeLLM(canned_response=_VAR_RESPONSE_TWO)
    inner = _FakeRetriever(
        hits_by_query={
            "q": [
                _hit("A", rank=1, source="hybrid"),
                _hit("B", rank=2, source="hybrid"),
                _hit("C", rank=3, source="hybrid"),
            ],
            "variation one": [
                _hit("B", rank=1, source="hybrid"),
                _hit("C", rank=2, source="hybrid"),
            ],
            "variation two": [_hit("C", rank=1, source="hybrid")],
        }
    )
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=2, rrf_k=60)

    hits = wrapper.retrieve("q", top_k=3)

    assert [h.chunk_id for h in hits] == ["C", "B", "A"]
    assert hits[0].score == pytest.approx(1 / 63 + 1 / 62 + 1 / 61)
    assert hits[1].score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[2].score == pytest.approx(1 / 61)
    assert [h.rank for h in hits] == [1, 2, 3]


def test_rrf_fusion_preserves_inner_source() -> None:
    """Fused hits carry the inner retriever's ``source`` (typically ``"hybrid"``)."""
    fake_llm = FakeLLM(canned_response=_VAR_RESPONSE_TWO)
    inner = _FakeRetriever(
        hits_by_query={
            "q": [_hit("A", rank=1, source="reranked")],
            "variation one": [_hit("A", rank=1, source="reranked")],
            "variation two": [_hit("B", rank=1, source="reranked")],
        }
    )
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=2)

    hits = wrapper.retrieve("q", top_k=3)

    assert all(h.source == "reranked" for h in hits)


def test_rrf_tie_break_on_chunk_id_ascending() -> None:
    """When two chunks accrue equal RRF mass, the lexicographically smaller id wins."""
    fake_llm = FakeLLM(canned_response="1. var-a")
    inner = _FakeRetriever(
        hits_by_query={
            "q": [_hit("zzz", rank=1), _hit("aaa", rank=2)],
            "var-a": [_hit("aaa", rank=1), _hit("zzz", rank=2)],
        }
    )
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=1, rrf_k=60)

    hits = wrapper.retrieve("q", top_k=2)

    # Both chunks accrue exactly 1/61 + 1/62 — tie → aaa wins on lexicographic order.
    assert [h.chunk_id for h in hits] == ["aaa", "zzz"]


def test_rrf_truncates_to_top_k() -> None:
    """Fused list is capped at the caller's ``top_k``."""
    fake_llm = FakeLLM(canned_response="1. v1")
    inner = _FakeRetriever(
        hits_by_query={
            "q": [_hit(f"c{i:02d}", rank=i + 1) for i in range(10)],
            "v1": [_hit(f"c{i:02d}", rank=i + 1) for i in range(10)],
        }
    )
    wrapper = MultiQueryRetriever(retriever=inner, llm=fake_llm, n_variations=1)

    hits = wrapper.retrieve("q", top_k=4)

    assert len(hits) == 4


# --- Fallback when the LLM raises ------------------------------------------------


def test_llm_raises_falls_back_to_single_inner_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLM transport failure → wrapper logs at ERROR and runs the inner retriever once."""

    def boom(_prompt: str) -> str:
        raise RuntimeError("LLM down")

    inner = _FakeRetriever(default=[_hit("A", rank=1), _hit("B", rank=2)])
    wrapper = MultiQueryRetriever(retriever=inner, llm=FakeLLM(callable_=boom))

    with caplog.at_level(logging.ERROR, logger="uae_rag.retrieval.multi_query"):
        hits = wrapper.retrieve("q", top_k=2)

    assert [h.chunk_id for h in hits] == ["A", "B"]
    assert len(inner.calls) == 1
    assert inner.calls[0] == ("q", 2)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "expected logger.exception to record an ERROR-level event"
    assert any("variation" in r.message.lower() for r in errors)


def test_llm_fallback_preserves_inner_source_and_ranks() -> None:
    """In the fallback path the wrapper returns the inner hits untouched."""

    def boom(_prompt: str) -> str:
        raise RuntimeError("LLM down")

    raw = [_hit("A", rank=1, source="hybrid"), _hit("B", rank=2, source="hybrid")]
    wrapper = MultiQueryRetriever(
        retriever=_FakeRetriever(default=raw),
        llm=FakeLLM(callable_=boom),
    )

    hits = wrapper.retrieve("q", top_k=2)

    assert [h.rank for h in hits] == [1, 2]
    assert all(h.source == "hybrid" for h in hits)


# --- Empty inner result ----------------------------------------------------------


def test_empty_inner_returns_empty() -> None:
    """When every leg comes back empty, the fused list is empty."""
    fake_llm = FakeLLM(canned_response=_VAR_RESPONSE_TWO)
    wrapper = MultiQueryRetriever(
        retriever=_FakeRetriever(default=[]),
        llm=fake_llm,
        n_variations=2,
    )

    assert wrapper.retrieve("q", top_k=5) == []


def test_inner_retriever_failure_propagates() -> None:
    """An exception from the inner retriever is not caught — caller decides."""
    fake_llm = FakeLLM(canned_response=_VAR_RESPONSE_TWO)
    wrapper = MultiQueryRetriever(
        retriever=_FakeRetriever(raises=RuntimeError("inner down")),
        llm=fake_llm,
    )

    with pytest.raises(RuntimeError, match="inner down"):
        wrapper.retrieve("q", top_k=5)
