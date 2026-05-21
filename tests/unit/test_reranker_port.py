"""Unit tests for ``RerankerPort`` — Protocol surface + FakeReranker + adapter.

Self-contained. ``FakeReranker`` is a deterministic no-model reranker that
scores by ``-len(text)`` (shorter passages rank higher) and records the last
query / passages it saw. It mirrors the role of ``_SpyRetriever`` in
``test_hybrid.py`` and is re-imported by ``test_rerank.py`` in Slice 2.

The slow + ``adapter_local`` subset at the bottom loads the real BGE-v2-m3
CrossEncoder and asserts a relevant passage outscores an irrelevant one — the
realistic-passage smoke captured in the Phase 5 pre-flight check log.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from uae_rag.adapters.local.reranker import (
    _DEFAULT_MAX_LENGTH,
    _DEFAULT_MODEL,
    _DEFAULT_REVISION,
    SentenceTransformersReranker,
)
from uae_rag.ports import RerankerPort, RetrievalHit


def _hit(
    chunk_id: str,
    text: str,
    *,
    score: float = 0.0,
    rank: int = 1,
    source: str = "hybrid",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_slug": "labour-law-en"},
        score=score,
        rank=rank,
        source=source,  # type: ignore[arg-type]
    )


class FakeReranker:
    """Deterministic ``RerankerPort`` fake — scores by ``-len(text)``.

    Records ``last_query`` and ``last_passages`` so callers can assert the
    wrapper actually invoked the reranker (or didn't, in the empty-input case).
    """

    model_id: str = "fake-reranker"

    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_passages: list[str] | None = None

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be ≥ 1")
        if not hits:
            return []
        self.last_query = query
        self.last_passages = [h.text for h in hits]
        ordered = sorted(hits, key=lambda h: (len(h.text), h.chunk_id))
        return [
            RetrievalHit(
                chunk_id=h.chunk_id,
                text=h.text,
                metadata=dict(h.metadata),
                score=-float(len(h.text)),
                rank=i + 1,
                source="reranked",
            )
            for i, h in enumerate(ordered[:top_k])
        ]


# --- Protocol conformance --------------------------------------------------------


def test_fake_reranker_satisfies_port() -> None:
    """Structural fake conforms to the runtime-checkable Protocol."""
    assert isinstance(FakeReranker(), RerankerPort)


def test_sentence_transformers_reranker_satisfies_port() -> None:
    """Real adapter is structurally compliant without loading any model."""
    assert isinstance(SentenceTransformersReranker(), RerankerPort)


# --- Env-var propagation (no model load) -----------------------------------------


def test_env_overrides_propagate_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RERANKER_MODEL", "custom-model")
    monkeypatch.setenv("LOCAL_RERANKER_REVISION", "abcdef1234567890")
    monkeypatch.setenv("LOCAL_RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("LOCAL_RERANKER_MAX_LENGTH", "512")

    adapter = SentenceTransformersReranker()

    assert adapter.model_id == "custom-model"
    assert adapter.revision == "abcdef1234567890"
    assert adapter.device == "cpu"
    assert adapter.max_length == 512


def test_default_env_vars_match_adr_0007(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env → defaults reflect ADR-0007 (BGE-v2-m3 + pinned revision)."""
    for var in (
        "LOCAL_RERANKER_MODEL",
        "LOCAL_RERANKER_REVISION",
        "LOCAL_RERANKER_DEVICE",
        "LOCAL_RERANKER_MAX_LENGTH",
    ):
        monkeypatch.delenv(var, raising=False)

    adapter = SentenceTransformersReranker()

    assert adapter.model_id == _DEFAULT_MODEL == "BAAI/bge-reranker-v2-m3"
    assert adapter.revision == _DEFAULT_REVISION
    assert adapter.device is None  # auto-detect when env var is unset
    assert adapter.max_length == _DEFAULT_MAX_LENGTH == 8192


def test_empty_device_env_falls_back_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``LOCAL_RERANKER_DEVICE`` defers to CrossEncoder's auto-detect."""
    monkeypatch.setenv("LOCAL_RERANKER_DEVICE", "")

    adapter = SentenceTransformersReranker()

    assert adapter.device is None


# --- FakeReranker behavior (the reranker contract callers depend on) -------------


def test_fake_rerank_orders_by_score_desc() -> None:
    """FakeReranker re-orders by descending score: shorter text → higher score."""
    hits = [
        _hit("long", "x" * 200, rank=1),
        _hit("short", "x" * 50, rank=2),
        _hit("medium", "x" * 100, rank=3),
    ]

    ranked = FakeReranker().rerank("q", hits, top_k=3)

    assert [h.chunk_id for h in ranked] == ["short", "medium", "long"]
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_fake_rerank_emits_source_reranked_and_one_based_rank() -> None:
    """All returned hits carry ``source="reranked"`` and 1-based rank."""
    hits = [_hit(f"c{i}", "x" * (10 - i), rank=10 - i) for i in range(3)]

    ranked = FakeReranker().rerank("q", hits, top_k=3)

    assert all(h.source == "reranked" for h in ranked)
    assert [h.rank for h in ranked] == [1, 2, 3]


def test_fake_rerank_preserves_chunk_id_text_metadata() -> None:
    """``chunk_id``, ``text``, and ``metadata`` survive the rerank; metadata is copied."""
    metadata = {"source_slug": "labour-law-en", "article_id": "29"}
    hit = RetrievalHit(
        chunk_id="art-29",
        text="annual leave entitlement",
        metadata=metadata,
        score=0.0,
        rank=1,
        source="hybrid",
    )

    ranked = FakeReranker().rerank("q", [hit], top_k=1)

    assert ranked[0].chunk_id == "art-29"
    assert ranked[0].text == "annual leave entitlement"
    assert ranked[0].metadata == metadata
    assert ranked[0].metadata is not metadata


def test_fake_rerank_records_query_and_passages() -> None:
    """Fake reranker tracks invocation for downstream wrapper tests."""
    fake = FakeReranker()
    hits = [_hit("a", "annual leave"), _hit("b", "probation")]

    fake.rerank("article 29", hits, top_k=2)

    assert fake.last_query == "article 29"
    assert fake.last_passages == ["annual leave", "probation"]


def test_top_k_zero_or_negative_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        FakeReranker().rerank("q", [_hit("a", "x")], top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        FakeReranker().rerank("q", [_hit("a", "x")], top_k=-1)


def test_empty_hits_returns_empty_without_recording_call() -> None:
    """Empty input → ``[]`` and the reranker is treated as not having been invoked."""
    fake = FakeReranker()

    result = fake.rerank("q", [], top_k=5)

    assert result == []
    assert fake.last_passages is None


def test_top_k_larger_than_hits_returns_all_reranked() -> None:
    """When ``top_k`` exceeds the input size, all hits come back in re-scored order."""
    hits = [_hit("a", "xxx"), _hit("b", "x"), _hit("c", "xx")]

    ranked = FakeReranker().rerank("q", hits, top_k=10)

    assert len(ranked) == 3
    assert [h.chunk_id for h in ranked] == ["b", "c", "a"]


# --- Adapter-level validation (no model load) ------------------------------------


def test_adapter_top_k_zero_raises_before_model_load() -> None:
    """The real adapter rejects bad top_k before touching the CrossEncoder."""
    adapter = SentenceTransformersReranker()

    with pytest.raises(ValueError, match="top_k"):
        adapter.rerank("q", [_hit("a", "x")], top_k=0)


def test_adapter_empty_hits_skips_model_load() -> None:
    """Empty hits short-circuit before any CrossEncoder import or load."""
    adapter = SentenceTransformersReranker()

    result = adapter.rerank("q", [], top_k=5)

    assert result == []
    # ``_model`` is a cached_property — if it never executed, the cache slot is empty.
    assert "_model" not in adapter.__dict__


# --- Slow / adapter_local: real BGE-v2-m3 ranking sanity -------------------------


@pytest.mark.adapter_local
@pytest.mark.slow
def test_real_model_scores_relevant_passage_higher() -> None:
    """Real BGE-v2-m3 ranks the Article-29 passage above the Article-35 passage.

    Mirrors the realistic-passage smoke captured in
    ``.claude/specs/005-phase-5-reranker/tasks.md`` (delta ~+0.95).
    """
    pytest.importorskip("sentence_transformers")

    adapter = SentenceTransformersReranker()
    hits = [
        _hit(
            "art-29",
            "Article 29: Every worker shall be entitled to annual leave for each "
            "year of service of not less than thirty days",
        ),
        _hit(
            "art-35",
            "Article 35: The probationary period shall not exceed six months from "
            "the date of commencement of work",
        ),
    ]

    ranked = adapter.rerank("annual leave entitlement", hits, top_k=2)

    assert [h.chunk_id for h in ranked] == ["art-29", "art-35"]
    assert ranked[0].score > ranked[1].score
    assert all(h.source == "reranked" for h in ranked)
    assert [h.rank for h in ranked] == [1, 2]
