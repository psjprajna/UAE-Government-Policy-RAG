"""Integration test for the rerank pipeline against the real local corpus.

Wraps the production hybrid retriever (BM25 + dense + RRF) with the real
``SentenceTransformersReranker`` (BGE-v2-m3) and runs the three canonical
queries from Phase 4 to confirm the reranked top-K still surfaces the gold
chunks. Marked ``adapter_local`` and ``slow`` so the default suite skips it;
further guarded by file-presence and ``sentence_transformers`` import checks
so a missing corpus or model just skips cleanly.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from uae_rag import config
from uae_rag.ingestion.chunker import chunk_articles
from uae_rag.ingestion.parser import parse
from uae_rag.ingestion.registry import SOURCES
from uae_rag.retrieval.rerank import RerankRetriever

if TYPE_CHECKING:
    from uae_rag.ingestion.chunker import Chunk

pytestmark = [pytest.mark.adapter_local, pytest.mark.slow]

_APP_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _APP_ROOT / "data"
_CHROMA_DIR = _DATA_DIR / "chroma_db"
_RAW_DIR = _DATA_DIR / "raw"
_MAX_TOKENS = 500  # matches scripts/build_index.py


def _skip_unless_corpus_and_model_available() -> None:
    if not _CHROMA_DIR.exists():
        pytest.skip(f"corpus index missing: {_CHROMA_DIR}")
    if not _RAW_DIR.exists():
        pytest.skip(f"raw PDFs missing: {_RAW_DIR}")
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed")
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence_transformers not installed")


def _reparse_chunks(count_tokens: Callable[[str], int]) -> list[Chunk]:
    """Rebuild the in-memory chunk corpus using the same pipeline as build_index.py."""
    chunks: list[Chunk] = []
    for src in SOURCES:
        pdf = _RAW_DIR / src.local_filename
        if not pdf.exists():
            continue
        articles = parse(pdf, src)
        chunks.extend(
            chunk_articles(
                articles,
                source_slug=src.slug,
                count_tokens=count_tokens,
                max_tokens=_MAX_TOKENS,
            )
        )
    return chunks


@pytest.fixture(scope="module")
def pipeline() -> RerankRetriever:
    """Build the full hybrid → rerank pipeline against the real corpus + model.

    The fixture is module-scoped so the ~6s cross-encoder cold-start is paid
    once. A trivial warmup call forces the lazy model load up front so the
    first real test sees steady-state latency.
    """
    _skip_unless_corpus_and_model_available()
    embedder = config.get_embeddings()
    index = config.get_vector_index(persist_dir=_CHROMA_DIR, embedder=embedder)
    chunks = _reparse_chunks(embedder.count_tokens)
    if not chunks:
        pytest.skip("re-parse produced no chunks; raw PDFs may be incomplete")
    hybrid = config.get_retriever(chunks=chunks, embedder=embedder, vector_index=index)
    reranker = config.get_reranker()
    wrapper = RerankRetriever(retriever=hybrid, reranker=reranker, candidate_top_k=20)
    # Warmup — pays the cross-encoder model load now rather than on the first assertion.
    wrapper.retrieve("warmup", top_k=1)
    return wrapper


def test_rerank_pipeline_returns_article_29_for_literal_en_query(
    pipeline: RerankRetriever,
) -> None:
    """Lexical 'Article 29' still surfaces an Article 29 EN chunk in the reranked top-3."""
    hits = pipeline.retrieve("Article 29", top_k=3)
    chunk_ids = [h.chunk_id for h in hits]
    assert any(
        cid.startswith("labour-law-en::art-29") for cid in chunk_ids
    ), f"no Article 29 EN chunk in reranked top-3: {chunk_ids}"


def test_rerank_pipeline_returns_article_29_for_semantic_en_query(
    pipeline: RerankRetriever,
) -> None:
    """Semantic 'annual leave entitlement' lands the Article 29 chunk in the reranked top-5."""
    hits = pipeline.retrieve("annual leave entitlement", top_k=5)
    chunk_ids = [h.chunk_id for h in hits]
    assert any(
        cid.startswith("labour-law-en::art-29") for cid in chunk_ids
    ), f"no Article 29 EN chunk in reranked top-5: {chunk_ids}"


def test_rerank_pipeline_returns_article_29_for_arabic_query(
    pipeline: RerankRetriever,
) -> None:
    """Arabic 'المادة 29' surfaces the AR Article 29 chunk in the reranked top-5."""
    hits = pipeline.retrieve("المادة 29", top_k=5)
    chunk_ids = [h.chunk_id for h in hits]
    assert any(
        cid.startswith("labour-law-ar::art-29") for cid in chunk_ids
    ), f"no Article 29 AR chunk in reranked top-5: {chunk_ids}"


def test_rerank_hits_carry_source_reranked(pipeline: RerankRetriever) -> None:
    """Every hit produced by the wrapper is labeled ``source="reranked"``."""
    hits = pipeline.retrieve("annual leave entitlement", top_k=5)
    assert hits, "expected at least one hit for a known-good query"
    assert all(h.source == "reranked" for h in hits)


def test_rerank_scores_monotonic_desc(pipeline: RerankRetriever) -> None:
    """Returned hits are sorted by descending cross-encoder score."""
    hits = pipeline.retrieve("annual leave entitlement", top_k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), f"scores not monotonic desc: {scores}"
