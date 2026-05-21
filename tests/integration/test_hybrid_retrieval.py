"""Integration test for the hybrid retriever against the real local corpus.

Loads the persistent ChromaDB collection at ``data/chroma_db/`` and re-derives
the BM25 corpus from ``data/raw/`` PDFs via the same parser/chunker the build
script uses. Marked ``adapter_local`` so the default suite skips it; further
guarded by file-presence and ``chromadb`` import checks so a missing corpus
just skips cleanly.
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

if TYPE_CHECKING:
    from uae_rag.ingestion.chunker import Chunk
    from uae_rag.ports import RetrievalPort

pytestmark = pytest.mark.adapter_local

_APP_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _APP_ROOT / "data"
_CHROMA_DIR = _DATA_DIR / "chroma_db"
_RAW_DIR = _DATA_DIR / "raw"
_MAX_TOKENS = 500  # matches scripts/build_index.py (e5-large budget minus buffer)


def _skip_unless_corpus_present() -> None:
    if not _CHROMA_DIR.exists():
        pytest.skip(f"corpus index missing: {_CHROMA_DIR}")
    if not _RAW_DIR.exists():
        pytest.skip(f"raw PDFs missing: {_RAW_DIR}")
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed")


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
def retriever() -> RetrievalPort:
    _skip_unless_corpus_present()
    embedder = config.get_embeddings()
    index = config.get_vector_index(persist_dir=_CHROMA_DIR, embedder=embedder)
    chunks = _reparse_chunks(embedder.count_tokens)
    if not chunks:
        pytest.skip("re-parse produced no chunks; raw PDFs may be incomplete")
    return config.get_retriever(chunks=chunks, embedder=embedder, vector_index=index)


def test_hybrid_returns_article_29_for_literal_query(retriever: RetrievalPort) -> None:
    """Lexical 'Article 29' surfaces an Article 29 EN chunk in the top-3."""
    hits = retriever.retrieve("Article 29", top_k=3)
    chunk_ids = [h.chunk_id for h in hits]
    assert any(
        cid.startswith("labour-law-en::art-29") for cid in chunk_ids
    ), f"no Article 29 EN chunk in top-3: {chunk_ids}"


def test_hybrid_returns_article_29_for_semantic_en_query(retriever: RetrievalPort) -> None:
    """Semantic 'annual leave entitlement' lands the Article 29 chunk in the top-5."""
    hits = retriever.retrieve("annual leave entitlement", top_k=5)
    chunk_ids = [h.chunk_id for h in hits]
    assert any(
        cid.startswith("labour-law-en::art-29") for cid in chunk_ids
    ), f"no Article 29 EN chunk in top-5: {chunk_ids}"


def test_hybrid_returns_article_29_for_arabic_query(retriever: RetrievalPort) -> None:
    """Arabic 'المادة 29' surfaces the AR Article 29 chunk in the top-5."""
    hits = retriever.retrieve("المادة 29", top_k=5)
    chunk_ids = [h.chunk_id for h in hits]
    assert any(
        cid.startswith("labour-law-ar::art-29") for cid in chunk_ids
    ), f"no Article 29 AR chunk in top-5: {chunk_ids}"
