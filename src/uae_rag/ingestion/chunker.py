"""Heading-aware chunker: turns parsed Articles into embedding-ready Chunks.

Per ADR-0003, each chunk inherits the parent breadcrumb (prepended to the body
before embedding) so retrieval sees the structural context. Articles whose
body exceeds MAX_CHUNK_WORDS are split on paragraph boundaries; sub-chunks
share the parent breadcrumb and append a paragraph-range suffix to the chunk id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from uae_rag.ingestion.parser import Article

logger = logging.getLogger(__name__)

# Proxy for ADR-0003's 800-token cap. Word count is a coarse approximation; Phase 3
# will swap to the real embedder tokenizer once intfloat/multilingual-e5-large loads.
MAX_CHUNK_WORDS = 600


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit with deterministic id and breadcrumb-prefixed text."""

    chunk_id: str
    source_slug: str
    breadcrumb: str
    article_id: str | None
    language: str
    page_start: int
    page_end: int
    text: str
    mode: str  # "article" | "subchunk" | "fallback"


def chunk_articles(
    articles: list[Article],
    *,
    source_slug: str,
    max_words: int = MAX_CHUNK_WORDS,
) -> list[Chunk]:
    """Walk ``articles`` and emit Chunks. Splits oversized bodies on paragraph boundaries."""
    chunks: list[Chunk] = []
    fallback_idx = 0

    for article in articles:
        if article.article_id is None:
            fallback_idx += 1
            chunks.append(_make_fallback_chunk(article, source_slug, fallback_idx))
            continue

        words = article.body.split()
        if len(words) <= max_words:
            chunks.append(_make_article_chunk(article, source_slug))
            continue

        for sub in _split_body(article, max_words):
            chunks.append(_make_subchunk(article, source_slug, *sub))

    return chunks


def _make_article_chunk(article: Article, source_slug: str) -> Chunk:
    return Chunk(
        chunk_id=f"{source_slug}::art-{article.article_id}",
        source_slug=source_slug,
        breadcrumb=article.breadcrumb,
        article_id=article.article_id,
        language=article.language,
        page_start=article.page_start,
        page_end=article.page_end,
        text=f"{article.breadcrumb}\n\n{article.body}",
        mode="article",
    )


def _make_fallback_chunk(article: Article, source_slug: str, idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"{source_slug}::sec-{idx}",
        source_slug=source_slug,
        breadcrumb=article.breadcrumb,
        article_id=None,
        language=article.language,
        page_start=article.page_start,
        page_end=article.page_end,
        text=f"{article.breadcrumb}\n\n{article.body}",
        mode="fallback",
    )


def _make_subchunk(
    article: Article,
    source_slug: str,
    para_start: int,
    para_end: int,
    body_segment: str,
) -> Chunk:
    return Chunk(
        chunk_id=f"{source_slug}::art-{article.article_id}#p{para_start}-p{para_end}",
        source_slug=source_slug,
        breadcrumb=article.breadcrumb,
        article_id=article.article_id,
        language=article.language,
        page_start=article.page_start,
        page_end=article.page_end,
        text=f"{article.breadcrumb}\n\n{body_segment}",
        mode="subchunk",
    )


def _split_body(article: Article, max_words: int) -> list[tuple[int, int, str]]:
    """Greedy-pack paragraphs into ``max_words`` segments.

    Returns ``(paragraph_start, paragraph_end, segment_text)`` triples
    where indices are 1-based and inclusive.
    """
    paragraphs = [p for p in article.body.split("\n\n") if p.strip()]
    if not paragraphs:
        return [(1, 1, article.body)]

    segments: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = 1
    current_word_count = 0

    for idx, paragraph in enumerate(paragraphs, start=1):
        words_in_paragraph = len(paragraph.split())
        if current and current_word_count + words_in_paragraph > max_words:
            segments.append((current_start, idx - 1, "\n\n".join(current)))
            current = []
            current_start = idx
            current_word_count = 0
        current.append(paragraph)
        current_word_count += words_in_paragraph

    if current:
        segments.append((current_start, len(paragraphs), "\n\n".join(current)))

    return segments


__all__ = ["MAX_CHUNK_WORDS", "Chunk", "chunk_articles"]
