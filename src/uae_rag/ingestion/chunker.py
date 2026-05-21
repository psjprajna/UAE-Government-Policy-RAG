"""Heading-aware chunker: turns parsed Articles into embedding-ready Chunks.

Per ADR-0003, each chunk inherits the parent breadcrumb (prepended to the body
before embedding) so retrieval sees the structural context. Articles whose
body exceeds the soft cap are split on paragraph boundaries; sub-chunks share
the parent breadcrumb and append a paragraph-range suffix to the chunk id.

A paragraph that itself exceeds the cap is split on sentence boundaries; the
chunk id then carries an additional ``s{x}-s{y}`` segment. A single sentence
that still exceeds the cap is emitted as one oversize sub-chunk and logged at
WARNING level — rare, surfaces during Phase 8 RAGAS if quality suffers.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from uae_rag.ingestion.parser import Article

logger = logging.getLogger(__name__)

# Default cap used when no count_tokens callable is supplied; matches the
# Phase-2 word-count proxy. Phase-3's build script overrides with the real
# embedder tokenizer and a 500-token cap (e5-large's 512 max minus a buffer).
MAX_CHUNK_WORDS = 600

# Sentence terminators: EN punctuation plus the Arabic question mark (؟).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+")


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


def _default_count_tokens(text: str) -> int:
    return len(text.split())


def chunk_articles(
    articles: list[Article],
    *,
    source_slug: str,
    count_tokens: Callable[[str], int] = _default_count_tokens,
    max_tokens: int = MAX_CHUNK_WORDS,
) -> list[Chunk]:
    """Walk ``articles`` and emit Chunks. Splits oversized bodies paragraph- then sentence-wise."""
    chunks: list[Chunk] = []
    fallback_idx = 0

    for article in articles:
        if article.article_id is None:
            fallback_idx += 1
            chunks.append(_make_fallback_chunk(article, source_slug, fallback_idx))
            continue

        if count_tokens(article.body) <= max_tokens:
            chunks.append(_make_article_chunk(article, source_slug))
            continue

        for sub in _split_body(article, count_tokens, max_tokens):
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
    sentence_range: tuple[int, int] | None,
    body_segment: str,
) -> Chunk:
    suffix = f"#p{para_start}-p{para_end}"
    if sentence_range is not None:
        suffix += f"s{sentence_range[0]}-s{sentence_range[1]}"
    return Chunk(
        chunk_id=f"{source_slug}::art-{article.article_id}{suffix}",
        source_slug=source_slug,
        breadcrumb=article.breadcrumb,
        article_id=article.article_id,
        language=article.language,
        page_start=article.page_start,
        page_end=article.page_end,
        text=f"{article.breadcrumb}\n\n{body_segment}",
        mode="subchunk",
    )


_SubChunk = tuple[int, int, tuple[int, int] | None, str]


def _split_body(
    article: Article,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[_SubChunk]:
    """Greedy-pack paragraphs into ``max_tokens`` segments; fall back to sentences when needed.

    Returns ``(paragraph_start, paragraph_end, sentence_range, segment_text)`` tuples
    where indices are 1-based and inclusive. ``sentence_range`` is None when the
    segment is paragraph-aligned; otherwise it carries the sentence indices within
    the (single) paragraph that was sentence-split.
    """
    paragraphs = [p for p in article.body.split("\n\n") if p.strip()]
    if not paragraphs:
        return [(1, 1, None, article.body)]

    segments: list[_SubChunk] = []
    current: list[str] = []
    current_start = 1
    current_count = 0

    for idx, paragraph in enumerate(paragraphs, start=1):
        paragraph_count = count_tokens(paragraph)

        if paragraph_count > max_tokens:
            if current:
                segments.append((current_start, idx - 1, None, "\n\n".join(current)))
                current = []
                current_count = 0
            segments.extend(_split_paragraph(article, idx, paragraph, count_tokens, max_tokens))
            current_start = idx + 1
            continue

        if current and current_count + paragraph_count > max_tokens:
            segments.append((current_start, idx - 1, None, "\n\n".join(current)))
            current = []
            current_start = idx
            current_count = 0
        current.append(paragraph)
        current_count += paragraph_count

    if current:
        segments.append((current_start, len(paragraphs), None, "\n\n".join(current)))

    return segments


def _split_paragraph(
    article: Article,
    paragraph_idx: int,
    paragraph: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[_SubChunk]:
    """Split one oversize paragraph into sentence-bucketed sub-chunks."""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    if not sentences:
        return [(paragraph_idx, paragraph_idx, None, paragraph)]

    segments: list[_SubChunk] = []
    current: list[str] = []
    current_start = 1
    current_count = 0

    for idx, sentence in enumerate(sentences, start=1):
        sentence_count = count_tokens(sentence)

        if sentence_count > max_tokens:
            if current:
                segments.append(
                    (
                        paragraph_idx,
                        paragraph_idx,
                        (current_start, idx - 1),
                        " ".join(current),
                    )
                )
                current = []
                current_count = 0
            logger.warning(
                "oversize sentence in %s art-%s (%d > %d tokens); emitting single sub-chunk",
                article.language,
                article.article_id,
                sentence_count,
                max_tokens,
            )
            segments.append((paragraph_idx, paragraph_idx, (idx, idx), sentence))
            current_start = idx + 1
            continue

        if current and current_count + sentence_count > max_tokens:
            segments.append(
                (
                    paragraph_idx,
                    paragraph_idx,
                    (current_start, idx - 1),
                    " ".join(current),
                )
            )
            current = []
            current_start = idx
            current_count = 0
        current.append(sentence)
        current_count += sentence_count

    if current:
        segments.append(
            (
                paragraph_idx,
                paragraph_idx,
                (current_start, len(sentences)),
                " ".join(current),
            )
        )

    return segments


__all__ = ["MAX_CHUNK_WORDS", "Chunk", "chunk_articles"]
