"""Unit tests for the heading-aware chunker — pure functions over synthetic Articles."""

from __future__ import annotations

import pytest

from uae_rag.ingestion.chunker import MAX_CHUNK_WORDS, Chunk, chunk_articles
from uae_rag.ingestion.parser import Article


@pytest.fixture
def small_article() -> Article:
    return Article(
        article_id="1",
        breadcrumb="Article (1)",
        language="en",
        page_start=2,
        page_end=2,
        body="A short body of fewer than six hundred words.",
    )


@pytest.fixture
def fallback_article() -> Article:
    return Article(
        article_id=None,
        breadcrumb="ICP Services Guide > Section 1",
        language="en",
        page_start=1,
        page_end=3,
        body="word " * 400,
    )


def test_chunk_articles_emits_single_chunk_when_under_cap(small_article: Article) -> None:
    chunks = chunk_articles([small_article], source_slug="labour-law-en")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "labour-law-en::art-1"
    assert chunk.mode == "article"
    assert chunk.source_slug == "labour-law-en"
    assert chunk.article_id == "1"
    assert chunk.text.startswith("Article (1)\n\n")
    assert chunk.page_start == 2
    assert chunk.page_end == 2


def test_chunk_articles_subchunks_when_over_cap() -> None:
    paragraphs = [("word " * 250).strip() for _ in range(5)]  # 5 x 250 = 1250 words
    article = Article(
        article_id="29",
        breadcrumb="Article (29)",
        language="en",
        page_start=10,
        page_end=12,
        body="\n\n".join(paragraphs),
    )

    chunks = chunk_articles([article], source_slug="labour-law-en")

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.mode == "subchunk"
        assert chunk.text.startswith("Article (29)\n\n")
        body_after_breadcrumb = chunk.text.split("\n\n", 1)[1]
        assert len(body_after_breadcrumb.split()) <= MAX_CHUNK_WORDS
    # chunk_id has the paragraph range suffix
    assert chunks[0].chunk_id.startswith("labour-law-en::art-29#p1-p")
    # All sub-chunks share the parent breadcrumb
    assert {c.breadcrumb for c in chunks} == {"Article (29)"}


def test_chunk_articles_fallback_id_shape(fallback_article: Article) -> None:
    chunks = chunk_articles([fallback_article], source_slug="visa-regulations")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "visa-regulations::sec-1"
    assert chunk.mode == "fallback"
    assert chunk.article_id is None
    assert chunk.text.startswith("ICP Services Guide > Section 1\n\n")


def test_chunk_articles_multiple_articles_get_sequential_section_ids() -> None:
    sections = [
        Article(
            article_id=None,
            breadcrumb=f"Doc > Section {i}",
            language="en",
            page_start=i,
            page_end=i,
            body=f"body {i}",
        )
        for i in (1, 2, 3)
    ]

    chunks = chunk_articles(sections, source_slug="visa-regulations")

    ids = [c.chunk_id for c in chunks]
    assert ids == [
        "visa-regulations::sec-1",
        "visa-regulations::sec-2",
        "visa-regulations::sec-3",
    ]


def test_chunk_articles_text_always_starts_with_breadcrumb(small_article: Article) -> None:
    chunks = chunk_articles([small_article], source_slug="labour-law-en")

    for chunk in chunks:
        assert chunk.text.startswith(chunk.breadcrumb + "\n\n")


def test_chunk_articles_is_deterministic(small_article: Article) -> None:
    first = chunk_articles([small_article], source_slug="labour-law-en")
    second = chunk_articles([small_article], source_slug="labour-law-en")

    assert first == second


def test_chunk_articles_subchunks_inherit_parent_page_range() -> None:
    paragraphs = [("w " * 250).strip() for _ in range(4)]
    article = Article(
        article_id="29",
        breadcrumb="Article (29)",
        language="en",
        page_start=10,
        page_end=12,
        body="\n\n".join(paragraphs),
    )

    chunks = chunk_articles([article], source_slug="labour-law-en")

    for chunk in chunks:
        assert chunk.page_start == 10
        assert chunk.page_end == 12


def test_chunk_articles_returns_concrete_chunk_type(small_article: Article) -> None:
    chunks = chunk_articles([small_article], source_slug="labour-law-en")

    assert all(isinstance(c, Chunk) for c in chunks)
