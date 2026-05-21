"""Unit tests for the heading-aware PDF parser.

Tests exercise pure functions (parse_articles + parse_unstructured) against
synthetic Page fixtures. The pdfplumber / pypdfium2 shims are not exercised
here — they're trivial wrappers, manually verified via scripts/preview_chunks.py.
"""

from __future__ import annotations

import pytest

from uae_rag.ingestion.parser import (
    Page,
    parse_articles,
    parse_unstructured,
)
from uae_rag.ingestion.registry import DocumentSource


@pytest.fixture
def en_source() -> DocumentSource:
    return DocumentSource(
        slug="labour-law-en",
        title="UAE Labour Law (EN) — sample",
        language="en",
        url="https://example.test/labour-law-en.pdf",
        local_filename="labour-law-en.pdf",
    )


@pytest.fixture
def ar_source() -> DocumentSource:
    return DocumentSource(
        slug="labour-law-ar",
        title="UAE Labour Law (AR) — sample",
        language="ar",
        url="https://example.test/labour-law-ar.pdf",
        local_filename="labour-law-ar.pdf",
    )


def test_parse_articles_en_single_article(en_source: DocumentSource) -> None:
    pages = [
        Page(number=1, text="Article (1)\nDefinitions\nIn application of the provisions ...\n"),
    ]

    articles = parse_articles(pages, en_source)

    assert len(articles) == 1
    art = articles[0]
    assert art.article_id == "1"
    assert art.breadcrumb == "Article (1)"
    assert art.language == "en"
    assert art.page_start == 1
    assert art.page_end == 1
    assert "In application of the provisions" in art.body


def test_parse_articles_en_handles_flipped_parens(en_source: DocumentSource) -> None:
    """UAE Labour Law PDF uses 'Article )N(' for articles 2-17 (typesetter quirk)."""
    pages = [
        Page(number=1, text="Article )5(\nEmployment of Juveniles\n1. Body text here.\n"),
        Page(number=2, text="Article )8 (\nEmployment Contract\n1. The employment contract ...\n"),
    ]

    articles = parse_articles(pages, en_source)

    assert [a.article_id for a in articles] == ["5", "8"]


def test_parse_articles_en_multiple_articles_with_page_ranges(en_source: DocumentSource) -> None:
    pages = [
        Page(
            number=1,
            text="Preamble text on page 1\nArticle (1)\nDefinitions\nFirst article body.\n",
        ),
        Page(number=2, text="Continues onto page 2.\nMore body.\n"),
        Page(number=3, text="Article (2)\nObjectives\nSecond article body.\n"),
    ]

    articles = parse_articles(pages, en_source)

    assert [a.article_id for a in articles] == ["1", "2"]
    assert articles[0].page_start == 1
    assert articles[0].page_end == 2
    assert articles[1].page_start == 3
    assert articles[1].page_end == 3


def test_parse_articles_en_drops_orphan_heading(en_source: DocumentSource) -> None:
    """A heading immediately followed by another heading has no body and is dropped."""
    pages = [
        Page(number=1, text="Article (1)\nArticle (2)\nReal body for article 2.\n"),
    ]

    articles = parse_articles(pages, en_source)

    assert [a.article_id for a in articles] == ["2"]
    assert "Real body for article 2" in articles[0].body


def test_parse_articles_en_preserves_paragraph_separators(en_source: DocumentSource) -> None:
    pages = [
        Page(
            number=1,
            text="Article (1)\nTitle\nFirst paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n",
        ),
    ]

    articles = parse_articles(pages, en_source)

    assert len(articles) == 1
    assert articles[0].body.count("\n\n") >= 2


def test_parse_articles_ar_uses_pdf_byte_sequence(ar_source: DocumentSource) -> None:
    """AR Labour Law extracted via pypdfium2 has ALIF MEEM LAM ALIF DAL TEH-MARBUTA
    (LAM/MEEM swapped vs canonical المادة) — parser must match the actual byte sequence.
    Breadcrumb uses the canonical Arabic ordering for citation faithfulness.
    """
    article_keyword = "املادة"  # ا م ل ا د ة
    pages = [
        Page(number=1, text=f"{article_keyword})1(\nbody of article 1.\n"),
        Page(number=2, text=f"{article_keyword} )5(\nbody of article 5.\n"),
    ]

    articles = parse_articles(pages, ar_source)

    assert [a.article_id for a in articles] == ["1", "5"]
    assert articles[0].language == "ar"
    # Canonical ordering: ALIF LAM MEEM ALIF DAL TEH-MARBUTA
    assert articles[0].breadcrumb == "المادة (1)"


def test_parse_articles_ar_supports_arabic_indic_digits(ar_source: DocumentSource) -> None:
    article_keyword = "املادة"
    pages = [
        Page(number=1, text=f"{article_keyword})٥(\nbody.\n"),  # Arabic-Indic 5
    ]

    articles = parse_articles(pages, ar_source)

    assert len(articles) == 1
    assert articles[0].article_id == "٥"  # store as-is for citation faithfulness


def test_parse_articles_returns_empty_when_no_headings(en_source: DocumentSource) -> None:
    pages = [
        Page(number=1, text="Some body without any article marker.\nMore text.\n"),
    ]

    assert parse_articles(pages, en_source) == []


def test_parse_articles_skips_blank_pages(en_source: DocumentSource) -> None:
    pages = [
        Page(number=1, text="   \n\n  "),
        Page(number=2, text="Article (1)\nTitle\nbody\n"),
        Page(number=3, text=""),
    ]

    articles = parse_articles(pages, en_source)

    assert [a.article_id for a in articles] == ["1"]
    assert articles[0].page_start == 2


def test_parse_articles_attaches_subtitle_to_body(en_source: DocumentSource) -> None:
    pages = [
        Page(number=1, text="Article (1)\nDefinitions\nIn application of the provisions ...\n"),
    ]

    articles = parse_articles(pages, en_source)

    assert "Definitions" in articles[0].body
    assert "In application of the provisions" in articles[0].body


def test_parse_unstructured_yields_section_chunks(en_source: DocumentSource) -> None:
    """PDFs without article markers (ICP Services Guide) get section-mode articles."""
    long_body = ("word " * 700).strip()
    pages = [Page(number=i, text=long_body) for i in (1, 2)]

    sections = parse_unstructured(pages, en_source, max_words=600)

    assert len(sections) >= 2
    for i, section in enumerate(sections, start=1):
        assert section.article_id is None
        assert section.breadcrumb.startswith(en_source.title)
        assert f"Section {i}" in section.breadcrumb
        assert section.language == "en"
        assert len(section.body.split()) <= 600
