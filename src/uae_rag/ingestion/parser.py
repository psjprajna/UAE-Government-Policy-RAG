"""Heading-aware PDF parser for the UAE Gov Policy RAG corpus.

Hybrid extractor: pdfplumber for English sources, pypdfium2 for Arabic
(pdfplumber returns Arabic in visual/reversed order; pypdfium2 returns
logical order, modulo an alif-lam ligature quirk handled in the regex).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from uae_rag.ingestion.registry import DocumentSource

logger = logging.getLogger(__name__)

# EN: 'Article (N)' (articles 1, 18-74) and 'Article )N(' (articles 2-17 — typesetter quirk).
_EN_ARTICLE_RE = re.compile(r"^\s*Article\s*[()]?\s*(\d+)\s*[()]?\s*$")

# AR: pypdfium2 outputs the article keyword as ALIF MEEM LAM ALIF DAL TEH-MARBUTA
# (LAM/MEEM swapped vs canonical due to alif-lam ligature handling).
_AR_ARTICLE_KEYWORD = "املادة"
_AR_ARTICLE_RE = re.compile(rf"^\s*{_AR_ARTICLE_KEYWORD}\s*[()]?\s*([0-9٠-٩]+)\s*[()]?\s*$")
_AR_BREADCRUMB_PREFIX = "المادة"  # canonical المادة


@dataclass(frozen=True, slots=True)
class Page:
    """One extracted PDF page (1-indexed)."""

    number: int
    text: str


@dataclass(frozen=True, slots=True)
class Article:
    """One parsed article (or fallback section when article_id is None)."""

    article_id: str | None
    breadcrumb: str
    language: str
    page_start: int
    page_end: int
    body: str


def extract_pages(pdf_path: Path, language: str) -> list[Page]:
    """Return non-blank Pages from ``pdf_path``. pdfplumber for EN, pypdfium2 for AR."""
    if language == "ar":
        return _extract_via_pypdfium2(pdf_path)
    return _extract_via_pdfplumber(pdf_path)


def _extract_via_pdfplumber(pdf_path: Path) -> list[Page]:
    import pdfplumber

    pages: list[Page] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=False, x_tolerance=2) or ""
            if text.strip():
                pages.append(Page(number=i, text=text))
    return pages


def _extract_via_pypdfium2(pdf_path: Path) -> list[Page]:
    import pypdfium2 as pdfium

    pages: list[Page] = []
    pdf = pdfium.PdfDocument(pdf_path)
    for i in range(len(pdf)):
        text = pdf[i].get_textpage().get_text_range()
        if text and text.strip():
            pages.append(Page(number=i + 1, text=text))
    return pages


def _classify(line: str, language: str) -> str | None:
    """Return the article id if ``line`` is an article heading; otherwise None."""
    regex = _AR_ARTICLE_RE if language == "ar" else _EN_ARTICLE_RE
    m = regex.match(line)
    return m.group(1) if m else None


def _make_breadcrumb(article_id: str, language: str) -> str:
    prefix = _AR_BREADCRUMB_PREFIX if language == "ar" else "Article"
    return f"{prefix} ({article_id})"


def parse_articles(pages: list[Page], source: DocumentSource) -> list[Article]:
    """Walk pages line-by-line and emit one Article per heading with non-empty body."""
    articles: list[Article] = []
    pending: _Pending | None = None

    for page in pages:
        if not page.text.strip():
            continue
        for line in page.text.splitlines():
            article_id = _classify(line, source.language)
            if article_id is not None:
                if pending is not None and pending.body_lines:
                    articles.append(_finalize(pending, source))
                pending = _Pending(id=article_id, page_start=page.number, page_end=page.number)
            elif pending is not None:
                pending.body_lines.append(line)
                pending.page_end = page.number

    if pending is not None and pending.body_lines:
        articles.append(_finalize(pending, source))

    return articles


@dataclass(slots=True)
class _Pending:
    id: str
    page_start: int
    page_end: int
    body_lines: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.body_lines is None:
            self.body_lines = []


def _finalize(state: _Pending, source: DocumentSource) -> Article:
    body = "\n".join(state.body_lines).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return Article(
        article_id=state.id,
        breadcrumb=_make_breadcrumb(state.id, source.language),
        language=source.language,
        page_start=state.page_start,
        page_end=state.page_end,
        body=body,
    )


def parse_unstructured(
    pages: list[Page], source: DocumentSource, *, max_words: int = 600
) -> list[Article]:
    """Fallback for PDFs without article markers: greedy ~max_words section blocks."""
    sections: list[Article] = []
    buffer: list[str] = []
    buf_page_start: int | None = None
    buf_page_end: int | None = None
    section_idx = 1

    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        if buf_page_start is None:
            buf_page_start = page.number
        buf_page_end = page.number

        for word in text.split():
            buffer.append(word)
            if len(buffer) >= max_words:
                sections.append(
                    _make_section(section_idx, buffer, buf_page_start, buf_page_end, source)
                )
                section_idx += 1
                buffer = []
                buf_page_start = page.number

    if buffer and buf_page_start is not None and buf_page_end is not None:
        sections.append(_make_section(section_idx, buffer, buf_page_start, buf_page_end, source))

    return sections


def _make_section(
    idx: int,
    words: list[str],
    page_start: int,
    page_end: int,
    source: DocumentSource,
) -> Article:
    return Article(
        article_id=None,
        breadcrumb=f"{source.title} > Section {idx}",
        language=source.language,
        page_start=page_start,
        page_end=page_end,
        body=" ".join(words),
    )


def parse(pdf_path: Path, source: DocumentSource) -> list[Article]:
    """Top-level entry: extract pages, try heading-aware parse, fall back to unstructured."""
    pages = extract_pages(pdf_path, source.language)
    if not pages:
        raise ValueError(f"No extractable text in {source.slug} ({pdf_path})")
    articles = parse_articles(pages, source)
    if articles:
        logger.info("parsed %d articles from %s", len(articles), source.slug)
        return articles
    logger.info("no article markers in %s; using unstructured fallback", source.slug)
    return parse_unstructured(pages, source)


__all__ = [
    "Article",
    "Page",
    "extract_pages",
    "parse",
    "parse_articles",
    "parse_unstructured",
]
