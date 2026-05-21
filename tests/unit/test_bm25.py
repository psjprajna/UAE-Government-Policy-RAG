"""Unit tests for BM25 retriever — tokenizer truth table + ranking + edges."""

from __future__ import annotations

import pytest

from uae_rag.ingestion.chunker import Chunk
from uae_rag.retrieval.bm25 import BM25Retriever, tokenize


def _chunk(cid: str, text: str, *, language: str = "en") -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_slug="labour-law-en" if language == "en" else "labour-law-ar",
        breadcrumb=f"Article ({cid})",
        article_id=cid,
        language=language,
        page_start=1,
        page_end=1,
        text=text,
        mode="article",
    )


def test_tokenize_casefolds_ascii() -> None:
    assert tokenize("Annual LEAVE Entitlement") == ["annual", "leave", "entitlement"]


def test_tokenize_strips_arabic_diacritics() -> None:
    """Harakat-bearing form tokenizes identically to its bare form."""
    bare = tokenize("الإجازة السنوية")
    with_harakat = tokenize("اَلْإِجَازَةُ السَّنَوِيَّة")
    assert bare == with_harakat
    assert len(bare) == 2


def test_tokenize_drops_punctuation() -> None:
    assert tokenize("Article 29, leave (paid).") == ["article", "29", "leave", "paid"]


def test_tokenize_keeps_digits_drops_single_chars() -> None:
    """Length-1 ASCII letters are dropped; digits are kept (article numbers matter)."""
    assert tokenize("a 1 I 22 do") == ["1", "22", "do"]


def test_tokenize_handles_mixed_en_ar() -> None:
    tokens = tokenize("Article 29 المادة 29")
    assert "article" in tokens
    assert "29" in tokens
    assert "المادة" in tokens


def test_bm25_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        BM25Retriever([])


def test_bm25_empty_query_returns_empty_list() -> None:
    bm25 = BM25Retriever([_chunk("1", "annual leave entitlement")])
    assert bm25.retrieve("") == []
    assert bm25.retrieve("   ") == []
    assert bm25.retrieve("?!.,") == []


def test_bm25_ranks_exact_term_match_first() -> None:
    """Query 'annual leave' ranks the chunk containing both terms above the one with only 'annual'."""
    chunks = [
        _chunk("1", "annual leave entitlement is thirty days"),
        _chunk("2", "probation period rules"),
        _chunk("3", "annual report on employment trends"),
        _chunk("4", "end of service gratuity calculation"),
        _chunk("5", "leave categories include sick and annual"),
    ]
    bm25 = BM25Retriever(chunks)

    hits = bm25.retrieve("annual leave", top_k=5)

    # Both "annual" and "leave" appear in chunks 1 and 5; chunk 3 has only "annual".
    top_two_ids = {hits[0].chunk_id, hits[1].chunk_id}
    assert top_two_ids == {"1", "5"}
    assert all(h.source == "bm25" for h in hits)
    assert all(h.rank == i + 1 for i, h in enumerate(hits))


def test_bm25_returns_at_most_top_k() -> None:
    chunks = [_chunk(str(i), f"term{i} annual leave") for i in range(10)]
    bm25 = BM25Retriever(chunks)

    hits = bm25.retrieve("annual leave", top_k=3)

    assert len(hits) == 3


def test_bm25_skips_zero_score_hits() -> None:
    """A query with no overlapping terms returns an empty list (not all chunks at score 0)."""
    chunks = [
        _chunk("1", "annual leave entitlement"),
        _chunk("2", "probation period rules"),
    ]
    bm25 = BM25Retriever(chunks)

    hits = bm25.retrieve("completely unrelated terminology", top_k=5)

    assert hits == []


def test_bm25_arabic_query_matches_arabic_chunk() -> None:
    """Symmetric AR tokenization: harakat-stripped query matches harakat-stripped indexed text."""
    chunks = [
        _chunk("ar-29", "المادة 29 الإجازة السنوية ثلاثون يوماً", language="ar"),
        _chunk("ar-30", "المادة 30 ساعات العمل", language="ar"),
    ]
    bm25 = BM25Retriever(chunks)

    hits = bm25.retrieve("اَلْإِجَازَةُ السَّنَوِيَّة", top_k=5)

    assert hits
    assert hits[0].chunk_id == "ar-29"


def test_bm25_hit_carries_chunk_metadata_and_text() -> None:
    bm25 = BM25Retriever(
        [_chunk("1", "annual leave entitlement", language="en")]
    )

    hits = bm25.retrieve("annual leave", top_k=1)

    assert hits[0].text == "annual leave entitlement"
    md = hits[0].metadata
    assert md["source_slug"] == "labour-law-en"
    assert md["language"] == "en"
    assert md["article_id"] == "1"
    assert md["breadcrumb"] == "Article (1)"
